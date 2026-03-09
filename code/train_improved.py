#!/usr/bin/env python3
"""
Improved Conspiracy Belief Classifier — Informed by ALL Experimental Insights
=============================================================================

Insights incorporated:
  1. CV Ablation: Single-phase training (+4.4 F1, 10× more stable)
  2. CV Ablation: No layerwise LR decay (+1.7 F1)
  3. CV Ablation: No focal loss (neutral; simpler CE preferred)
  4. CV Ablation: Mean pooling essential (+1.1 over CLS)
  5. CV Ablation: Fine-tuning essential (+6.0 over frozen)
  6. Can't-tell data: Including as Yes improves F1 (0.806 vs 0.791)
  7. Layer probing: Signal peaks at layer 16 → unfreeze more layers
  8. Noise robustness: 5-15% noise HELPS → increase label smoothing
  9. Calibration: ECE=0.20, probs narrow → add temperature scaling
  10. Error analysis: 10FP/4FN, all low-conf → better threshold search
  11. Ensemble: Majority vote gives 0.824 → multi-seed ensemble
  12. Max length: 128 gives 0.819 w/ 18% truncation → try shorter

Pipeline:
  Phase 1: Config search (8 configs × full train → dev)
  Phase 2: 5-fold CV on top 2 configs (statistical validation)
  Phase 3: 5-seed ensemble with best config → weighted vote → submission
"""

import json
import os
import sys
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent  # subtask2/
DATA_DIR = BASE_DIR.parent  # /home/divya/Desktop/Conspiracy/
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
SUBMISSION_DIR = BASE_DIR / "submission"

TRAIN_FILE = DATA_DIR / "train_rehydrated.jsonl"
DEV_FILE = DATA_DIR / "dev_public.jsonl"
TEST_FILE = DATA_DIR / "test_data.jsonl"

MODEL_NAME = "microsoft/deberta-v3-large"
SEED = 2026
NUM_WORKERS = 2

for d in [RESULTS_DIR, MODELS_DIR, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    log_file = RESULTS_DIR / f"improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger('improved')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# DATA
# ============================================================================

def load_data(filepath, filter_ambiguous=True, canttell_as_yes=False):
    """Load JSONL data with options for can't-tell handling."""
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line.strip())
            label = item.get('conspiracy', '').lower().strip()

            if label == "can't tell":
                if canttell_as_yes:
                    item['conspiracy'] = 'Yes'
                elif filter_ambiguous:
                    continue
                else:
                    continue

            data.append(item)
    return data


class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('full_text', item.get('text', ''))
        label = 1 if item.get('conspiracy', '').lower() == 'yes' else 0

        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }


# ============================================================================
# MODEL
# ============================================================================

class ImprovedConspiracyClassifier(nn.Module):
    """
    Mean pooling + MLP with temperature scaling.
    Insight 4: Mean pooling confirmed superior by CV ablation.
    Insight 9: Temperature parameter for calibration.
    """
    def __init__(self, model_name, dropout=0.1, freeze_encoder=False,
                 unfreeze_layers=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 1024

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        # Insight 9: Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))

        # Freeze/unfreeze layers
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            # Freeze all first, then selectively unfreeze
            for p in self.encoder.parameters():
                p.requires_grad = False

            # Unfreeze top N layers
            layers = self._get_encoder_layers()
            n_layers = len(layers)
            for i in range(max(0, n_layers - unfreeze_layers), n_layers):
                for p in layers[i].parameters():
                    p.requires_grad = True

            # Note: keep embeddings frozen to save VRAM
            # (layer probing shows signal is in layers 16+, not embeddings)

    def _get_encoder_layers(self):
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            return self.encoder.encoder.layer
        elif hasattr(self.encoder, 'layers'):
            return self.encoder.layers
        return []

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        # Mean pooling over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(pooled)

        # Temperature scaling for better calibration
        return logits / self.temperature.clamp(min=0.01)

    def get_temperature(self):
        return self.temperature.item()


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion, device,
                grad_accum=1, max_grad_norm=1.0, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # FP16 mixed precision to save VRAM
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    """Evaluate with fine-grained threshold search."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits = model(input_ids, attention_mask)
            probs = F.softmax(logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # Insight 10: Fine-grained threshold search (errors are all near 0.5)
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.25, 0.75, 0.005):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return probs, labels, best_t, best_f1


def train_single_phase(model, train_loader, val_loader, device, config,
                       logger=None):
    """
    Single-phase joint training — the clear winner from CV ablation.

    Insight 1: Single-phase is +4.4 CV F1, 10× more stable (σ=0.002 vs 0.023)
    Insight 2: Uniform LR confirmed +1.7 CV F1
    Insight 3: CE loss (no focal)
    Insight 8: Higher label smoothing from noise robustness analysis
    """
    lr = config.get('lr', 2e-5)
    epochs = config.get('epochs', 9)
    dropout = config.get('dropout', 0.1)
    grad_accum = config.get('grad_accum', 2)
    patience = config.get('patience', 4)
    label_smoothing = config.get('label_smoothing', 0.05)
    weight_decay = config.get('weight_decay', 0.01)

    # Insight 3: Plain CE with label smoothing (no focal loss)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Insight 2: Uniform LR for all parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )

    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # FP16 mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_f1, best_threshold = 0, 0.5
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        t0 = time.time()
        loss, acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, grad_accum, scaler=scaler
        )
        probs, labels, threshold, f1 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        msg = (f"  Epoch {epoch+1}/{epochs}: loss={loss:.4f} acc={acc:.4f} "
               f"val_f1={f1:.4f} thresh={threshold:.3f} "
               f"temp={model.get_temperature():.3f} ({elapsed:.0f}s)")
        if logger:
            logger.info(msg)
        else:
            print(msg)

        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if logger:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_f1, best_threshold


# ============================================================================
# CONFIGURATION SEARCH
# ============================================================================

# Each config is informed by specific experimental insights

SEARCH_CONFIGS = {
    # Baseline: our current best CV config (no_two_phase)
    'A_ablation_best': {
        'desc': 'CV-best: single-phase + layerwise LR',
        'epochs': 9, 'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2,
        'patience': 4, 'label_smoothing': 0.05, 'weight_decay': 0.01,
        'unfreeze_layers': 6, 'max_length': 256,
        'canttell_as_yes': False, 'batch_size': 16,
    },

    # Combine ablation insights: no two-phase + no layerwise + no focal
    'B_combined_ablation': {
        'desc': 'Combined: single + uniform LR + CE',
        'epochs': 9, 'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2,
        'patience': 4, 'label_smoothing': 0.05, 'weight_decay': 0.01,
        'unfreeze_layers': 6, 'max_length': 256,
        'canttell_as_yes': False, 'batch_size': 16,
    },

    # Insight 7: Unfreeze more layers to capture layer-16 peak
    'C_more_layers': {
        'desc': 'B + unfreeze 12 layers (captures layer 16 peak)',
        'epochs': 9, 'lr': 1.5e-5, 'dropout': 0.1, 'grad_accum': 2,
        'patience': 4, 'label_smoothing': 0.05, 'weight_decay': 0.01,
        'unfreeze_layers': 12, 'max_length': 256,
        'canttell_as_yes': False, 'batch_size': 8,  # less VRAM
    },

    # Insight 8: Higher label smoothing (noise robustness showed 5-15% noise helps)
    'D_high_smoothing': {
        'desc': 'B + label_smoothing=0.15 (from noise insight)',
        'epochs': 9, 'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2,
        'patience': 4, 'label_smoothing': 0.15, 'weight_decay': 0.01,
        'unfreeze_layers': 6, 'max_length': 256,
        'canttell_as_yes': False, 'batch_size': 16,
    },

    # Insight 6: Include can't-tell as Yes (+785 samples, was F1=0.806)
    'E_canttell_yes': {
        'desc': 'B + can\'t-tell as Yes (more data)',
        'epochs': 9, 'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2,
        'patience': 4, 'label_smoothing': 0.05, 'weight_decay': 0.01,
        'unfreeze_layers': 6, 'max_length': 256,
        'canttell_as_yes': True, 'batch_size': 16,
    },

    # Insight 12: Shorter max length (128 gave 0.819 on dev)
    'F_short_length': {
        'desc': 'B + max_length=128 (focus on key signal)',
        'epochs': 12, 'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2,
        'patience': 4, 'label_smoothing': 0.05, 'weight_decay': 0.01,
        'unfreeze_layers': 6, 'max_length': 128,
        'canttell_as_yes': False, 'batch_size': 16,
    },

    # Kitchen sink: combine all promising insights
    'G_kitchen_sink': {
        'desc': 'All insights: +layers, +smooth, +canttell, +dropout',
        'epochs': 12, 'lr': 1.5e-5, 'dropout': 0.15, 'grad_accum': 2,
        'patience': 5, 'label_smoothing': 0.10, 'weight_decay': 0.02,
        'unfreeze_layers': 12, 'max_length': 256,
        'canttell_as_yes': True, 'batch_size': 8,
    },

    # Conservative improvement: just the safest changes
    'H_conservative': {
        'desc': 'Safe: single-phase + smoothing=0.10 + canttell',
        'epochs': 9, 'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2,
        'patience': 4, 'label_smoothing': 0.10, 'weight_decay': 0.01,
        'unfreeze_layers': 6, 'max_length': 256,
        'canttell_as_yes': True, 'batch_size': 16,
    },
}


# ============================================================================
# PHASE 1: CONFIG SEARCH
# ============================================================================

def run_config_search(logger):
    """Train each config on full train → evaluate on dev."""
    logger.info("=" * 70)
    logger.info("PHASE 1: CONFIGURATION SEARCH (8 configs)")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load dev data (always filter can't-tell for evaluation)
    dev_data = load_data(DEV_FILE)
    dev_dataset = SimpleDataset(dev_data, tokenizer, 256)
    dev_loader = DataLoader(dev_dataset, batch_size=16, num_workers=NUM_WORKERS)

    results = {}
    results_file = RESULTS_DIR / 'improved_search.json'

    # Load checkpoint if exists
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"Loaded partial results: {list(results.keys())}")

    for name, config in SEARCH_CONFIGS.items():
        if name in results:
            logger.info(f"\n  {name}: already done (F1={results[name]['macro_f1']:.4f})")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Config: {name} — {config['desc']}")
        logger.info(f"{'='*50}")

        set_seed(SEED)
        torch.cuda.empty_cache()

        # Load train data with config-specific options
        train_data = load_data(
            TRAIN_FILE,
            canttell_as_yes=config.get('canttell_as_yes', False)
        )
        logger.info(f"  Train samples: {len(train_data)}")

        max_length = config.get('max_length', 256)
        batch_size = config.get('batch_size', 16)

        train_dataset = SimpleDataset(train_data, tokenizer, max_length)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=NUM_WORKERS
        )

        # Dev loader with matching max_length
        dev_dataset_ml = SimpleDataset(dev_data, tokenizer, max_length)
        dev_loader_ml = DataLoader(
            dev_dataset_ml, batch_size=batch_size,
            num_workers=NUM_WORKERS
        )

        # Create model
        model = ImprovedConspiracyClassifier(
            MODEL_NAME,
            dropout=config.get('dropout', 0.1),
            freeze_encoder=False,
            unfreeze_layers=config.get('unfreeze_layers', 6)
        ).to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Params: {trainable:,} trainable / {total_params:,} total")

        t0 = time.time()

        try:
            best_f1, best_threshold = train_single_phase(
                model, train_loader, dev_loader_ml, device, config, logger
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.info(f"  OOM! Retrying with batch_size=4...")
                torch.cuda.empty_cache()
                del model

                train_loader = DataLoader(
                    train_dataset, batch_size=4,
                    shuffle=True, num_workers=NUM_WORKERS
                )
                dev_loader_ml = DataLoader(
                    dev_dataset_ml, batch_size=4,
                    num_workers=NUM_WORKERS
                )

                set_seed(SEED)
                model = ImprovedConspiracyClassifier(
                    MODEL_NAME,
                    dropout=config.get('dropout', 0.1),
                    freeze_encoder=False,
                    unfreeze_layers=config.get('unfreeze_layers', 6)
                ).to(device)

                config_retry = dict(config)
                config_retry['grad_accum'] = 4  # maintain effective batch
                best_f1, best_threshold = train_single_phase(
                    model, train_loader, dev_loader_ml, device,
                    config_retry, logger
                )
            else:
                logger.info(f"  ERROR: {e}")
                results[name] = {'macro_f1': 0, 'error': str(e)}
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                del model
                torch.cuda.empty_cache()
                continue

        elapsed = time.time() - t0

        # Full eval with best model
        probs, labels, threshold, f1 = evaluate(model, dev_loader_ml, device)
        preds = (probs >= threshold).astype(int)
        p_mac, r_mac, _, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        results[name] = {
            'desc': config['desc'],
            'macro_f1': round(float(f1), 4),
            'precision': round(float(p_mac), 4),
            'recall': round(float(r_mac), 4),
            'accuracy': round(float(acc), 4),
            'threshold': round(float(threshold), 3),
            'temperature': round(model.get_temperature(), 4),
            'confusion_matrix': cm.tolist(),
            'train_time_sec': round(elapsed, 1),
            'config': {k: v for k, v in config.items()
                      if not isinstance(v, (torch.Tensor,))},
        }

        logger.info(f"  → F1={f1:.4f} P={p_mac:.4f} R={r_mac:.4f} "
                    f"thresh={threshold:.3f} temp={model.get_temperature():.3f} "
                    f"({elapsed:.0f}s)")
        logger.info(f"  CM: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

        # Save checkpoint
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1 RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    ranked = sorted(
        [(k, v.get('macro_f1', 0)) for k, v in results.items()],
        key=lambda x: x[1], reverse=True
    )
    for rank, (name, f1) in enumerate(ranked, 1):
        marker = " ← BEST" if rank == 1 else ""
        logger.info(f"  {rank}. {name}: F1={f1:.4f}{marker}")

    return results, ranked[0][0]  # results + best config name


# ============================================================================
# PHASE 2: CV VALIDATION OF TOP CONFIGS
# ============================================================================

def run_cv_validation(logger, results, top_n=3):
    """5-fold CV on top configs to confirm they generalize."""
    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 2: CV VALIDATION (top {top_n} configs × 5 folds)")
    logger.info(f"{'='*70}")

    ranked = sorted(
        [(k, v.get('macro_f1', 0)) for k, v in results.items()
         if isinstance(v, dict) and 'macro_f1' in v],
        key=lambda x: x[1], reverse=True
    )
    top_configs = [name for name, _ in ranked[:top_n]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    cv_results = {}
    cv_file = RESULTS_DIR / 'improved_cv.json'
    if cv_file.exists():
        with open(cv_file) as f:
            cv_results = json.load(f)

    for config_name in top_configs:
        if config_name in cv_results:
            s = cv_results[config_name]['summary']
            logger.info(f"\n  {config_name}: already done "
                       f"(CV F1={s['mean_f1']:.4f} ± {s['std_f1']:.4f})")
            continue

        config = SEARCH_CONFIGS[config_name]
        logger.info(f"\n  CV for {config_name}: {config['desc']}")

        train_data = load_data(
            TRAIN_FILE,
            canttell_as_yes=config.get('canttell_as_yes', False)
        )
        labels_arr = np.array([
            1 if item.get('conspiracy', '').lower() == 'yes' else 0
            for item in train_data
        ])

        max_length = config.get('max_length', 256)
        batch_size = config.get('batch_size', 16)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_f1s = []
        fold_results = {}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_data, labels_arr)):
            fold_key = f"fold_{fold_idx}"
            logger.info(f"    Fold {fold_idx+1}/5...")

            set_seed(SEED + fold_idx)
            torch.cuda.empty_cache()

            fold_train = [train_data[i] for i in train_idx]
            fold_val = [train_data[i] for i in val_idx]

            train_dataset = SimpleDataset(fold_train, tokenizer, max_length)
            val_dataset = SimpleDataset(fold_val, tokenizer, max_length)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=NUM_WORKERS
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                num_workers=NUM_WORKERS
            )

            model = ImprovedConspiracyClassifier(
                MODEL_NAME,
                dropout=config.get('dropout', 0.1),
                freeze_encoder=False,
                unfreeze_layers=config.get('unfreeze_layers', 6)
            ).to(device)

            t0 = time.time()
            try:
                best_f1, best_thresh = train_single_phase(
                    model, train_loader, val_loader, device, config, logger
                )
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    del model
                    train_loader = DataLoader(
                        train_dataset, batch_size=4,
                        shuffle=True, num_workers=NUM_WORKERS
                    )
                    val_loader = DataLoader(
                        val_dataset, batch_size=4,
                        num_workers=NUM_WORKERS
                    )
                    set_seed(SEED + fold_idx)
                    model = ImprovedConspiracyClassifier(
                        MODEL_NAME,
                        dropout=config.get('dropout', 0.1),
                        freeze_encoder=False,
                        unfreeze_layers=config.get('unfreeze_layers', 6)
                    ).to(device)
                    config_retry = dict(config)
                    config_retry['grad_accum'] = 4
                    best_f1, best_thresh = train_single_phase(
                        model, train_loader, val_loader, device,
                        config_retry, logger
                    )
                else:
                    raise

            elapsed = time.time() - t0
            fold_f1s.append(best_f1)
            fold_results[fold_key] = {
                'macro_f1': round(float(best_f1), 4),
                'threshold': round(float(best_thresh), 3),
                'train_time_sec': round(elapsed, 1),
            }
            logger.info(f"    Fold {fold_idx+1}: F1={best_f1:.4f} ({elapsed:.0f}s)")

            del model
            torch.cuda.empty_cache()

        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        cv_results[config_name] = {
            'folds': fold_results,
            'summary': {
                'mean_f1': round(mean_f1, 4),
                'std_f1': round(std_f1, 4),
                'fold_f1s': [round(f, 4) for f in fold_f1s],
            }
        }

        with open(cv_file, 'w') as f:
            json.dump(cv_results, f, indent=2)

        logger.info(f"  {config_name} CV: {mean_f1:.4f} ± {std_f1:.4f}")

    # Summary
    logger.info(f"\n  CV SUMMARY:")
    cv_ranked = sorted(
        [(k, v['summary']['mean_f1'], v['summary']['std_f1'])
         for k, v in cv_results.items()],
        key=lambda x: x[1], reverse=True
    )
    best_cv_name = cv_ranked[0][0]
    for name, mean, std in cv_ranked:
        marker = " ← BEST" if name == best_cv_name else ""
        logger.info(f"    {name}: {mean:.4f} ± {std:.4f}{marker}")

    return cv_results, best_cv_name


# ============================================================================
# PHASE 3: MULTI-SEED ENSEMBLE
# ============================================================================

def run_ensemble(logger, best_config_name, seeds=None):
    """Train best config with multiple seeds, create ensemble for submission."""
    if seeds is None:
        seeds = [2026, 42, 1337, 7, 2024]

    config = SEARCH_CONFIGS[best_config_name]

    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 3: MULTI-SEED ENSEMBLE ({len(seeds)} seeds)")
    logger.info(f"Config: {best_config_name} — {config['desc']}")
    logger.info(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    max_length = config.get('max_length', 256)
    batch_size = config.get('batch_size', 16)

    train_data = load_data(
        TRAIN_FILE,
        canttell_as_yes=config.get('canttell_as_yes', False)
    )
    dev_data = load_data(DEV_FILE)

    train_dataset = SimpleDataset(train_data, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=NUM_WORKERS
    )
    dev_dataset = SimpleDataset(dev_data, tokenizer, max_length)
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size,
        num_workers=NUM_WORKERS
    )

    ensemble_results = {}
    ensemble_file = RESULTS_DIR / 'improved_ensemble.json'
    all_dev_probs = {}

    for seed in seeds:
        seed_key = str(seed)
        logger.info(f"\n  Seed {seed}...")

        set_seed(seed)
        torch.cuda.empty_cache()

        model = ImprovedConspiracyClassifier(
            MODEL_NAME,
            dropout=config.get('dropout', 0.1),
            freeze_encoder=False,
            unfreeze_layers=config.get('unfreeze_layers', 6)
        ).to(device)

        t0 = time.time()
        try:
            best_f1, best_threshold = train_single_phase(
                model, train_loader, dev_loader, device, config, logger
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                del model
                train_loader_sm = DataLoader(
                    train_dataset, batch_size=4,
                    shuffle=True, num_workers=NUM_WORKERS
                )
                dev_loader_sm = DataLoader(
                    dev_dataset, batch_size=4,
                    num_workers=NUM_WORKERS
                )
                set_seed(seed)
                model = ImprovedConspiracyClassifier(
                    MODEL_NAME,
                    dropout=config.get('dropout', 0.1),
                    freeze_encoder=False,
                    unfreeze_layers=config.get('unfreeze_layers', 6)
                ).to(device)
                config_retry = dict(config)
                config_retry['grad_accum'] = 4
                best_f1, best_threshold = train_single_phase(
                    model, train_loader_sm, dev_loader_sm, device,
                    config_retry, logger
                )
            else:
                raise

        elapsed = time.time() - t0

        # Get dev probs for ensemble
        probs, labels, threshold, f1 = evaluate(model, dev_loader, device)
        preds = (probs >= threshold).astype(int)
        p_mac, r_mac, _, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )

        all_dev_probs[seed_key] = probs.tolist()

        ensemble_results[seed_key] = {
            'macro_f1': round(float(f1), 4),
            'precision': round(float(p_mac), 4),
            'recall': round(float(r_mac), 4),
            'threshold': round(float(threshold), 3),
            'temperature': round(model.get_temperature(), 4),
            'train_time_sec': round(elapsed, 1),
        }

        # Save model
        model_path = MODELS_DIR / f'improved_seed{seed}.pt'
        torch.save({
            'model_state_dict': {k: v.cpu().clone()
                                for k, v in model.state_dict().items()},
            'threshold': best_threshold,
            'val_f1': best_f1,
            'seed': seed,
            'config_name': best_config_name,
            'config': config,
        }, model_path)

        logger.info(f"  Seed {seed}: F1={f1:.4f} thresh={threshold:.3f} "
                    f"temp={model.get_temperature():.3f} ({elapsed:.0f}s)")

        del model
        torch.cuda.empty_cache()

    # Ensemble evaluation on dev
    dev_labels = labels  # same for all seeds
    seed_f1s = [v['macro_f1'] for v in ensemble_results.values()]

    # Average probability ensemble
    avg_probs = np.mean([np.array(p) for p in all_dev_probs.values()], axis=0)
    best_ens_f1, best_ens_t = 0, 0.5
    for t in np.arange(0.25, 0.75, 0.005):
        preds = (avg_probs >= t).astype(int)
        f1 = f1_score(dev_labels, preds, average='macro', zero_division=0)
        if f1 > best_ens_f1:
            best_ens_f1, best_ens_t = f1, t

    # Majority vote ensemble (per-seed threshold)
    per_seed_preds = []
    for seed_key in all_dev_probs:
        thresh = ensemble_results[seed_key]['threshold']
        seed_probs = np.array(all_dev_probs[seed_key])
        per_seed_preds.append((seed_probs >= thresh).astype(int))

    vote_preds = (np.mean(per_seed_preds, axis=0) >= 0.5).astype(int)
    vote_f1 = f1_score(dev_labels, vote_preds, average='macro', zero_division=0)

    # Weighted average by individual F1
    weights = np.array(seed_f1s) / np.sum(seed_f1s)
    weighted_probs = np.average(
        [np.array(p) for p in all_dev_probs.values()],
        axis=0, weights=weights
    )
    best_wt_f1, best_wt_t = 0, 0.5
    for t in np.arange(0.25, 0.75, 0.005):
        preds = (weighted_probs >= t).astype(int)
        f1 = f1_score(dev_labels, preds, average='macro', zero_division=0)
        if f1 > best_wt_f1:
            best_wt_f1, best_wt_t = f1, t

    ensemble_summary = {
        'individual_seeds': ensemble_results,
        'individual_mean_f1': round(float(np.mean(seed_f1s)), 4),
        'individual_std_f1': round(float(np.std(seed_f1s)), 4),
        'avg_prob_ensemble_f1': round(float(best_ens_f1), 4),
        'avg_prob_threshold': round(float(best_ens_t), 3),
        'majority_vote_f1': round(float(vote_f1), 4),
        'weighted_avg_f1': round(float(best_wt_f1), 4),
        'weighted_avg_threshold': round(float(best_wt_t), 3),
        'best_method': 'unknown',
        'config_name': best_config_name,
        'config': config,
    }

    # Pick best ensemble method
    methods = {
        'avg_prob': (best_ens_f1, best_ens_t, avg_probs),
        'majority_vote': (vote_f1, 0.5, None),
        'weighted_avg': (best_wt_f1, best_wt_t, weighted_probs),
    }
    best_method = max(methods.items(), key=lambda x: x[1][0])
    ensemble_summary['best_method'] = best_method[0]
    ensemble_summary['best_ensemble_f1'] = round(float(best_method[1][0]), 4)

    with open(ensemble_file, 'w') as f:
        json.dump(ensemble_summary, f, indent=2)

    logger.info(f"\n  ENSEMBLE RESULTS:")
    logger.info(f"    Individual mean: {np.mean(seed_f1s):.4f} ± {np.std(seed_f1s):.4f}")
    logger.info(f"    Avg prob ensemble: F1={best_ens_f1:.4f} (t={best_ens_t:.3f})")
    logger.info(f"    Majority vote:     F1={vote_f1:.4f}")
    logger.info(f"    Weighted avg:      F1={best_wt_f1:.4f} (t={best_wt_t:.3f})")
    logger.info(f"    Best method: {best_method[0]} → F1={best_method[1][0]:.4f}")

    return ensemble_summary, all_dev_probs


# ============================================================================
# PHASE 4: GENERATE SUBMISSION
# ============================================================================

def generate_submission(logger, best_config_name, ensemble_summary,
                       all_dev_probs):
    """Generate test predictions using the ensemble of saved models."""

    if not TEST_FILE.exists():
        logger.info(f"\n  Test file not found: {TEST_FILE}")
        logger.info("  Skipping submission generation.")
        return

    config = SEARCH_CONFIGS[best_config_name]
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 4: GENERATING SUBMISSION")
    logger.info(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    max_length = config.get('max_length', 256)

    test_data = load_data(TEST_FILE, filter_ambiguous=False)
    test_dataset = SimpleDataset(test_data, tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset, batch_size=16, num_workers=NUM_WORKERS
    )

    logger.info(f"  Test samples: {len(test_data)}")

    all_test_probs = []
    seeds = [2026, 42, 1337, 7, 2024]

    for seed in seeds:
        model_path = MODELS_DIR / f'improved_seed{seed}.pt'
        if not model_path.exists():
            logger.info(f"  Model for seed {seed} not found, skipping.")
            continue

        checkpoint = torch.load(model_path, map_location=device)
        model = ImprovedConspiracyClassifier(
            MODEL_NAME,
            dropout=config.get('dropout', 0.1),
            freeze_encoder=False,
            unfreeze_layers=config.get('unfreeze_layers', 6)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        probs = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits = model(input_ids, attention_mask)
                p = F.softmax(logits, dim=-1)[:, 1]
                probs.extend(p.cpu().numpy())

        all_test_probs.append(np.array(probs))
        del model
        torch.cuda.empty_cache()

    if not all_test_probs:
        logger.info("  No models found! Cannot generate submission.")
        return

    # Average probabilities
    avg_test_probs = np.mean(all_test_probs, axis=0)

    # Use best ensemble threshold
    best_method = ensemble_summary.get('best_method', 'avg_prob')
    if best_method == 'avg_prob':
        threshold = ensemble_summary['avg_prob_threshold']
    elif best_method == 'weighted_avg':
        threshold = ensemble_summary['weighted_avg_threshold']
    else:
        threshold = 0.5  # majority vote

    test_preds = (avg_test_probs >= threshold).astype(int)

    # Save submission
    predictions = []
    for i, item in enumerate(test_data):
        predictions.append({
            '_id': item.get('_id', str(i)),
            'conspiracy': 'Yes' if test_preds[i] == 1 else 'No'
        })

    submission_path = SUBMISSION_DIR / 'submission_improved.jsonl'
    with open(submission_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    logger.info(f"  Threshold: {threshold:.3f}")
    logger.info(f"  Predictions: {Counter(test_preds)}")
    logger.info(f"  Saved to: {submission_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Improved Conspiracy Classifier')
    parser.add_argument('--phase', type=int, default=0,
                       help='1=search only, 2=+CV, 3=+ensemble, 0=all')
    parser.add_argument('--best-config', type=str, default=None,
                       help='Skip search, use this config for ensemble')
    parser.add_argument('--seeds', type=str, default='2026,42,1337,7,2024',
                       help='Seeds for ensemble (comma-separated)')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("IMPROVED CONSPIRACY BELIEF CLASSIFIER")
    logger.info("Informed by: CV ablation, layer probing, noise robustness,")
    logger.info("  error analysis, calibration, can't-tell, max-length experiments")
    logger.info(f"Start: {datetime.now()}")
    logger.info("=" * 70)

    seeds = [int(s) for s in args.seeds.split(',')]
    t_total = time.time()

    if args.best_config:
        # Skip search, go straight to ensemble
        best_config_name = args.best_config
        logger.info(f"Using specified config: {best_config_name}")
        search_results = None
    else:
        # Phase 1: Config search
        search_results, best_config_name = run_config_search(logger)

        if args.phase == 1:
            logger.info(f"\nBest config: {best_config_name}")
            logger.info(f"Total time: {(time.time()-t_total)/60:.1f}min")
            return

        # Phase 2: CV validation
        cv_results, best_cv_name = run_cv_validation(
            logger, search_results, top_n=3
        )
        best_config_name = best_cv_name
        logger.info(f"\nBest CV config: {best_config_name}")

        if args.phase == 2:
            logger.info(f"Total time: {(time.time()-t_total)/60:.1f}min")
            return

    # Phase 3: Multi-seed ensemble
    ensemble_summary, all_dev_probs = run_ensemble(
        logger, best_config_name, seeds=seeds
    )

    # Phase 4: Generate submission
    generate_submission(logger, best_config_name, ensemble_summary,
                       all_dev_probs)

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Best config: {best_config_name}")
    logger.info(f"  {SEARCH_CONFIGS[best_config_name]['desc']}")
    logger.info(f"Individual mean F1: "
               f"{ensemble_summary['individual_mean_f1']:.4f} ± "
               f"{ensemble_summary['individual_std_f1']:.4f}")
    logger.info(f"Best ensemble F1: {ensemble_summary['best_ensemble_f1']:.4f}")
    logger.info(f"Best method: {ensemble_summary['best_method']}")
    logger.info(f"Total time: {(time.time()-t_total)/60:.1f} minutes")
    logger.info("DONE")


if __name__ == '__main__':
    main()
