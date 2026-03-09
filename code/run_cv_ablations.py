#!/usr/bin/env python3
"""
CV Ablation Experiments for SemEval 2026 Task 10 — Subtask 2
=============================================================
Runs all ablation configs under 5-fold stratified CV on train set (n=3,531).
This validates that mean pooling is the only component that matters,
with proper statistical support (mean ± std across folds).

Also runs:
  - Multi-seed experiment (5 seeds × full system on dev)
  - Minimal system (CE + single-phase + uniform LR + mean pooling)

Estimated time: ~3 hours on RTX 4080 Super
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
    precision_recall_fscore_support, accuracy_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent  # subtask2/
CODE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent  # /home/divya/Desktop/Conspiracy/
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

TRAIN_FILE = DATA_DIR / "train_rehydrated.jsonl"
DEV_FILE = DATA_DIR / "dev_public.jsonl"

SEED = 2026
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 256
BATCH_SIZE = 16
NUM_WORKERS = 2


def setup_logging():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = RESULTS_DIR / f"cv_ablations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATA
# ============================================================================

def load_data(filepath, filter_ambiguous=True):
    data = []
    with open(filepath) as f:
        for line in f:
            d = json.loads(line.strip())
            label = d.get('conspiracy', '').lower()
            if filter_ambiguous and label not in ('yes', 'no'):
                continue
            d['label'] = 1 if label == 'yes' else 0
            data.append(d)
    return data


class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', item.get('full_text', ''))
        encoding = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


# ============================================================================
# MODELS
# ============================================================================

class SimpleConspiracyClassifier(nn.Module):
    """Mean pooling + MLP classifier"""
    def __init__(self, model_name, dropout=0.1, freeze_encoder=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over non-padding tokens
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)


class CLSConspiracyClassifier(nn.Module):
    """[CLS] token + MLP classifier"""
    def __init__(self, model_name, dropout=0.1, freeze_encoder=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_hidden)


# ============================================================================
# LOSS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none',
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ============================================================================
# LR HELPERS
# ============================================================================

def get_layerwise_lr_groups(model, base_lr, lr_decay, finetune_layers=6):
    """Layerwise learning rate decay for encoder layers."""
    encoder = model.encoder
    classifier = model.classifier

    params = []
    # Classifier gets full LR
    params.append({'params': list(classifier.parameters()), 'lr': base_lr})

    # Encoder layers (top finetune_layers only)
    if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
        layers = encoder.encoder.layer
    elif hasattr(encoder, 'layers'):
        layers = encoder.layers
    else:
        return params

    n_layers = len(layers)
    for i in range(n_layers - 1, max(n_layers - finetune_layers - 1, -1), -1):
        depth = n_layers - 1 - i
        lr = base_lr * (lr_decay ** (depth + 1))
        params.append({'params': list(layers[i].parameters()), 'lr': lr})

    return params


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_accum=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels) / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # Find best threshold
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.3, 0.7, 0.005):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return probs, labels, best_t, best_f1


def train_full_model(model, train_loader, val_loader, device, config, logger=None):
    """Train model with given config. Returns (best_f1, best_threshold, epoch_log)."""
    use_focal = config.get('use_focal', True)
    focal_gamma = config.get('focal_gamma', 2.0)
    label_smoothing = config.get('label_smoothing', 0.05)
    two_phase = config.get('two_phase', True)
    warmup_epochs = config.get('warmup_epochs', 5)
    finetune_epochs = config.get('finetune_epochs', 4)
    finetune_layers = config.get('finetune_layers', 6)
    finetune_lr_mult = config.get('finetune_lr_mult', 0.2)
    use_layerwise_lr = config.get('use_layerwise_lr', True)
    lr_decay = config.get('lr_decay', 0.9)
    lr = config.get('lr', 2e-5)
    dropout = config.get('dropout', 0.1)
    grad_accum = config.get('grad_accum', 2)
    patience = config.get('patience', 3)
    weight_decay = config.get('weight_decay', 0.01)

    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    epoch_log = []
    best_f1, best_threshold = 0, 0.5
    best_state = None

    if two_phase:
        # --- Phase 1: Frozen encoder, train classifier only ---
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay
        )
        total_steps = warmup_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        no_improve = 0
        for epoch in range(warmup_epochs):
            t0 = time.time()
            loss, acc = train_epoch(model, train_loader, optimizer, scheduler,
                                    criterion, device, grad_accum)
            _, _, threshold, f1 = evaluate(model, val_loader, device)

            epoch_log.append({
                'phase': 1, 'epoch': epoch + 1,
                'loss': loss, 'acc': acc, 'val_f1': f1, 'threshold': threshold,
                'time': time.time() - t0
            })

            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # --- Phase 2: Unfreeze top layers ---
        if finetune_epochs > 0:
            # Unfreeze top layers
            encoder = model.encoder
            if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
                layers = encoder.encoder.layer
            elif hasattr(encoder, 'layers'):
                layers = encoder.layers
            else:
                layers = []

            n_layers = len(layers)
            for i in range(max(0, n_layers - finetune_layers), n_layers):
                for p in layers[i].parameters():
                    p.requires_grad = True

            finetune_lr = lr * finetune_lr_mult
            if use_layerwise_lr:
                param_groups = get_layerwise_lr_groups(
                    model, finetune_lr, lr_decay, finetune_layers
                )
            else:
                param_groups = [
                    {'params': [p for p in model.parameters() if p.requires_grad],
                     'lr': finetune_lr}
                ]

            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
            total_steps = finetune_epochs * len(train_loader)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(total_steps * 0.1),
                num_training_steps=total_steps
            )

            no_improve = 0
            for epoch in range(finetune_epochs):
                t0 = time.time()
                loss, acc = train_epoch(model, train_loader, optimizer, scheduler,
                                        criterion, device, grad_accum)
                _, _, threshold, f1 = evaluate(model, val_loader, device)

                epoch_log.append({
                    'phase': 2, 'epoch': epoch + 1,
                    'loss': loss, 'acc': acc, 'val_f1': f1, 'threshold': threshold,
                    'time': time.time() - t0
                })

                if f1 > best_f1:
                    best_f1, best_threshold = f1, threshold
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

    else:
        # --- Single-phase training ---
        # Unfreeze top layers immediately
        encoder = model.encoder
        if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
            layers = encoder.encoder.layer
        elif hasattr(encoder, 'layers'):
            layers = encoder.layers
        else:
            layers = []

        n_layers = len(layers)
        for i in range(max(0, n_layers - finetune_layers), n_layers):
            for p in layers[i].parameters():
                p.requires_grad = True

        if use_layerwise_lr:
            param_groups = get_layerwise_lr_groups(
                model, lr, lr_decay, finetune_layers
            )
        else:
            param_groups = [
                {'params': [p for p in model.parameters() if p.requires_grad],
                 'lr': lr}
            ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        total_epochs = warmup_epochs + finetune_epochs
        total_steps = total_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        no_improve = 0
        for epoch in range(total_epochs):
            t0 = time.time()
            loss, acc = train_epoch(model, train_loader, optimizer, scheduler,
                                    criterion, device, grad_accum)
            _, _, threshold, f1 = evaluate(model, val_loader, device)

            epoch_log.append({
                'phase': 0, 'epoch': epoch + 1,
                'loss': loss, 'acc': acc, 'val_f1': f1, 'threshold': threshold,
                'time': time.time() - t0
            })

            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_f1, best_threshold, epoch_log


# ============================================================================
# ABLATION CONFIGS
# ============================================================================

ABLATION_CONFIGS = {
    'full_system': {
        'desc': 'Full System',
        'model_class': 'simple',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'no_focal': {
        'desc': '- Focal Loss',
        'model_class': 'simple',
        'use_focal': False, 'focal_gamma': 0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'no_two_phase': {
        'desc': '- Two-Phase',
        'model_class': 'simple',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': False, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'no_layerwise_lr': {
        'desc': '- Layerwise LR',
        'model_class': 'simple',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': False, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'frozen_only': {
        'desc': '- Fine-Tuning',
        'model_class': 'simple',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 0,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'cls_pooling': {
        'desc': '- Mean Pool (CLS)',
        'model_class': 'cls',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
}

# Minimal system: CE + single-phase + uniform LR + mean pooling
MINIMAL_CONFIG = {
    'desc': 'Minimal (CE only)',
    'model_class': 'simple',
    'use_focal': False, 'focal_gamma': 0, 'label_smoothing': 0.0,
    'two_phase': False, 'warmup_epochs': 5, 'finetune_epochs': 4,
    'finetune_layers': 6, 'finetune_lr_mult': 1.0,
    'use_layerwise_lr': False, 'lr_decay': 1.0,
    'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
    'weight_decay': 0.01,
}


# ============================================================================
# EXPERIMENT 1: CV ABLATION (6 configs × 5 folds + minimal config)
# ============================================================================

def run_cv_ablations(logger, n_folds=5):
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT 1: CV ABLATION ({len(ABLATION_CONFIGS)} configs × {n_folds} folds)")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    train_data = load_data(TRAIN_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    labels = [d['label'] for d in train_data]
    logger.info(f"Training data: {len(train_data)} samples, "
                f"Yes: {sum(labels)}, No: {len(labels) - sum(labels)}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = list(skf.split(train_data, labels))

    # Add minimal config
    all_configs = dict(ABLATION_CONFIGS)
    all_configs['minimal'] = MINIMAL_CONFIG

    results_file = RESULTS_DIR / 'cv_ablations.json'
    results = {}
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"Loaded partial results: {list(results.keys())}")

    total_runs = len(all_configs) * n_folds
    completed = sum(1 for cfg_name in all_configs
                    for fold_idx in range(n_folds)
                    if f"{cfg_name}_fold{fold_idx}" in results.get(cfg_name, {}).get('folds', {}))
    logger.info(f"Progress: {completed}/{total_runs} runs completed")

    for cfg_name, config in all_configs.items():
        if cfg_name not in results:
            results[cfg_name] = {'desc': config['desc'], 'folds': {}}

        # Check if this config is already fully done
        existing_folds = results[cfg_name].get('folds', {})
        if len(existing_folds) >= n_folds and 'summary' in results[cfg_name]:
            logger.info(f"Skipping {cfg_name} (all folds complete, "
                        f"F1={results[cfg_name]['summary']['mean_f1']:.4f}±"
                        f"{results[cfg_name]['summary']['std_f1']:.4f})")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Config: {cfg_name} ({config['desc']})")
        logger.info(f"{'='*50}")

        fold_f1s = []
        fold_precs = []
        fold_recs = []
        fold_accs = []

        for fold_idx, (train_ids, val_ids) in enumerate(folds):
            fold_key = f"fold_{fold_idx}"

            if fold_key in existing_folds:
                logger.info(f"  Fold {fold_idx}: already done "
                            f"(F1={existing_folds[fold_key]['macro_f1']:.4f})")
                fold_f1s.append(existing_folds[fold_key]['macro_f1'])
                fold_precs.append(existing_folds[fold_key]['precision'])
                fold_recs.append(existing_folds[fold_key]['recall'])
                fold_accs.append(existing_folds[fold_key]['accuracy'])
                continue

            fold_train = [train_data[i] for i in train_ids]
            fold_val = [train_data[i] for i in val_ids]

            fold_train_ds = SimpleDataset(fold_train, tokenizer, MAX_LEN)
            fold_val_ds = SimpleDataset(fold_val, tokenizer, MAX_LEN)
            fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)
            fold_val_loader = DataLoader(fold_val_ds, batch_size=BATCH_SIZE,
                                         num_workers=NUM_WORKERS)

            set_seed(SEED)
            torch.cuda.empty_cache()

            # Use smaller batch for single-phase (more params unfrozen = more VRAM)
            effective_batch = BATCH_SIZE
            effective_grad_accum = config.get('grad_accum', 2)
            if not config.get('two_phase', True):
                # Single-phase training: halve batch, double grad_accum
                effective_batch = BATCH_SIZE // 2
                effective_grad_accum = config.get('grad_accum', 2) * 2

            # Rebuild loaders with possibly smaller batch
            fold_train_loader = DataLoader(fold_train_ds, batch_size=effective_batch,
                                           shuffle=True, num_workers=NUM_WORKERS)
            fold_val_loader = DataLoader(fold_val_ds, batch_size=effective_batch,
                                         num_workers=NUM_WORKERS)

            # Create model
            freeze = config.get('two_phase', True)
            if config['model_class'] == 'cls':
                model = CLSConspiracyClassifier(MODEL_NAME, dropout=config['dropout'],
                                                 freeze_encoder=freeze).to(device)
            else:
                model = SimpleConspiracyClassifier(MODEL_NAME, dropout=config['dropout'],
                                                    freeze_encoder=freeze).to(device)

            # Override grad_accum in config for this run
            run_config = dict(config)
            run_config['grad_accum'] = effective_grad_accum

            t0 = time.time()
            try:
                best_f1, best_threshold, epoch_log = train_full_model(
                    model, fold_train_loader, fold_val_loader, device, run_config, logger
                )
            except torch.cuda.OutOfMemoryError:
                logger.info(f"  OOM on {cfg_name} fold {fold_idx}, retrying with batch=4")
                del model
                torch.cuda.empty_cache()
                set_seed(SEED)
                fold_train_loader = DataLoader(fold_train_ds, batch_size=4,
                                               shuffle=True, num_workers=NUM_WORKERS)
                fold_val_loader = DataLoader(fold_val_ds, batch_size=4,
                                             num_workers=NUM_WORKERS)
                if config['model_class'] == 'cls':
                    model = CLSConspiracyClassifier(MODEL_NAME, dropout=config['dropout'],
                                                     freeze_encoder=freeze).to(device)
                else:
                    model = SimpleConspiracyClassifier(MODEL_NAME, dropout=config['dropout'],
                                                        freeze_encoder=freeze).to(device)
                run_config['grad_accum'] = 8  # Keep effective batch size ~32
                best_f1, best_threshold, epoch_log = train_full_model(
                    model, fold_train_loader, fold_val_loader, device, run_config, logger
                )
            elapsed = time.time() - t0

            # Final evaluation with best state
            probs, fold_labels, threshold, f1 = evaluate(model, fold_val_loader, device)
            preds = (probs >= threshold).astype(int)
            p_mac, r_mac, _, _ = precision_recall_fscore_support(
                fold_labels, preds, average='macro', zero_division=0
            )

            results[cfg_name]['folds'][fold_key] = {
                'macro_f1': round(float(best_f1), 4),
                'precision': round(float(p_mac), 4),
                'recall': round(float(r_mac), 4),
                'accuracy': round(float(accuracy_score(fold_labels, preds)), 4),
                'threshold': round(float(best_threshold), 3),
                'train_size': len(fold_train),
                'val_size': len(fold_val),
                'train_time_sec': round(elapsed, 1),
                'n_epochs': len(epoch_log),
            }

            fold_f1s.append(best_f1)
            fold_precs.append(float(p_mac))
            fold_recs.append(float(r_mac))
            fold_accs.append(float(accuracy_score(fold_labels, preds)))

            logger.info(f"  Fold {fold_idx}: F1={best_f1:.4f}, P={p_mac:.4f}, "
                        f"R={r_mac:.4f} ({elapsed:.0f}s)")

            # Save checkpoint after each fold
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            del model
            torch.cuda.empty_cache()

        # Compute summary for this config
        if len(fold_f1s) == n_folds:
            results[cfg_name]['summary'] = {
                'mean_f1': round(float(np.mean(fold_f1s)), 4),
                'std_f1': round(float(np.std(fold_f1s)), 4),
                'min_f1': round(float(np.min(fold_f1s)), 4),
                'max_f1': round(float(np.max(fold_f1s)), 4),
                'mean_precision': round(float(np.mean(fold_precs)), 4),
                'std_precision': round(float(np.std(fold_precs)), 4),
                'mean_recall': round(float(np.mean(fold_recs)), 4),
                'std_recall': round(float(np.std(fold_recs)), 4),
                'mean_accuracy': round(float(np.mean(fold_accs)), 4),
                'std_accuracy': round(float(np.std(fold_accs)), 4),
                'fold_f1s': [round(float(f), 4) for f in fold_f1s],
            }
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"  => {cfg_name}: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("CV ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Config':<25} {'Mean F1':>10} {'Std':>8} {'ΔF1':>8}")
    logger.info("-" * 55)

    base_f1 = results.get('full_system', {}).get('summary', {}).get('mean_f1', 0)
    for cfg_name in list(ABLATION_CONFIGS.keys()) + ['minimal']:
        if cfg_name in results and 'summary' in results[cfg_name]:
            s = results[cfg_name]['summary']
            delta = s['mean_f1'] - base_f1
            delta_str = f"{delta:+.4f}" if cfg_name != 'full_system' else "—"
            logger.info(f"{results[cfg_name]['desc']:<25} {s['mean_f1']:>10.4f} "
                        f"{s['std_f1']:>8.4f} {delta_str:>8}")

    return results


# ============================================================================
# EXPERIMENT 2: MULTI-SEED (5 seeds × dev set)
# ============================================================================

def run_multi_seed(logger, seeds=None):
    if seeds is None:
        seeds = [2026, 42, 1337, 7, 2024]

    logger.info("\n" + "=" * 70)
    logger.info(f"EXPERIMENT 2: MULTI-SEED ({len(seeds)} seeds)")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SimpleDataset(train_data, tokenizer, MAX_LEN)
    dev_dataset = SimpleDataset(dev_data, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=NUM_WORKERS)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS)

    results_file = RESULTS_DIR / 'multi_seed.json'
    results = {}
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"Loaded partial results: {list(results.get('seeds', {}).keys())}")

    if 'seeds' not in results:
        results['seeds'] = {}

    config = ABLATION_CONFIGS['full_system']
    seed_f1s = []

    for seed in seeds:
        seed_key = str(seed)
        if seed_key in results['seeds']:
            logger.info(f"  Seed {seed}: already done "
                        f"(F1={results['seeds'][seed_key]['macro_f1']:.4f})")
            seed_f1s.append(results['seeds'][seed_key]['macro_f1'])
            continue

        logger.info(f"\n  Running seed {seed}...")
        set_seed(seed)
        torch.cuda.empty_cache()

        model = SimpleConspiracyClassifier(MODEL_NAME, dropout=config['dropout'],
                                            freeze_encoder=True).to(device)

        t0 = time.time()
        best_f1, best_threshold, epoch_log = train_full_model(
            model, train_loader, dev_loader, device, config, logger
        )
        elapsed = time.time() - t0

        probs, labels, threshold, f1 = evaluate(model, dev_loader, device)
        preds = (probs >= threshold).astype(int)
        p_mac, r_mac, _, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )

        results['seeds'][seed_key] = {
            'macro_f1': round(float(best_f1), 4),
            'precision': round(float(p_mac), 4),
            'recall': round(float(r_mac), 4),
            'accuracy': round(float(accuracy_score(labels, preds)), 4),
            'threshold': round(float(best_threshold), 3),
            'train_time_sec': round(elapsed, 1),
        }
        seed_f1s.append(best_f1)
        logger.info(f"  Seed {seed}: F1={best_f1:.4f} ({elapsed:.0f}s)")

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Summary
    if seed_f1s:
        results['summary'] = {
            'mean_f1': round(float(np.mean(seed_f1s)), 4),
            'std_f1': round(float(np.std(seed_f1s)), 4),
            'min_f1': round(float(np.min(seed_f1s)), 4),
            'max_f1': round(float(np.max(seed_f1s)), 4),
            'n_seeds': len(seed_f1s),
            'seed_f1s': {str(s): round(float(f), 4) for s, f in zip(seeds, seed_f1s)},
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n  Multi-Seed Summary: {np.mean(seed_f1s):.4f} ± {np.std(seed_f1s):.4f}")
        logger.info(f"  Individual: {[f'{f:.4f}' for f in seed_f1s]}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CV Ablation Experiments')
    parser.add_argument('--cv-ablations', action='store_true', help='Run CV ablation experiment')
    parser.add_argument('--multi-seed', action='store_true', help='Run multi-seed experiment')
    parser.add_argument('--all', action='store_true', help='Run everything')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("SemEval 2026 Task 10 — CV Ablation Experiments")
    logger.info(f"Start time: {datetime.now()}")

    run_all = args.all or not any([args.cv_ablations, args.multi_seed])

    t_start = time.time()

    if run_all or args.cv_ablations:
        run_cv_ablations(logger)

    if run_all or args.multi_seed:
        run_multi_seed(logger)

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    logger.info("DONE")


if __name__ == '__main__':
    main()
