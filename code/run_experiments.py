#!/usr/bin/env python3
"""
Master Experiment Runner for SemEval 2026 Task 10 — Subtask 2
==============================================================
Runs ALL experiments needed for the paper:

  Part 1: Baselines           (~1 min, no GPU)
  Part 2: Main model eval     (~1 min, GPU)
  Part 3: Ablation studies    (~25 min, GPU)
  Part 4: 5-Fold CV           (~25 min, GPU)
  Part 5: Figure generation   (~1 min, no GPU)
  Part 6: Results compilation (~instant)

Usage:
  python run_experiments.py --all                    # Run everything
  python run_experiments.py --baselines              # Part 1 only
  python run_experiments.py --ablations              # Part 3 only
  python run_experiments.py --kfold                  # Part 4 only
  python run_experiments.py --figures                # Part 5 only
  python run_experiments.py --compile                # Part 6 only
"""

import argparse
import json
import os
import sys
import time
import random
import copy
import logging
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    classification_report, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent  # /home/divya/Desktop/Conspiracy/
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

TRAIN_FILE = DATA_DIR / "train_rehydrated.jsonl"
DEV_FILE = DATA_DIR / "dev_public.jsonl"
TEST_FILE = DATA_DIR / "test_rehydrated.jsonl"

# Existing best checkpoint
BEST_CHECKPOINT = BASE_DIR / "outputs" / "simple_v4" / "best_model.pt"

SEED = 2026
MODEL_NAME = "microsoft/deberta-v3-large"


def setup_dirs():
    for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def setup_logging():
    setup_dirs()
    log_file = LOGS_DIR / f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
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
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if filter_ambiguous and item.get('conspiracy', '').lower() == "can't tell":
                continue
            data.append(item)
    return data


class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        conspiracy = item.get('conspiracy')
        label = 1 if conspiracy and conspiracy.lower() == 'yes' else 0

        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'id': item.get('_id', str(idx)),
            'text': text
        }


# ============================================================================
# MODELS
# ============================================================================

class SimpleConspiracyClassifier(nn.Module):
    """Mean pooling + 2-layer MLP (our best)"""
    def __init__(self, model_name, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.hidden_size, 512),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 2)
        )

    def unfreeze_encoder(self, num_layers=6):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = True
        n = len(self.encoder.encoder.layer)
        for i in range(n - num_layers, n):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"  Unfroze top {num_layers} layers. Trainable: {trainable:,}")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)


class CLSConspiracyClassifier(nn.Module):
    """CLS token only + 2-layer MLP (ablation)"""
    def __init__(self, model_name, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.hidden_size, 512),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 2)
        )

    def unfreeze_encoder(self, num_layers=6):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = True
        n = len(self.encoder.encoder.layer)
        for i in range(n - num_layers, n):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_rep)


class DualConspiracyClassifier(nn.Module):
    """CLS + Mean pooling (dual model)"""
    def __init__(self, model_name, dropout=0.15, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.pooling_attention = nn.Linear(self.hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, 768), nn.LayerNorm(768), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 2)
        )

    def unfreeze_encoder(self, num_layers=6):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = True
        n = len(self.encoder.encoder.layer)
        for i in range(n - num_layers, n):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        cls_rep = hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        mean_rep = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.cat([cls_rep, mean_rep], dim=-1)
        return self.classifier(pooled)


# ============================================================================
# LOSS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ============================================================================
# LAYERWISE LR
# ============================================================================

def get_layerwise_lr_groups(model, base_lr, lr_decay=0.9, weight_decay=0.05):
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    num_layers = len(model.encoder.encoder.layer)
    groups = []

    # Classifier head — highest LR
    groups.append({
        'params': [p for n, p in model.classifier.named_parameters() if p.requires_grad],
        'lr': base_lr * 5, 'weight_decay': weight_decay
    })

    # Embeddings — lowest LR
    groups.append({
        'params': [p for n, p in model.encoder.embeddings.named_parameters()
                   if p.requires_grad and not any(nd in n for nd in no_decay)],
        'lr': base_lr * (lr_decay ** num_layers), 'weight_decay': weight_decay
    })
    groups.append({
        'params': [p for n, p in model.encoder.embeddings.named_parameters()
                   if p.requires_grad and any(nd in n for nd in no_decay)],
        'lr': base_lr * (lr_decay ** num_layers), 'weight_decay': 0.0
    })

    # Encoder layers
    for layer_idx in range(num_layers):
        layer = model.encoder.encoder.layer[layer_idx]
        layer_lr = base_lr * (lr_decay ** (num_layers - layer_idx - 1))
        trainable = [p for p in layer.parameters() if p.requires_grad]
        if not trainable:
            continue
        groups.append({
            'params': [p for n, p in layer.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            'lr': layer_lr, 'weight_decay': weight_decay
        })
        groups.append({
            'params': [p for n, p in layer.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            'lr': layer_lr, 'weight_decay': 0.0
        })

    return [g for g in groups if len(g['params']) > 0]


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def find_best_threshold(probs, labels):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.2, 0.8, 0.02):
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    for t in np.arange(best_t - 0.05, best_t + 0.05, 0.005):
        if t < 0.1 or t > 0.9:
            continue
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def train_epoch(model, loader, criterion, optimizer, scheduler, device,
                grad_accum=2, max_grad_norm=1.0, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(ids, mask)
                loss = criterion(logits, labels) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            logits = model(ids, mask)
            loss = criterion(logits, labels) / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        total_loss += loss.item() * grad_accum
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        logits = model(ids, mask)
        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(batch['labels'].numpy())
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    threshold, f1 = find_best_threshold(probs, labels)
    return probs, labels, threshold, f1


def train_full_model(model, train_loader, dev_loader, device, config, logger):
    """
    Full 2-phase training pipeline. Returns (best_f1, best_threshold, epoch_log).
    config is a dict with all hyperparams.
    """
    # Class weights
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch['labels'].numpy())
    counts = Counter(train_labels)
    n = len(train_labels)
    class_weights = torch.tensor(
        [n / (2 * counts[0]), n / (2 * counts[1])], dtype=torch.float32
    ).to(device)

    # Loss
    if config.get('use_focal', True):
        criterion = FocalLoss(
            gamma=config.get('focal_gamma', 2.0),
            weight=class_weights,
            label_smoothing=config.get('label_smoothing', 0.05)
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.get('label_smoothing', 0.05)
        )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    grad_accum = config.get('grad_accum', 2)
    best_f1, best_threshold, best_state = 0, 0.5, None
    epoch_log = []

    # ---- Phase 1: Warmup (frozen encoder) ----
    warmup_epochs = config.get('warmup_epochs', 5)
    if config.get('two_phase', True) and warmup_epochs > 0:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.get('lr', 2e-5) * 5,
            weight_decay=config.get('weight_decay', 0.01)
        )
        total_steps = len(train_loader) * warmup_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        for epoch in range(warmup_epochs):
            t0 = time.time()
            loss, acc = train_epoch(model, train_loader, criterion, optimizer,
                                    scheduler, device, grad_accum, scaler=scaler)
            probs, labels, threshold, val_f1 = evaluate(model, dev_loader, device)
            elapsed = time.time() - t0
            entry = {'phase': 1, 'epoch': epoch + 1, 'loss': loss, 'acc': acc,
                     'val_f1': val_f1, 'threshold': threshold, 'time': elapsed}
            epoch_log.append(entry)
            logger.info(f"  P1 E{epoch+1}/{warmup_epochs}: loss={loss:.4f} acc={acc:.4f} "
                        f"val_f1={val_f1:.4f} τ={threshold:.3f} ({elapsed:.1f}s)")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- Phase 2: Fine-tuning ----
    finetune_epochs = config.get('finetune_epochs', 4)
    if config.get('two_phase', True) and finetune_epochs > 0:
        if best_state:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.unfreeze_encoder(num_layers=config.get('finetune_layers', 6))

        if config.get('use_layerwise_lr', True):
            param_groups = get_layerwise_lr_groups(
                model,
                base_lr=config.get('lr', 2e-5) * config.get('finetune_lr_mult', 0.2),
                lr_decay=config.get('lr_decay', 0.9),
                weight_decay=config.get('weight_decay', 0.01)
            )
            optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.get('lr', 2e-5) * config.get('finetune_lr_mult', 0.2),
                weight_decay=config.get('weight_decay', 0.01)
            )

        total_steps = len(train_loader) * finetune_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        patience, patience_counter = config.get('patience', 3), 0
        for epoch in range(finetune_epochs):
            t0 = time.time()
            loss, acc = train_epoch(model, train_loader, criterion, optimizer,
                                    scheduler, device, grad_accum, scaler=scaler)
            probs, labels, threshold, val_f1 = evaluate(model, dev_loader, device)
            elapsed = time.time() - t0
            entry = {'phase': 2, 'epoch': epoch + 1, 'loss': loss, 'acc': acc,
                     'val_f1': val_f1, 'threshold': threshold, 'time': elapsed}
            epoch_log.append(entry)
            logger.info(f"  P2 E{epoch+1}/{finetune_epochs}: loss={loss:.4f} acc={acc:.4f} "
                        f"val_f1={val_f1:.4f} τ={threshold:.3f} ({elapsed:.1f}s)")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

    # ---- Single-phase (no 2-phase ablation) ----
    if not config.get('two_phase', True):
        model.unfreeze_encoder(num_layers=config.get('finetune_layers', 6))
        total_epochs = warmup_epochs + finetune_epochs

        if config.get('use_layerwise_lr', True):
            param_groups = get_layerwise_lr_groups(
                model,
                base_lr=config.get('lr', 2e-5),
                lr_decay=config.get('lr_decay', 0.9),
                weight_decay=config.get('weight_decay', 0.01)
            )
            optimizer = torch.optim.AdamW(param_groups)
        else:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.get('lr', 2e-5),
                weight_decay=config.get('weight_decay', 0.01)
            )

        total_steps = len(train_loader) * total_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )

        patience, patience_counter = config.get('patience', 3), 0
        for epoch in range(total_epochs):
            t0 = time.time()
            loss, acc = train_epoch(model, train_loader, criterion, optimizer,
                                    scheduler, device, grad_accum, scaler=scaler)
            probs, labels, threshold, val_f1 = evaluate(model, dev_loader, device)
            elapsed = time.time() - t0
            entry = {'phase': 0, 'epoch': epoch + 1, 'loss': loss, 'acc': acc,
                     'val_f1': val_f1, 'threshold': threshold, 'time': elapsed}
            epoch_log.append(entry)
            logger.info(f"  E{epoch+1}/{total_epochs}: loss={loss:.4f} acc={acc:.4f} "
                        f"val_f1={val_f1:.4f} τ={threshold:.3f} ({elapsed:.1f}s)")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

    # ---- Frozen-only ablation ----
    if config.get('frozen_only', False):
        # Already done in phase 1, no phase 2
        pass

    # Restore best
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_f1, best_threshold, epoch_log


# ============================================================================
# PART 1: BASELINES
# ============================================================================

def run_baselines(logger):
    logger.info("=" * 70)
    logger.info("PART 1: BASELINES")
    logger.info("=" * 70)

    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)

    train_texts = [d['text'] for d in train_data]
    train_labels = [1 if d['conspiracy'].lower() == 'yes' else 0 for d in train_data]
    dev_texts = [d['text'] for d in dev_data]
    dev_labels = [1 if d['conspiracy'].lower() == 'yes' else 0 for d in dev_data]

    results = {}

    # 1. Majority class baseline
    majority = Counter(train_labels).most_common(1)[0][0]
    maj_preds = [majority] * len(dev_labels)
    _, _, maj_f1, _ = precision_recall_fscore_support(dev_labels, maj_preds, average='macro', zero_division=0)
    maj_p, maj_r, _, _ = precision_recall_fscore_support(dev_labels, maj_preds, average='macro', zero_division=0)
    maj_acc = accuracy_score(dev_labels, maj_preds)
    results['majority'] = {
        'name': 'Majority Class',
        'macro_f1': round(float(maj_f1), 4),
        'precision': round(float(maj_p), 4),
        'recall': round(float(maj_r), 4),
        'accuracy': round(float(maj_acc), 4),
    }
    logger.info(f"  Majority class: F1={maj_f1:.4f}, Acc={maj_acc:.4f}")

    # 2. Random baseline
    random.seed(SEED)
    rand_rate = sum(train_labels) / len(train_labels)
    rand_preds = [1 if random.random() < rand_rate else 0 for _ in dev_labels]
    _, _, rand_f1, _ = precision_recall_fscore_support(dev_labels, rand_preds, average='macro', zero_division=0)
    rand_p, rand_r, _, _ = precision_recall_fscore_support(dev_labels, rand_preds, average='macro', zero_division=0)
    rand_acc = accuracy_score(dev_labels, rand_preds)
    results['random'] = {
        'name': 'Weighted Random',
        'macro_f1': round(float(rand_f1), 4),
        'precision': round(float(rand_p), 4),
        'recall': round(float(rand_r), 4),
        'accuracy': round(float(rand_acc), 4),
    }
    logger.info(f"  Weighted random: F1={rand_f1:.4f}, Acc={rand_acc:.4f}")

    # 3. TF-IDF + Logistic Regression
    tfidf_lr = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 3),
                                   sublinear_tf=True, min_df=2)),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                                    random_state=SEED))
    ])
    tfidf_lr.fit(train_texts, train_labels)
    lr_preds = tfidf_lr.predict(dev_texts)
    lr_probs = tfidf_lr.predict_proba(dev_texts)[:, 1]
    _, best_lr_f1 = find_best_threshold(lr_probs, np.array(dev_labels))
    lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(dev_labels, lr_preds, average='macro', zero_division=0)
    lr_acc = accuracy_score(dev_labels, lr_preds)
    # Use threshold-optimized F1 if better
    actual_f1 = max(float(lr_f1), float(best_lr_f1))
    results['tfidf_lr'] = {
        'name': 'TF-IDF + LR',
        'macro_f1': round(actual_f1, 4),
        'precision': round(float(lr_p), 4),
        'recall': round(float(lr_r), 4),
        'accuracy': round(float(lr_acc), 4),
    }
    logger.info(f"  TF-IDF + LR: F1={actual_f1:.4f}, Acc={lr_acc:.4f}")

    # 4. TF-IDF + SVM
    tfidf_svm = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 3),
                                   sublinear_tf=True, min_df=2)),
        ('clf', LinearSVC(max_iter=5000, C=1.0, class_weight='balanced',
                          random_state=SEED))
    ])
    tfidf_svm.fit(train_texts, train_labels)
    svm_preds = tfidf_svm.predict(dev_texts)
    svm_p, svm_r, svm_f1, _ = precision_recall_fscore_support(dev_labels, svm_preds, average='macro', zero_division=0)
    svm_acc = accuracy_score(dev_labels, svm_preds)
    results['tfidf_svm'] = {
        'name': 'TF-IDF + SVM',
        'macro_f1': round(float(svm_f1), 4),
        'precision': round(float(svm_p), 4),
        'recall': round(float(svm_r), 4),
        'accuracy': round(float(svm_acc), 4),
    }
    logger.info(f"  TF-IDF + SVM: F1={svm_f1:.4f}, Acc={svm_acc:.4f}")

    # Save
    with open(RESULTS_DIR / 'baselines.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {RESULTS_DIR / 'baselines.json'}")

    return results


# ============================================================================
# PART 2: EVALUATE EXISTING CHECKPOINTS
# ============================================================================

def eval_existing_models(logger):
    logger.info("=" * 70)
    logger.info("PART 2: EVALUATE EXISTING MODELS")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dev_dataset = SimpleDataset(dev_data, tokenizer, 256)
    dev_loader = DataLoader(dev_dataset, batch_size=16, num_workers=2)

    results = {}

    # Simple model (best checkpoint)
    logger.info("  Loading simple model checkpoint...")
    ckpt = torch.load(BEST_CHECKPOINT, map_location='cpu', weights_only=False)
    model = SimpleConspiracyClassifier(MODEL_NAME, dropout=0.1).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    probs, labels, threshold, f1 = evaluate(model, dev_loader, device)
    preds = (probs >= threshold).astype(int)
    p_mac, r_mac, _, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    _, _, f1_per, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)

    results['simple'] = {
        'name': 'DeBERTa + Mean Pool (Ours)',
        'macro_f1': round(float(f1), 4),
        'precision': round(float(p_mac), 4),
        'recall': round(float(r_mac), 4),
        'accuracy': round(float(accuracy_score(labels, preds)), 4),
        'threshold': round(float(threshold), 3),
        'f1_no': round(float(f1_per[0]), 4),
        'f1_yes': round(float(f1_per[1]), 4),
        'confusion_matrix': cm.tolist(),
        'probs': probs.tolist(),
        'labels': labels.tolist(),
    }
    logger.info(f"  Simple: F1={f1:.4f}, P={p_mac:.4f}, R={r_mac:.4f}, τ={threshold:.3f}")
    logger.info(f"    F1(No)={f1_per[0]:.4f}, F1(Yes)={f1_per[1]:.4f}")
    logger.info(f"    CM: TP={cm[1][1]}, FP={cm[0][1]}, FN={cm[1][0]}, TN={cm[0][0]}")

    del model
    torch.cuda.empty_cache()

    # Dual model checkpoint
    dual_ckpt_path = BASE_DIR / "outputs" / "final" / "best_checkpoint.pt"
    if dual_ckpt_path.exists():
        logger.info("  Loading dual model checkpoint...")
        ckpt = torch.load(dual_ckpt_path, map_location='cpu', weights_only=False)
        model = DualConspiracyClassifier(MODEL_NAME, dropout=0.15).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        probs, labels, threshold, f1 = evaluate(model, dev_loader, device)
        preds = (probs >= threshold).astype(int)
        p_mac, r_mac, _, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        _, _, f1_per, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

        results['dual'] = {
            'name': 'DeBERTa + CLS+Mean Pool',
            'macro_f1': round(float(f1), 4),
            'precision': round(float(p_mac), 4),
            'recall': round(float(r_mac), 4),
            'accuracy': round(float(accuracy_score(labels, preds)), 4),
            'threshold': round(float(threshold), 3),
            'f1_no': round(float(f1_per[0]), 4),
            'f1_yes': round(float(f1_per[1]), 4),
        }
        logger.info(f"  Dual: F1={f1:.4f}, P={p_mac:.4f}, R={r_mac:.4f}, τ={threshold:.3f}")
        del model
        torch.cuda.empty_cache()

    with open(RESULTS_DIR / 'models.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# PART 3: ABLATIONS
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
        'desc': '- Focal Loss (use CE)',
        'model_class': 'simple',
        'use_focal': False, 'focal_gamma': 0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'no_two_phase': {
        'desc': '- Two-Phase Training',
        'model_class': 'simple',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': False, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'cls_pooling': {
        'desc': '- Mean Pool (use CLS)',
        'model_class': 'cls',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'frozen_only': {
        'desc': '- Fine-Tuning (frozen)',
        'model_class': 'simple',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 0,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': True, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
    'no_layerwise_lr': {
        'desc': '- Layerwise LR Decay',
        'model_class': 'simple',
        'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
        'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 4,
        'finetune_layers': 6, 'finetune_lr_mult': 0.2,
        'use_layerwise_lr': False, 'lr_decay': 0.9,
        'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
        'weight_decay': 0.01,
    },
}


def run_ablations(logger):
    logger.info("=" * 70)
    logger.info("PART 3: ABLATION STUDIES")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SimpleDataset(train_data, tokenizer, 256)
    dev_dataset = SimpleDataset(dev_data, tokenizer, 256)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=16, num_workers=2)

    results = {}
    # Check for partial results
    results_file = RESULTS_DIR / 'ablations.json'
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"  Loaded partial results: {list(results.keys())}")

    for name, config in ABLATION_CONFIGS.items():
        if name in results:
            logger.info(f"  Skipping {name} (already done: F1={results[name]['macro_f1']})")
            continue

        logger.info(f"\n  Running ablation: {name} ({config['desc']})")
        set_seed(SEED)
        torch.cuda.empty_cache()

        # Create model
        freeze = config.get('two_phase', True)  # Freeze only if 2-phase
        if config['model_class'] == 'cls':
            model = CLSConspiracyClassifier(MODEL_NAME, dropout=config['dropout'],
                                            freeze_encoder=freeze).to(device)
        else:
            model = SimpleConspiracyClassifier(MODEL_NAME, dropout=config['dropout'],
                                               freeze_encoder=freeze).to(device)

        t0 = time.time()
        best_f1, best_threshold, epoch_log = train_full_model(
            model, train_loader, dev_loader, device, config, logger
        )
        elapsed = time.time() - t0

        # Final evaluation
        probs, labels, threshold, f1 = evaluate(model, dev_loader, device)
        preds = (probs >= threshold).astype(int)
        p_mac, r_mac, _, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        _, _, f1_per, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

        results[name] = {
            'desc': config['desc'],
            'macro_f1': round(float(best_f1), 4),
            'precision': round(float(p_mac), 4),
            'recall': round(float(r_mac), 4),
            'accuracy': round(float(accuracy_score(labels, preds)), 4),
            'threshold': round(float(best_threshold), 3),
            'f1_no': round(float(f1_per[0]), 4),
            'f1_yes': round(float(f1_per[1]), 4),
            'epoch_log': epoch_log,
            'train_time_sec': round(elapsed, 1),
        }
        logger.info(f"  => {name}: F1={best_f1:.4f}, P={p_mac:.4f}, R={r_mac:.4f} ({elapsed:.0f}s)")

        # Save after each ablation (checkpoint progress)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    logger.info(f"\n  All ablations saved to {results_file}")
    return results


# ============================================================================
# PART 4: K-FOLD CROSS-VALIDATION
# ============================================================================

def run_kfold(logger, n_folds=5):
    logger.info("=" * 70)
    logger.info(f"PART 4: {n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    logger.info("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = load_data(TRAIN_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    labels = [1 if d['conspiracy'].lower() == 'yes' else 0 for d in train_data]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    results = {}
    results_file = RESULTS_DIR / 'kfold.json'
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        logger.info(f"  Loaded partial results: {list(results.keys())}")

    fold_f1s = []
    for fold_idx, (train_ids, val_ids) in enumerate(skf.split(train_data, labels)):
        fold_key = f"fold_{fold_idx}"
        if fold_key in results:
            logger.info(f"  Skipping fold {fold_idx} (already done: F1={results[fold_key]['macro_f1']})")
            fold_f1s.append(results[fold_key]['macro_f1'])
            continue

        logger.info(f"\n  Fold {fold_idx + 1}/{n_folds}")

        fold_train = [train_data[i] for i in train_ids]
        fold_val = [train_data[i] for i in val_ids]
        logger.info(f"    Train: {len(fold_train)}, Val: {len(fold_val)}")

        fold_train_ds = SimpleDataset(fold_train, tokenizer, 256)
        fold_val_ds = SimpleDataset(fold_val, tokenizer, 256)
        fold_train_loader = DataLoader(fold_train_ds, batch_size=16, shuffle=True, num_workers=2)
        fold_val_loader = DataLoader(fold_val_ds, batch_size=16, num_workers=2)

        set_seed(SEED)
        torch.cuda.empty_cache()
        model = SimpleConspiracyClassifier(MODEL_NAME, dropout=0.1,
                                            freeze_encoder=True).to(device)

        config = {
            'use_focal': True, 'focal_gamma': 2.0, 'label_smoothing': 0.05,
            'two_phase': True, 'warmup_epochs': 5, 'finetune_epochs': 4,
            'finetune_layers': 6, 'finetune_lr_mult': 0.2,
            'use_layerwise_lr': True, 'lr_decay': 0.9,
            'lr': 2e-5, 'dropout': 0.1, 'grad_accum': 2, 'patience': 3,
            'weight_decay': 0.01,
        }

        t0 = time.time()
        best_f1, best_threshold, epoch_log = train_full_model(
            model, fold_train_loader, fold_val_loader, device, config, logger
        )
        elapsed = time.time() - t0

        # Detailed metrics
        probs, fold_labels, threshold, f1 = evaluate(model, fold_val_loader, device)
        preds = (probs >= threshold).astype(int)
        p_mac, r_mac, _, _ = precision_recall_fscore_support(fold_labels, preds, average='macro', zero_division=0)
        _, _, f1_per, _ = precision_recall_fscore_support(fold_labels, preds, average=None, zero_division=0)

        results[fold_key] = {
            'macro_f1': round(float(best_f1), 4),
            'precision': round(float(p_mac), 4),
            'recall': round(float(r_mac), 4),
            'accuracy': round(float(accuracy_score(fold_labels, preds)), 4),
            'threshold': round(float(best_threshold), 3),
            'f1_no': round(float(f1_per[0]), 4),
            'f1_yes': round(float(f1_per[1]), 4),
            'epoch_log': epoch_log,
            'train_time_sec': round(elapsed, 1),
            'train_size': len(fold_train),
            'val_size': len(fold_val),
        }
        fold_f1s.append(best_f1)
        logger.info(f"    Fold {fold_idx + 1}: F1={best_f1:.4f} ({elapsed:.0f}s)")

        # Save after each fold
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    # Summary
    if fold_f1s:
        results['summary'] = {
            'mean_f1': round(float(np.mean(fold_f1s)), 4),
            'std_f1': round(float(np.std(fold_f1s)), 4),
            'min_f1': round(float(np.min(fold_f1s)), 4),
            'max_f1': round(float(np.max(fold_f1s)), 4),
            'fold_f1s': [round(float(f), 4) for f in fold_f1s],
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n  K-Fold Summary: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
        logger.info(f"  Folds: {[f'{f:.4f}' for f in fold_f1s]}")

    return results


# ============================================================================
# PART 5: FIGURES
# ============================================================================

def generate_figures(logger):
    logger.info("=" * 70)
    logger.info("PART 5: GENERATING FIGURES")
    logger.info("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'serif',
    })

    # Load all results
    baselines = {}
    if (RESULTS_DIR / 'baselines.json').exists():
        with open(RESULTS_DIR / 'baselines.json') as f:
            baselines = json.load(f)

    models = {}
    if (RESULTS_DIR / 'models.json').exists():
        with open(RESULTS_DIR / 'models.json') as f:
            models = json.load(f)

    ablations = {}
    if (RESULTS_DIR / 'ablations.json').exists():
        with open(RESULTS_DIR / 'ablations.json') as f:
            ablations = json.load(f)

    kfold = {}
    if (RESULTS_DIR / 'kfold.json').exists():
        with open(RESULTS_DIR / 'kfold.json') as f:
            kfold = json.load(f)

    # ------------------------------------------------------------------
    # Figure 1: Confusion Matrix Heatmap
    # ------------------------------------------------------------------
    if 'simple' in models and 'confusion_matrix' in models['simple']:
        logger.info("  Fig 1: Confusion matrix")
        cm = np.array(models['simple']['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix (Dev Set)')
        fig.savefig(FIGURES_DIR / 'confusion_matrix.pdf')
        fig.savefig(FIGURES_DIR / 'confusion_matrix.png')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Probability Distribution
    # ------------------------------------------------------------------
    if 'simple' in models and 'probs' in models['simple']:
        logger.info("  Fig 2: Probability distribution")
        probs = np.array(models['simple']['probs'])
        labels = np.array(models['simple']['labels'])
        threshold = models['simple']['threshold']

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(probs[labels == 0], bins=20, alpha=0.6, label='No', color='#2196F3', edgecolor='white')
        ax.hist(probs[labels == 1], bins=20, alpha=0.6, label='Yes', color='#F44336', edgecolor='white')
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                    label=f'Threshold ({threshold:.3f})')
        ax.set_xlabel('P(Conspiracy)')
        ax.set_ylabel('Count')
        ax.set_title('Predicted Probability Distribution')
        ax.legend()
        fig.savefig(FIGURES_DIR / 'prob_distribution.pdf')
        fig.savefig(FIGURES_DIR / 'prob_distribution.png')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Threshold Sensitivity
    # ------------------------------------------------------------------
    if 'simple' in models and 'probs' in models['simple']:
        logger.info("  Fig 3: Threshold sensitivity")
        probs = np.array(models['simple']['probs'])
        labels = np.array(models['simple']['labels'])

        thresholds = np.arange(0.1, 0.9, 0.01)
        f1s, precs, recs = [], [], []
        for t in thresholds:
            preds = (probs >= t).astype(int)
            p, r, f, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
            f1s.append(f)
            precs.append(p)
            recs.append(r)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(thresholds, f1s, 'b-', linewidth=2, label='Macro F1')
        ax.plot(thresholds, precs, 'g--', linewidth=1, label='Macro Precision')
        ax.plot(thresholds, recs, 'r--', linewidth=1, label='Macro Recall')
        best_idx = np.argmax(f1s)
        ax.plot(thresholds[best_idx], f1s[best_idx], 'k*', markersize=12,
                label=f'Best: {f1s[best_idx]:.4f} @ {thresholds[best_idx]:.3f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Threshold Sensitivity Analysis')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / 'threshold_sensitivity.pdf')
        fig.savefig(FIGURES_DIR / 'threshold_sensitivity.png')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 4: Ablation Study Bar Chart
    # ------------------------------------------------------------------
    if ablations:
        logger.info("  Fig 4: Ablation study")
        abl_names = []
        abl_f1s = []
        abl_order = ['full_system', 'no_focal', 'no_two_phase', 'cls_pooling',
                      'frozen_only', 'no_layerwise_lr']
        for key in abl_order:
            if key in ablations:
                abl_names.append(ablations[key]['desc'])
                abl_f1s.append(ablations[key]['macro_f1'])

        if abl_names:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#4CAF50'] + ['#F44336'] * (len(abl_names) - 1)
            y_pos = range(len(abl_names))
            bars = ax.barh(y_pos, abl_f1s, color=colors, edgecolor='white', height=0.6)

            # Add value labels
            for bar, f1val in zip(bars, abl_f1s):
                ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{f1val:.4f}', va='center', fontsize=9)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(abl_names)
            ax.set_xlabel('Macro F1')
            ax.set_title('Ablation Study')
            ax.set_xlim(0, max(abl_f1s) * 1.1)
            ax.invert_yaxis()
            ax.grid(True, axis='x', alpha=0.3)

            # Add delta labels
            if abl_f1s:
                base = abl_f1s[0]
                for i, (bar, f1val) in enumerate(zip(bars, abl_f1s)):
                    if i > 0:
                        delta = f1val - base
                        ax.text(0.01, bar.get_y() + bar.get_height()/2,
                                f'  Δ={delta:+.4f}', va='center', fontsize=8,
                                color='white', fontweight='bold')

            fig.savefig(FIGURES_DIR / 'ablation_study.pdf')
            fig.savefig(FIGURES_DIR / 'ablation_study.png')
            plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 5: K-Fold CV Box Plot
    # ------------------------------------------------------------------
    if kfold and 'summary' in kfold:
        logger.info("  Fig 5: K-fold CV")
        fold_f1s = kfold['summary']['fold_f1s']
        mean_f1 = kfold['summary']['mean_f1']
        std_f1 = kfold['summary']['std_f1']

        fig, ax = plt.subplots(figsize=(4, 4))
        bp = ax.boxplot(fold_f1s, patch_artist=True, widths=0.4)
        bp['boxes'][0].set_facecolor('#2196F3')
        bp['boxes'][0].set_alpha(0.6)
        bp['medians'][0].set_color('black')

        # Overlay individual fold points
        ax.scatter([1] * len(fold_f1s), fold_f1s, c='#F44336', zorder=5,
                   s=50, edgecolors='black', linewidth=0.5)

        ax.axhline(mean_f1, color='green', linestyle='--', linewidth=1,
                    label=f'Mean: {mean_f1:.4f} ± {std_f1:.4f}')
        ax.set_ylabel('Macro F1')
        ax.set_title(f'5-Fold Cross-Validation')
        ax.set_xticklabels(['DeBERTa + Mean Pool'])
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

        fig.savefig(FIGURES_DIR / 'kfold_cv.pdf')
        fig.savefig(FIGURES_DIR / 'kfold_cv.png')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 6: Training Curves (from full_system ablation)
    # ------------------------------------------------------------------
    if ablations and 'full_system' in ablations and 'epoch_log' in ablations['full_system']:
        logger.info("  Fig 6: Training curves")
        log = ablations['full_system']['epoch_log']
        epochs = list(range(1, len(log) + 1))
        losses = [e['loss'] for e in log]
        f1s = [e['val_f1'] for e in log]
        phases = [e['phase'] for e in log]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

        # Loss curve
        p1_mask = [i for i, p in enumerate(phases) if p == 1]
        p2_mask = [i for i, p in enumerate(phases) if p == 2]
        p0_mask = [i for i, p in enumerate(phases) if p == 0]

        if p1_mask:
            ax1.plot([epochs[i] for i in p1_mask], [losses[i] for i in p1_mask],
                     'b-o', markersize=4, label='Phase 1 (frozen)')
        if p2_mask:
            ax1.plot([epochs[i] for i in p2_mask], [losses[i] for i in p2_mask],
                     'r-o', markersize=4, label='Phase 2 (fine-tune)')
        if p0_mask:
            ax1.plot([epochs[i] for i in p0_mask], [losses[i] for i in p0_mask],
                     'g-o', markersize=4, label='Single phase')

        if p1_mask and p2_mask:
            boundary = max(p1_mask) + 0.5
            ax1.axvline(boundary, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # F1 curve
        if p1_mask:
            ax2.plot([epochs[i] for i in p1_mask], [f1s[i] for i in p1_mask],
                     'b-o', markersize=4, label='Phase 1 (frozen)')
        if p2_mask:
            ax2.plot([epochs[i] for i in p2_mask], [f1s[i] for i in p2_mask],
                     'r-o', markersize=4, label='Phase 2 (fine-tune)')
        if p0_mask:
            ax2.plot([epochs[i] for i in p0_mask], [f1s[i] for i in p0_mask],
                     'g-o', markersize=4, label='Single phase')

        if p1_mask and p2_mask:
            ax2.axvline(boundary, color='gray', linestyle=':', alpha=0.5)
        best_epoch = np.argmax(f1s)
        ax2.plot(epochs[best_epoch], f1s[best_epoch], 'k*', markersize=12,
                 label=f'Best: {f1s[best_epoch]:.4f}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dev Macro F1')
        ax2.set_title('Validation F1')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Training Dynamics', fontsize=13, y=1.02)
        fig.savefig(FIGURES_DIR / 'training_curves.pdf')
        fig.savefig(FIGURES_DIR / 'training_curves.png')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 7: Combined Results Overview
    # ------------------------------------------------------------------
    logger.info("  Fig 7: Results overview")
    all_systems = []
    for key in ['majority', 'random', 'tfidf_lr', 'tfidf_svm']:
        if key in baselines:
            all_systems.append((baselines[key]['name'], baselines[key]['macro_f1'], 'baseline'))
    if 'dual' in models:
        all_systems.append(('DeBERTa + CLS+Mean', models['dual']['macro_f1'], 'model'))
    if 'simple' in models:
        all_systems.append(('DeBERTa + Mean Pool (Ours)', models['simple']['macro_f1'], 'ours'))

    if all_systems:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names = [s[0] for s in all_systems]
        f1s = [s[1] for s in all_systems]
        colors = []
        for s in all_systems:
            if s[2] == 'baseline':
                colors.append('#9E9E9E')
            elif s[2] == 'ours':
                colors.append('#4CAF50')
            else:
                colors.append('#2196F3')

        y_pos = range(len(names))
        bars = ax.barh(y_pos, f1s, color=colors, edgecolor='white', height=0.6)
        for bar, f1val in zip(bars, f1s):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{f1val:.4f}', va='center', fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Macro F1')
        ax.set_title('System Comparison')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        legend_elements = [
            mpatches.Patch(facecolor='#9E9E9E', label='Baselines'),
            mpatches.Patch(facecolor='#2196F3', label='Neural models'),
            mpatches.Patch(facecolor='#4CAF50', label='Our best'),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

        fig.savefig(FIGURES_DIR / 'results_overview.pdf')
        fig.savefig(FIGURES_DIR / 'results_overview.png')
        plt.close(fig)

    logger.info(f"  All figures saved to {FIGURES_DIR}/")


# ============================================================================
# PART 6: COMPILE RESULTS
# ============================================================================

def compile_results(logger):
    logger.info("=" * 70)
    logger.info("PART 6: COMPILING RESULTS")
    logger.info("=" * 70)

    # Load everything
    baselines, models, ablations, kfold = {}, {}, {}, {}
    for name, var in [('baselines', baselines), ('models', models),
                      ('ablations', ablations), ('kfold', kfold)]:
        path = RESULTS_DIR / f'{name}.json'
        if path.exists():
            with open(path) as f:
                var.update(json.load(f))

    bootstrap = {}
    bi_path = BASE_DIR / "model" / "analysis" / "bootstrap_ci.json"
    if bi_path.exists():
        with open(bi_path) as f:
            bootstrap = json.load(f)

    lines = []
    lines.append("=" * 70)
    lines.append("FULL RESULTS — SemEval 2026 Task 10, Subtask 2")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    # Table 1: Main Results
    lines.append("\n\nTABLE 1: MAIN RESULTS")
    lines.append("-" * 65)
    lines.append(f"{'System':<30} {'Macro F1':>10} {'P':>8} {'R':>8} {'Acc':>8}")
    lines.append("-" * 65)

    for key in ['majority', 'random', 'tfidf_lr', 'tfidf_svm']:
        if key in baselines:
            b = baselines[key]
            lines.append(f"{b['name']:<30} {b['macro_f1']:>10.4f} {b['precision']:>8.4f} "
                         f"{b['recall']:>8.4f} {b['accuracy']:>8.4f}")

    lines.append("-" * 65)

    if 'dual' in models:
        m = models['dual']
        lines.append(f"{m['name']:<30} {m['macro_f1']:>10.4f} {m['precision']:>8.4f} "
                     f"{m['recall']:>8.4f} {m['accuracy']:>8.4f}")

    if 'simple' in models:
        m = models['simple']
        ci_str = ""
        if bootstrap and 'macro_f1' in bootstrap:
            ci_str = f" [{bootstrap['macro_f1']['ci_low']:.3f}-{bootstrap['macro_f1']['ci_high']:.3f}]"
        lines.append(f"{m['name']:<30} {m['macro_f1']:>10.4f} {m['precision']:>8.4f} "
                     f"{m['recall']:>8.4f} {m['accuracy']:>8.4f}{ci_str}")

    lines.append("-" * 65)

    # Table 2: Ablation
    if ablations:
        lines.append("\n\nTABLE 2: ABLATION STUDY")
        lines.append("-" * 55)
        lines.append(f"{'Configuration':<30} {'Macro F1':>10} {'ΔF1':>8}")
        lines.append("-" * 55)

        base_f1 = ablations.get('full_system', {}).get('macro_f1', 0)
        abl_order = ['full_system', 'no_focal', 'no_two_phase', 'cls_pooling',
                      'frozen_only', 'no_layerwise_lr']
        for key in abl_order:
            if key in ablations:
                a = ablations[key]
                delta = a['macro_f1'] - base_f1
                delta_str = f"{delta:+.4f}" if key != 'full_system' else "—"
                lines.append(f"{a['desc']:<30} {a['macro_f1']:>10.4f} {delta_str:>8}")

        lines.append("-" * 55)

    # Table 3: K-Fold
    if kfold and 'summary' in kfold:
        s = kfold['summary']
        lines.append("\n\nTABLE 3: 5-FOLD CROSS-VALIDATION")
        lines.append("-" * 40)
        for i, f1 in enumerate(s['fold_f1s']):
            lines.append(f"  Fold {i+1}: {f1:.4f}")
        lines.append("-" * 40)
        lines.append(f"  Mean:  {s['mean_f1']:.4f} ± {s['std_f1']:.4f}")
        lines.append(f"  Range: [{s['min_f1']:.4f}, {s['max_f1']:.4f}]")

    # Table 4: Per-class
    if 'simple' in models:
        m = models['simple']
        lines.append("\n\nTABLE 4: PER-CLASS PERFORMANCE (Best System)")
        lines.append("-" * 40)
        lines.append(f"  F1(No):  {m.get('f1_no', 'N/A')}")
        lines.append(f"  F1(Yes): {m.get('f1_yes', 'N/A')}")
        if 'confusion_matrix' in m:
            cm = m['confusion_matrix']
            lines.append(f"\n  Confusion Matrix:")
            lines.append(f"                Pred No  Pred Yes")
            lines.append(f"  Actual No      {cm[0][0]:5d}    {cm[0][1]:5d}")
            lines.append(f"  Actual Yes     {cm[1][0]:5d}    {cm[1][1]:5d}")

    # LaTeX tables
    lines.append("\n\n" + "=" * 70)
    lines.append("LATEX TABLES")
    lines.append("=" * 70)

    # LaTeX Table 1
    lines.append("\n% Table 1: Main Results")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("System & Macro F1 & P & R & Acc \\\\")
    lines.append("\\midrule")

    for key in ['majority', 'tfidf_lr', 'tfidf_svm']:
        if key in baselines:
            b = baselines[key]
            lines.append(f"{b['name']} & {b['macro_f1']:.3f} & {b['precision']:.3f} "
                         f"& {b['recall']:.3f} & {b['accuracy']:.3f} \\\\")
    lines.append("\\midrule")

    if 'dual' in models:
        m = models['dual']
        lines.append(f"DeBERTa + CLS+Mean & {m['macro_f1']:.3f} & {m['precision']:.3f} "
                     f"& {m['recall']:.3f} & {m['accuracy']:.3f} \\\\")
    if 'simple' in models:
        m = models['simple']
        lines.append(f"\\textbf{{DeBERTa + Mean Pool}} & \\textbf{{{m['macro_f1']:.3f}}} "
                     f"& {m['precision']:.3f} & {m['recall']:.3f} & {m['accuracy']:.3f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Main results on the development set.}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\end{table}")

    # LaTeX Table 2: Ablation
    if ablations:
        lines.append("\n% Table 2: Ablation Study")
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append("Configuration & Macro F1 & $\\Delta$ \\\\")
        lines.append("\\midrule")

        base_f1 = ablations.get('full_system', {}).get('macro_f1', 0)
        for key in abl_order:
            if key in ablations:
                a = ablations[key]
                delta = a['macro_f1'] - base_f1
                if key == 'full_system':
                    lines.append(f"\\textbf{{Full System}} & \\textbf{{{a['macro_f1']:.3f}}} & --- \\\\")
                    lines.append("\\midrule")
                else:
                    lines.append(f"{a['desc']} & {a['macro_f1']:.3f} & {delta:+.3f} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Ablation study results.}")
        lines.append("\\label{tab:ablation}")
        lines.append("\\end{table}")

    result_text = '\n'.join(lines)

    with open(RESULTS_DIR / 'full_results.txt', 'w') as f:
        f.write(result_text)

    logger.info(result_text)
    logger.info(f"\n  Full results saved to {RESULTS_DIR / 'full_results.txt'}")

    return result_text


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SemEval Experiment Runner')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ablations', action='store_true')
    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("SemEval 2026 Task 10 — Experiment Runner")
    logger.info(f"Start time: {datetime.now()}")

    run_all = args.all or not any([args.baselines, args.eval, args.ablations,
                                    args.kfold, args.figures, args.compile])

    t_start = time.time()

    if run_all or args.baselines:
        run_baselines(logger)

    if run_all or args.eval:
        eval_existing_models(logger)

    if run_all or args.ablations:
        run_ablations(logger)

    if run_all or args.kfold:
        run_kfold(logger)

    if run_all or args.figures:
        generate_figures(logger)

    if run_all or args.compile:
        compile_results(logger)

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    logger.info("DONE ✓")


if __name__ == '__main__':
    main()
