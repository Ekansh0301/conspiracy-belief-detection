#!/usr/bin/env python3
"""
K-Fold Cross-Validation for Conspiracy Belief Detection
========================================================
5-fold stratified CV on the full training set (3,531 filtered samples).
Runs main config + ablation variants. All logs and results saved to model/.

Usage:
    python model/run_kfold.py                    # main config only
    python model/run_kfold.py --ablations         # main + all ablations
    python model/run_kfold.py --folds 3           # fewer folds (faster)
"""

import argparse
import json
import sys
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support, 
                             classification_report, confusion_matrix, f1_score)
from transformers import AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================
MODEL_DIR = Path(__file__).parent
BASE = MODEL_DIR.parent  # subtask2/
LOG_DIR = MODEL_DIR / 'logs'
ANALYSIS_DIR = MODEL_DIR / 'analysis'
CKPT_DIR = MODEL_DIR / 'checkpoints'

for d in [LOG_DIR, ANALYSIS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
        self.log.write(f"K-Fold CV Log — {datetime.now().isoformat()}\n{'='*80}\n\n")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ============================================================================
# CONFIG (mirrors train_simple.py exactly)
# ============================================================================
class Config:
    SEED = 2026
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 256
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    NUM_WARMUP_EPOCHS = 5
    NUM_FINETUNE_EPOCHS = 4
    FINETUNE_LAYERS = 6
    FINETUNE_LR_MULTIPLIER = 0.04  # so base_lr = 1e-4 * 0.04 = 4e-6
    LAYERWISE_LR_DECAY = 0.9
    WARMUP_RATIO = 0.1
    DROPOUT = 0.1
    MAX_GRAD_NORM = 1.0
    GAMMA = 2.0
    LABEL_SMOOTHING = 0.05
    USE_INSTRUCTION = False
    PATIENCE = 3

# ============================================================================
# DATASET
# ============================================================================
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=256, instruction=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        if self.instruction:
            text = self.instruction + text
        conspiracy = item.get('conspiracy')
        label = 1 if conspiracy and conspiracy.lower() == 'yes' else 0
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'id': item.get('_id', str(idx)),
        }

# ============================================================================
# MODEL
# ============================================================================
class SimpleConspiracyClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        
    def unfreeze_encoder(self, num_layers=6):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = True
        num_encoder_layers = len(self.encoder.encoder.layer)
        for i in range(num_encoder_layers - num_layers, num_encoder_layers):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfroze top {num_layers} layers. Trainable: {trainable:,}")
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)

# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
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
    num_layers = len(model.encoder.encoder.layer)
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    groups = []
    # Classifier head
    groups.append({
        'params': [p for n, p in model.classifier.named_parameters() if p.requires_grad],
        'lr': base_lr * 5, 'weight_decay': weight_decay})
    # Embeddings
    groups.append({
        'params': [p for n, p in model.encoder.embeddings.named_parameters()
                   if p.requires_grad and not any(nd in n for nd in no_decay)],
        'lr': base_lr * (lr_decay ** num_layers), 'weight_decay': weight_decay})
    groups.append({
        'params': [p for n, p in model.encoder.embeddings.named_parameters()
                   if p.requires_grad and any(nd in n for nd in no_decay)],
        'lr': base_lr * (lr_decay ** num_layers), 'weight_decay': 0.0})
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
            'lr': layer_lr, 'weight_decay': weight_decay})
        groups.append({
            'params': [p for n, p in layer.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            'lr': layer_lr, 'weight_decay': 0.0})
    return [g for g in groups if len(g['params']) > 0]

# ============================================================================
# THRESHOLD FINDER
# ============================================================================
def find_best_threshold(probs, labels):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.30, 0.70, 0.02):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    # Fine search around best
    for t in np.arange(max(0.2, best_t - 0.03), min(0.8, best_t + 0.03), 0.005):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

# ============================================================================
# TRAIN ONE EPOCH
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, scheduler, device, config, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    
    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / config.GRADIENT_ACCUMULATION
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / config.GRADIENT_ACCUMULATION
            loss.backward()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (step + 1) % config.GRADIENT_ACCUMULATION == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    return total_loss / len(loader), correct / total if total > 0 else 0

# ============================================================================
# EVALUATE
# ============================================================================
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    threshold, macro_f1 = find_best_threshold(probs, labels)
    preds = (probs >= threshold).astype(int)
    return probs, labels, threshold, macro_f1, preds

# ============================================================================
# TRAIN ONE FOLD (full 2-phase pipeline)
# ============================================================================
def train_fold(fold_idx, train_data, val_data, tokenizer, device, config, 
               variant_name="main"):
    """Train a single fold with the 2-phase pipeline. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx+1} | {variant_name}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"{'='*60}")
    
    fold_start = time.time()
    
    instruction = "" if not config.USE_INSTRUCTION else None
    train_ds = SimpleDataset(train_data, tokenizer, config.MAX_LENGTH, instruction)
    val_ds = SimpleDataset(val_data, tokenizer, config.MAX_LENGTH, instruction)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=2, 
                            pin_memory=True)
    
    # Class weights
    labels_arr = [1 if d['conspiracy'].lower() == 'yes' else 0 for d in train_data]
    counts = Counter(labels_arr)
    n = len(labels_arr)
    class_weights = torch.tensor([n/(2*counts[0]), n/(2*counts[1])], dtype=torch.float32).to(device)
    
    # Model
    model = SimpleConspiracyClassifier(
        config.MODEL_NAME, dropout=config.DROPOUT, freeze_encoder=True).to(device)
    
    criterion = FocalLoss(
        gamma=config.GAMMA, weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING)
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_f1 = 0
    best_threshold = 0.5
    best_state = None
    
    # Phase 1: Warmup (frozen encoder)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE * 5, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.NUM_WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps)
    
    for epoch in range(config.NUM_WARMUP_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, scaler)
        probs, labels, threshold, val_f1, preds = evaluate(model, val_loader, device)
        print(f"  P1 E{epoch+1}: loss={train_loss:.4f} acc={train_acc:.3f} val_f1={val_f1:.4f} (τ={threshold:.3f})")
        if val_f1 > best_f1:
            best_f1, best_threshold = val_f1, threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Phase 2: Fine-tuning (top layers unfrozen)
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.unfreeze_encoder(num_layers=config.FINETUNE_LAYERS)
    
    param_groups = get_layerwise_lr_groups(
        model, base_lr=config.LEARNING_RATE * config.FINETUNE_LR_MULTIPLIER,
        lr_decay=config.LAYERWISE_LR_DECAY, weight_decay=config.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = len(train_loader) * config.NUM_FINETUNE_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps)
    
    patience = 0
    for epoch in range(config.NUM_FINETUNE_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, scaler)
        probs, labels, threshold, val_f1, preds = evaluate(model, val_loader, device)
        _, _, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        print(f"  P2 E{epoch+1}: loss={train_loss:.4f} acc={train_acc:.3f} val_f1={val_f1:.4f} (τ={threshold:.3f}) No-F1={f1_per_class[0]:.3f} Yes-F1={f1_per_class[1]:.3f}")
        if val_f1 > best_f1:
            best_f1, best_threshold = val_f1, threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation with best checkpoint
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    probs, labels, threshold, val_f1, preds = evaluate(model, val_loader, device)
    
    # Detailed metrics
    prec, rec, f1_class, sup = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)
    
    fold_time = time.time() - fold_start
    
    metrics = {
        'fold': fold_idx,
        'variant': variant_name,
        'macro_f1': float(val_f1),
        'threshold': float(threshold),
        'no_f1': float(f1_class[0]), 'no_prec': float(prec[0]), 'no_rec': float(rec[0]),
        'yes_f1': float(f1_class[1]), 'yes_prec': float(prec[1]), 'yes_rec': float(rec[1]),
        'accuracy': float((preds == labels).mean()),
        'confusion_matrix': cm.tolist(),
        'n_train': len(train_data), 'n_val': len(val_data),
        'time_seconds': fold_time,
    }
    
    print(f"\n  Fold {fold_idx+1} RESULT: Macro F1={val_f1:.4f}, Yes-F1={f1_class[1]:.4f}, No-F1={f1_class[0]:.4f}, τ={threshold:.3f}, time={fold_time:.0f}s")
    
    # Clean up GPU memory
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    
    return metrics

# ============================================================================
# ABLATION CONFIGS
# ============================================================================
def make_ablation_configs():
    """Return dict of ablation variants."""
    ablations = {}
    
    # (a) No two-phase: direct fine-tuning from scratch (no frozen warmup)
    cfg_no_twophase = Config()
    cfg_no_twophase.NUM_WARMUP_EPOCHS = 0
    cfg_no_twophase.NUM_FINETUNE_EPOCHS = 9  # same total epochs
    ablations['no_two_phase'] = cfg_no_twophase
    
    # (b) No focal loss: standard CE
    cfg_no_focal = Config()
    cfg_no_focal.GAMMA = 0.0  # gamma=0 → standard CE
    ablations['no_focal_loss'] = cfg_no_focal
    
    # (c) No layerwise LR decay: uniform LR
    cfg_no_layerwise = Config()
    cfg_no_layerwise.LAYERWISE_LR_DECAY = 1.0  # no decay
    ablations['no_layerwise_lr'] = cfg_no_layerwise
    
    return ablations

# ============================================================================
# MODIFIED train_fold for no-two-phase ablation
# ============================================================================
def train_fold_ablation(fold_idx, train_data, val_data, tokenizer, device, config, 
                        variant_name="ablation"):
    """Train fold — handles ablation via config differences."""
    if config.NUM_WARMUP_EPOCHS == 0:
        # No two-phase: go directly to fine-tuning from start
        return train_fold_no_twophase(fold_idx, train_data, val_data, tokenizer, 
                                       device, config, variant_name)
    else:
        return train_fold(fold_idx, train_data, val_data, tokenizer, device, 
                         config, variant_name)

def train_fold_no_twophase(fold_idx, train_data, val_data, tokenizer, device, config,
                           variant_name):
    """Direct fine-tuning without frozen warmup phase."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx+1} | {variant_name} (no two-phase)")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"{'='*60}")
    
    fold_start = time.time()
    instruction = "" if not config.USE_INSTRUCTION else None
    train_ds = SimpleDataset(train_data, tokenizer, config.MAX_LENGTH, instruction)
    val_ds = SimpleDataset(val_data, tokenizer, config.MAX_LENGTH, instruction)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, num_workers=2,
                            pin_memory=True)
    
    labels_arr = [1 if d['conspiracy'].lower() == 'yes' else 0 for d in train_data]
    counts = Counter(labels_arr)
    n = len(labels_arr)
    class_weights = torch.tensor([n/(2*counts[0]), n/(2*counts[1])], dtype=torch.float32).to(device)
    
    model = SimpleConspiracyClassifier(
        config.MODEL_NAME, dropout=config.DROPOUT, freeze_encoder=False).to(device)
    model.unfreeze_encoder(num_layers=config.FINETUNE_LAYERS)
    
    criterion = FocalLoss(gamma=config.GAMMA, weight=class_weights,
                          label_smoothing=config.LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    param_groups = get_layerwise_lr_groups(
        model, base_lr=config.LEARNING_RATE * config.FINETUNE_LR_MULTIPLIER,
        lr_decay=config.LAYERWISE_LR_DECAY, weight_decay=config.WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = len(train_loader) * config.NUM_FINETUNE_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps)
    
    best_f1, best_threshold, best_state = 0, 0.5, None
    patience = 0
    
    for epoch in range(config.NUM_FINETUNE_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, scaler)
        probs, labels, threshold, val_f1, preds = evaluate(model, val_loader, device)
        _, _, f1_class, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        print(f"  E{epoch+1}: loss={train_loss:.4f} acc={train_acc:.3f} val_f1={val_f1:.4f} (τ={threshold:.3f})")
        if val_f1 > best_f1:
            best_f1, best_threshold = val_f1, threshold
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    probs, labels, threshold, val_f1, preds = evaluate(model, val_loader, device)
    prec, rec, f1_class, sup = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    cm = confusion_matrix(labels, preds)
    fold_time = time.time() - fold_start
    
    metrics = {
        'fold': fold_idx, 'variant': variant_name,
        'macro_f1': float(val_f1), 'threshold': float(threshold),
        'no_f1': float(f1_class[0]), 'no_prec': float(prec[0]), 'no_rec': float(rec[0]),
        'yes_f1': float(f1_class[1]), 'yes_prec': float(prec[1]), 'yes_rec': float(rec[1]),
        'accuracy': float((preds == labels).mean()),
        'confusion_matrix': cm.tolist(),
        'n_train': len(train_data), 'n_val': len(val_data),
        'time_seconds': fold_time,
    }
    
    print(f"\n  Fold {fold_idx+1} RESULT: Macro F1={val_f1:.4f}, time={fold_time:.0f}s")
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return metrics

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--ablations', action='store_true', help='Also run ablation variants')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sys.stdout = DualLogger(LOG_DIR / f'kfold_{timestamp}.log')
    
    print(f"K-Fold Cross-Validation")
    print(f"Folds: {args.folds}, Ablations: {args.ablations}, Seed: {args.seed}")
    print(f"Started: {datetime.now().isoformat()}")
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    train_path = BASE.parent / 'train_rehydrated.jsonl'
    with open(train_path) as f:
        all_data = [json.loads(line) for line in f]
    
    # Filter (no Can't tell)
    filtered = [d for d in all_data if d['conspiracy'] in ('Yes', 'No')]
    print(f"\nData: {len(all_data)} total → {len(filtered)} (Yes={sum(1 for d in filtered if d['conspiracy']=='Yes')}, No={sum(1 for d in filtered if d['conspiracy']=='No')})")
    
    # Tokenizer (slow tokenizer for DeBERTa v3)
    from transformers import DebertaV2Tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(Config.MODEL_NAME)
    print(f"Tokenizer: {type(tokenizer).__name__}")
    
    # Prepare labels for stratified split
    labels = np.array([1 if d['conspiracy'] == 'Yes' else 0 for d in filtered])
    
    # ================================================================
    # RUN MAIN CONFIG
    # ================================================================
    print(f"\n{'#'*70}")
    print(f"# MAIN CONFIG: {args.folds}-Fold CV")
    print(f"{'#'*70}")
    
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    main_results = []
    
    total_start = time.time()
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filtered, labels)):
        train_data = [filtered[i] for i in train_idx]
        val_data = [filtered[i] for i in val_idx]
        metrics = train_fold(fold_idx, train_data, val_data, tokenizer, device, Config(), "main")
        main_results.append(metrics)
        
        # Save intermediate results
        with open(ANALYSIS_DIR / 'kfold_main_partial.json', 'w') as f:
            json.dump(main_results, f, indent=2)
    
    # Summarize main results
    main_f1s = [m['macro_f1'] for m in main_results]
    main_yes_f1s = [m['yes_f1'] for m in main_results]
    main_no_f1s = [m['no_f1'] for m in main_results]
    main_accs = [m['accuracy'] for m in main_results]
    
    print(f"\n{'='*70}")
    print(f"MAIN CONFIG SUMMARY ({args.folds}-fold)")
    print(f"{'='*70}")
    print(f"  Macro F1:  {np.mean(main_f1s):.4f} ± {np.std(main_f1s):.4f}")
    print(f"  Yes F1:    {np.mean(main_yes_f1s):.4f} ± {np.std(main_yes_f1s):.4f}")
    print(f"  No F1:     {np.mean(main_no_f1s):.4f} ± {np.std(main_no_f1s):.4f}")
    print(f"  Accuracy:  {np.mean(main_accs):.4f} ± {np.std(main_accs):.4f}")
    print(f"  Per-fold:  {[f'{f:.4f}' for f in main_f1s]}")
    print(f"  Time: {sum(m['time_seconds'] for m in main_results):.0f}s total")
    
    all_results = {'main': main_results}
    summary = {
        'main': {
            'macro_f1_mean': float(np.mean(main_f1s)),
            'macro_f1_std': float(np.std(main_f1s)),
            'yes_f1_mean': float(np.mean(main_yes_f1s)),
            'yes_f1_std': float(np.std(main_yes_f1s)),
            'no_f1_mean': float(np.mean(main_no_f1s)),
            'no_f1_std': float(np.std(main_no_f1s)),
            'accuracy_mean': float(np.mean(main_accs)),
            'accuracy_std': float(np.std(main_accs)),
            'per_fold_f1': [float(f) for f in main_f1s],
        }
    }
    
    # ================================================================
    # ABLATIONS
    # ================================================================
    if args.ablations:
        ablation_configs = make_ablation_configs()
        
        for abl_name, abl_cfg in ablation_configs.items():
            print(f"\n{'#'*70}")
            print(f"# ABLATION: {abl_name}")
            print(f"{'#'*70}")
            
            abl_results = []
            skf2 = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf2.split(filtered, labels)):
                train_data = [filtered[i] for i in train_idx]
                val_data = [filtered[i] for i in val_idx]
                metrics = train_fold_ablation(
                    fold_idx, train_data, val_data, tokenizer, device, 
                    abl_cfg, abl_name)
                abl_results.append(metrics)
            
            abl_f1s = [m['macro_f1'] for m in abl_results]
            abl_yes = [m['yes_f1'] for m in abl_results]
            abl_no = [m['no_f1'] for m in abl_results]
            abl_accs = [m['accuracy'] for m in abl_results]
            
            print(f"\n  {abl_name} SUMMARY:")
            print(f"    Macro F1: {np.mean(abl_f1s):.4f} ± {np.std(abl_f1s):.4f}")
            print(f"    vs main:  Δ = {np.mean(abl_f1s) - np.mean(main_f1s):+.4f}")
            
            all_results[abl_name] = abl_results
            summary[abl_name] = {
                'macro_f1_mean': float(np.mean(abl_f1s)),
                'macro_f1_std': float(np.std(abl_f1s)),
                'yes_f1_mean': float(np.mean(abl_yes)),
                'yes_f1_std': float(np.std(abl_yes)),
                'no_f1_mean': float(np.mean(abl_no)),
                'no_f1_std': float(np.std(abl_no)),
                'accuracy_mean': float(np.mean(abl_accs)),
                'accuracy_std': float(np.std(abl_accs)),
                'per_fold_f1': [float(f) for f in abl_f1s],
                'delta_vs_main': float(np.mean(abl_f1s) - np.mean(main_f1s)),
            }
    
    # ================================================================
    # SAVE EVERYTHING
    # ================================================================
    total_time = time.time() - total_start
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_folds': args.folds,
            'seed': args.seed,
            'n_samples': len(filtered),
            'model': Config.MODEL_NAME,
            'total_time_seconds': total_time,
        },
        'summary': summary,
        'detailed_results': all_results,
    }
    
    with open(ANALYSIS_DIR / 'kfold_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"ALL DONE — Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"\nResults saved to: {ANALYSIS_DIR / 'kfold_results.json'}")
    print(f"Log saved to: {LOG_DIR}/kfold_{timestamp}.log")
    
    # Print final comparison table
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Variant':<25} {'Macro F1':>15} {'Yes F1':>15} {'No F1':>15}")
    print(f"{'-'*70}")
    for name, s in summary.items():
        delta = f" ({s.get('delta_vs_main', 0):+.3f})" if name != 'main' else ""
        print(f"{name:<25} {s['macro_f1_mean']:.4f}±{s['macro_f1_std']:.4f}{delta:>8} {s['yes_f1_mean']:.4f}±{s['yes_f1_std']:.4f} {s['no_f1_mean']:.4f}±{s['no_f1_std']:.4f}")


if __name__ == '__main__':
    main()
