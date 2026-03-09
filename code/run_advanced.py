#!/usr/bin/env python3
"""
Advanced Experiments for SemEval 2026 Task 10 Paper
====================================================
Novel analyses that go beyond standard ablations:

  Exp 1: Multi-seed ensemble (3 seeds, averaged probabilities)
  Exp 2: Can't-tell inclusion (train WITH ambiguous data)
  Exp 3: MAX_LENGTH ablation (128 / 256 / 512)
  Exp 4: Cross-subreddit generalization (leave-group-out)
  Exp 5: Layer probing (which DeBERTa layer is most informative?)
  Exp 6: Label noise robustness (flip 5/10/15/20% labels)

Usage:
  python run_advanced.py --all
  python run_advanced.py --ensemble
  python run_advanced.py --canttell
  python run_advanced.py --maxlen
  python run_advanced.py --subreddit
  python run_advanced.py --probing
  python run_advanced.py --noise
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
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

TRAIN_FILE = DATA_DIR / "train_rehydrated.jsonl"
DEV_FILE = DATA_DIR / "dev_public.jsonl"
MODEL_NAME = "microsoft/deberta-v3-large"
SEED = 2026


def setup_dirs():
    for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def setup_logging():
    setup_dirs()
    log_file = LOGS_DIR / f"advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
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


def load_data_all(filepath):
    return load_data(filepath, filter_ambiguous=False)


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
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)


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


def get_layerwise_lr_groups(model, base_lr, lr_decay=0.9, weight_decay=0.05):
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    num_layers = len(model.encoder.encoder.layer)
    groups = []
    groups.append({
        'params': [p for n, p in model.classifier.named_parameters() if p.requires_grad],
        'lr': base_lr * 5, 'weight_decay': weight_decay
    })
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
# CORE TRAINING
# ============================================================================

def find_best_threshold(probs, labels):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.2, 0.8, 0.02):
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    for t in np.arange(best_t - 0.05, best_t + 0.05, 0.005):
        if 0.1 <= t <= 0.9:
            preds = (probs >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    return best_t, best_f1


def train_model(train_data, dev_data, device, tokenizer, seed=SEED,
                max_length=256, logger=None, tag=""):
    """Train a full model and return (best_f1, threshold, dev_probs, dev_labels)."""
    set_seed(seed)
    torch.cuda.empty_cache()

    # Reduce batch size for long sequences to avoid OOM
    if max_length >= 512:
        batch_size = 8
        grad_accum = 4
    else:
        batch_size = 16
        grad_accum = 2

    train_ds = SimpleDataset(train_data, tokenizer, max_length)
    dev_ds = SimpleDataset(dev_data, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, num_workers=2)

    train_labels = [1 if d['conspiracy'].lower() == 'yes' else 0 for d in train_data]
    counts = Counter(train_labels)
    n = len(train_labels)
    cw = torch.tensor([n / (2 * counts[0]), n / (2 * counts[1])],
                       dtype=torch.float32).to(device)

    model = SimpleConspiracyClassifier(MODEL_NAME, dropout=0.1,
                                        freeze_encoder=True).to(device)
    if max_length >= 512:
        model.encoder.gradient_checkpointing_enable()
    criterion = FocalLoss(gamma=2.0, weight=cw, label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler()

    best_f1, best_threshold, best_state = 0, 0.5, None

    # Phase 1: Warmup (5 epochs, frozen encoder)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01
    )
    total_steps = len(train_loader) * 5
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast():
                logits = model(ids, mask)
                loss = criterion(logits, labels) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        # Eval
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask)
                probs = F.softmax(logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)
        t, f1 = find_best_threshold(probs_arr, labels_arr)
        if logger:
            logger.info(f"  {tag} P1 E{epoch+1}: F1={f1:.4f} τ={t:.3f}")
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Phase 2: Fine-tune (4 epochs, top-6 layers)
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.unfreeze_encoder(6)
    param_groups = get_layerwise_lr_groups(model, base_lr=4e-6, lr_decay=0.9)
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = len(train_loader) * 4
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    patience = 0
    for epoch in range(4):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast():
                logits = model(ids, mask)
                loss = criterion(logits, labels) / grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = model(ids, mask)
                probs = F.softmax(logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch['labels'].numpy())
        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)
        t, f1 = find_best_threshold(probs_arr, labels_arr)
        if logger:
            logger.info(f"  {tag} P2 E{epoch+1}: F1={f1:.4f} τ={t:.3f}")
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break

    # Restore best and get final probs
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    final_probs, final_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)
            probs = F.softmax(logits, dim=-1)[:, 1]
            final_probs.extend(probs.cpu().numpy())
            final_labels.extend(batch['labels'].numpy())

    del model
    torch.cuda.empty_cache()

    return best_f1, best_threshold, np.array(final_probs), np.array(final_labels)


# ============================================================================
# EXP 1: MULTI-SEED ENSEMBLE
# ============================================================================

def run_ensemble(logger):
    logger.info("=" * 70)
    logger.info("EXP 1: MULTI-SEED ENSEMBLE")
    logger.info("=" * 70)

    device = torch.device('cuda')
    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    seeds = [2026, 42, 1337]
    all_probs = []
    seed_results = {}

    for seed in seeds:
        logger.info(f"\n  Training seed={seed}")
        f1, threshold, probs, labels = train_model(
            train_data, dev_data, device, tokenizer, seed=seed,
            logger=logger, tag=f"seed{seed}"
        )
        all_probs.append(probs)
        seed_results[seed] = {'f1': round(float(f1), 4), 'threshold': round(float(threshold), 3)}
        logger.info(f"  Seed {seed}: F1={f1:.4f}")

    # Ensemble: average probabilities
    ensemble_probs = np.mean(all_probs, axis=0)
    t, ensemble_f1 = find_best_threshold(ensemble_probs, labels)
    preds = (ensemble_probs >= t).astype(int)
    p, r, _, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)

    logger.info(f"\n  ENSEMBLE (3 seeds, avg probs): F1={ensemble_f1:.4f}, τ={t:.3f}")
    logger.info(f"  Individual: {[seed_results[s]['f1'] for s in seeds]}")

    # Majority vote ensemble
    all_preds = []
    for i, seed in enumerate(seeds):
        t_i = seed_results[seed]['threshold']
        all_preds.append((all_probs[i] >= t_i).astype(int))
    vote_preds = (np.sum(all_preds, axis=0) >= 2).astype(int)  # majority
    _, _, vote_f1, _ = precision_recall_fscore_support(labels, vote_preds, average='macro', zero_division=0)
    logger.info(f"  MAJORITY VOTE: F1={vote_f1:.4f}")

    result = {
        'seeds': seeds,
        'seed_results': {str(k): v for k, v in seed_results.items()},
        'ensemble_avg_f1': round(float(ensemble_f1), 4),
        'ensemble_avg_threshold': round(float(t), 3),
        'majority_vote_f1': round(float(vote_f1), 4),
        'ensemble_probs': ensemble_probs.tolist(),
        'labels': labels.tolist(),
    }
    return result


# ============================================================================
# EXP 2: CAN'T-TELL INCLUSION
# ============================================================================

def run_canttell(logger):
    logger.info("=" * 70)
    logger.info("EXP 2: CAN'T-TELL INCLUSION EXPERIMENT")
    logger.info("=" * 70)

    device = torch.device('cuda')
    train_all = load_data_all(TRAIN_FILE)
    train_filtered = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = {}

    # Baseline: filtered (no can't-tell)
    logger.info("\n  Config A: Without Can't-tell (baseline)")
    f1_a, t_a, _, _ = train_model(
        train_filtered, dev_data, device, tokenizer,
        logger=logger, tag="no_CT"
    )
    results['without_canttell'] = {'f1': round(float(f1_a), 4), 'n_train': len(train_filtered)}
    logger.info(f"  Without Can't-tell: F1={f1_a:.4f} (n={len(train_filtered)})")

    # Config B: Can't-tell as "No"
    logger.info("\n  Config B: Can't-tell → No")
    train_ct_no = []
    for d in train_all:
        d2 = dict(d)
        if d2.get('conspiracy', '').lower() == "can't tell":
            d2['conspiracy'] = 'No'
        train_ct_no.append(d2)
    f1_b, t_b, _, _ = train_model(
        train_ct_no, dev_data, device, tokenizer,
        logger=logger, tag="CT→No"
    )
    results['canttell_as_no'] = {'f1': round(float(f1_b), 4), 'n_train': len(train_ct_no)}
    logger.info(f"  Can't-tell→No: F1={f1_b:.4f} (n={len(train_ct_no)})")

    # Config C: Can't-tell as "Yes"
    logger.info("\n  Config C: Can't-tell → Yes")
    train_ct_yes = []
    for d in train_all:
        d2 = dict(d)
        if d2.get('conspiracy', '').lower() == "can't tell":
            d2['conspiracy'] = 'Yes'
        train_ct_yes.append(d2)
    f1_c, t_c, _, _ = train_model(
        train_ct_yes, dev_data, device, tokenizer,
        logger=logger, tag="CT→Yes"
    )
    results['canttell_as_yes'] = {'f1': round(float(f1_c), 4), 'n_train': len(train_ct_yes)}
    logger.info(f"  Can't-tell→Yes: F1={f1_c:.4f} (n={len(train_ct_yes)})")

    # Config D: Simply drop (same as baseline, for N comparison)
    results['canttell_dropped_n'] = len(train_all) - len(train_filtered)

    logger.info(f"\n  Summary:")
    logger.info(f"    Without Can't-tell: F1={results['without_canttell']['f1']}")
    logger.info(f"    Can't-tell → No:    F1={results['canttell_as_no']['f1']}")
    logger.info(f"    Can't-tell → Yes:   F1={results['canttell_as_yes']['f1']}")

    return results


# ============================================================================
# EXP 3: MAX_LENGTH ABLATION
# ============================================================================

def run_maxlen(logger):
    logger.info("=" * 70)
    logger.info("EXP 3: MAX_LENGTH ABLATION")
    logger.info("=" * 70)

    device = torch.device('cuda')
    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Check text length distribution
    all_texts = [d['text'] for d in train_data + dev_data]
    token_lengths = [len(tokenizer.encode(t)) for t in all_texts]
    logger.info(f"  Token lengths: mean={np.mean(token_lengths):.0f}, "
                f"median={np.median(token_lengths):.0f}, "
                f"p95={np.percentile(token_lengths, 95):.0f}, "
                f"max={max(token_lengths)}")

    results = {}
    for ml in [128, 256, 512]:
        logger.info(f"\n  MAX_LENGTH={ml}")
        truncated = sum(1 for l in token_lengths if l > ml)
        logger.info(f"  Truncated: {truncated}/{len(token_lengths)} ({truncated/len(token_lengths)*100:.1f}%)")

        f1, threshold, _, _ = train_model(
            train_data, dev_data, device, tokenizer, max_length=ml,
            logger=logger, tag=f"ml{ml}"
        )
        results[str(ml)] = {
            'f1': round(float(f1), 4),
            'threshold': round(float(threshold), 3),
            'truncated_pct': round(truncated / len(token_lengths) * 100, 1),
        }
        logger.info(f"  MAX_LENGTH={ml}: F1={f1:.4f}")

    logger.info(f"\n  Summary:")
    for ml, r in results.items():
        logger.info(f"    ML={ml}: F1={r['f1']} (truncated={r['truncated_pct']}%)")

    return results


# ============================================================================
# EXP 4: CROSS-SUBREDDIT GENERALIZATION
# ============================================================================

def run_subreddit_generalization(logger):
    logger.info("=" * 70)
    logger.info("EXP 4: CROSS-SUBREDDIT GENERALIZATION")
    logger.info("=" * 70)

    device = torch.device('cuda')
    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Group by subreddit
    sub_counts = Counter(d.get('subreddit', 'unknown') for d in train_data)
    top_subs = [s for s, c in sub_counts.most_common(5)]

    logger.info(f"  Top subreddits: {top_subs}")

    results = {}

    # Experiment: train WITHOUT the largest subreddit, test on full dev
    for held_sub in top_subs[:3]:  # Top 3 subreddits
        filtered_train = [d for d in train_data if d.get('subreddit', '') != held_sub]
        n_removed = len(train_data) - len(filtered_train)
        logger.info(f"\n  Held out: r/{held_sub} ({n_removed} posts)")

        f1, threshold, probs, labels = train_model(
            filtered_train, dev_data, device, tokenizer,
            logger=logger, tag=f"no_{held_sub[:10]}"
        )
        results[held_sub] = {
            'f1': round(float(f1), 4),
            'n_removed': n_removed,
            'n_train': len(filtered_train),
        }
        logger.info(f"  Without r/{held_sub}: F1={f1:.4f}")

    # Full training baseline (for comparison)
    logger.info(f"\n  Full training (baseline):")
    f1_full, _, _, _ = train_model(
        train_data, dev_data, device, tokenizer,
        logger=logger, tag="full"
    )
    results['full'] = {'f1': round(float(f1_full), 4), 'n_train': len(train_data)}

    logger.info(f"\n  Summary:")
    logger.info(f"    Full training: F1={results['full']['f1']}")
    for sub in top_subs[:3]:
        r = results[sub]
        delta = r['f1'] - results['full']['f1']
        logger.info(f"    Without r/{sub}: F1={r['f1']} (Δ={delta:+.4f}, -{r['n_removed']} posts)")

    return results


# ============================================================================
# EXP 5: LAYER PROBING
# ============================================================================

def run_layer_probing(logger):
    logger.info("=" * 70)
    logger.info("EXP 5: LAYER PROBING (LINEAR PROBE PER LAYER)")
    logger.info("=" * 70)

    device = torch.device('cuda')
    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    set_seed(SEED)

    # Load encoder once
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    encoder.eval()

    # Extract representations from each layer
    def extract_layer_reps(data, layer_idx):
        ds = SimpleDataset(data, tokenizer, 256)
        loader = DataLoader(ds, batch_size=16, num_workers=2)
        reps, labels = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                outputs = encoder(ids, mask, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx]  # (batch, seq, hidden)
                # Mean pool
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
                reps.append(pooled.cpu().numpy())
                labels.extend(batch['labels'].numpy())
        return np.vstack(reps), np.array(labels)

    num_layers = encoder.config.num_hidden_layers
    logger.info(f"  Encoder has {num_layers} layers (+embeddings)")

    # Probe layers: embeddings (0), then every 4th, plus last
    probe_layers = [0] + list(range(4, num_layers + 1, 4))
    if num_layers not in probe_layers:
        probe_layers.append(num_layers)

    from sklearn.linear_model import LogisticRegression

    results = {}
    for layer_idx in probe_layers:
        logger.info(f"  Probing layer {layer_idx}...")
        train_reps, train_labels = extract_layer_reps(train_data, layer_idx)
        dev_reps, dev_labels = extract_layer_reps(dev_data, layer_idx)

        # Simple logistic regression probe
        probe = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                                    random_state=SEED)
        probe.fit(train_reps, train_labels)

        # Use probability threshold optimization
        dev_probs = probe.predict_proba(dev_reps)[:, 1]
        t, f1 = find_best_threshold(dev_probs, dev_labels)

        results[layer_idx] = {
            'f1': round(float(f1), 4),
            'threshold': round(float(t), 3),
        }
        logger.info(f"  Layer {layer_idx}: F1={f1:.4f}")

    del encoder
    torch.cuda.empty_cache()

    logger.info(f"\n  Layer Probing Summary:")
    for layer_idx in sorted(results.keys()):
        bar = '█' * int(results[layer_idx]['f1'] * 50)
        logger.info(f"    Layer {layer_idx:2d}: F1={results[layer_idx]['f1']:.4f} {bar}")

    return results


# ============================================================================
# EXP 6: LABEL NOISE ROBUSTNESS
# ============================================================================

def run_noise_robustness(logger):
    logger.info("=" * 70)
    logger.info("EXP 6: LABEL NOISE ROBUSTNESS")
    logger.info("=" * 70)

    device = torch.device('cuda')
    train_data = load_data(TRAIN_FILE)
    dev_data = load_data(DEV_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    results = {}

    for noise_pct in noise_levels:
        logger.info(f"\n  Noise level: {noise_pct*100:.0f}%")
        random.seed(SEED)

        # Add label noise
        noisy_train = []
        n_flipped = 0
        for d in train_data:
            d2 = dict(d)
            if random.random() < noise_pct:
                # Flip label
                if d2['conspiracy'].lower() == 'yes':
                    d2['conspiracy'] = 'No'
                else:
                    d2['conspiracy'] = 'Yes'
                n_flipped += 1
            noisy_train.append(d2)

        logger.info(f"  Flipped {n_flipped}/{len(train_data)} labels")

        f1, threshold, _, _ = train_model(
            noisy_train, dev_data, device, tokenizer,
            logger=logger, tag=f"noise{int(noise_pct*100)}"
        )

        results[f"{int(noise_pct*100)}%"] = {
            'f1': round(float(f1), 4),
            'n_flipped': n_flipped,
            'pct': noise_pct,
        }
        logger.info(f"  {noise_pct*100:.0f}% noise: F1={f1:.4f}")

    logger.info(f"\n  Noise Robustness Summary:")
    for level, r in results.items():
        delta = r['f1'] - results['0%']['f1']
        bar = '█' * int(r['f1'] * 50)
        logger.info(f"    {level:>4s}: F1={r['f1']:.4f} (Δ={delta:+.4f}) {bar}")

    return results


# ============================================================================
# GENERATE ADVANCED FIGURES
# ============================================================================

def generate_advanced_figures(all_results, logger):
    logger.info("=" * 70)
    logger.info("GENERATING ADVANCED FIGURES")
    logger.info("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'serif',
    })

    # Fig: Layer Probing
    if 'probing' in all_results:
        logger.info("  Fig: Layer probing")
        data = all_results['probing']
        layers = sorted([int(k) for k in data.keys()])
        f1s = [data[str(k)]['f1'] for k in layers]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(layers, f1s, 'bo-', markersize=8, linewidth=2)
        ax.fill_between(layers, f1s, alpha=0.1, color='blue')
        best_layer = layers[np.argmax(f1s)]
        ax.plot(best_layer, max(f1s), 'r*', markersize=15,
                label=f'Best: Layer {best_layer} (F1={max(f1s):.3f})')
        ax.set_xlabel('DeBERTa Layer')
        ax.set_ylabel('Macro F1 (Linear Probe)')
        ax.set_title('Layer-wise Probing: Where is Conspiracy Signal?')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(FIGURES_DIR / 'layer_probing.pdf')
        fig.savefig(FIGURES_DIR / 'layer_probing.png')
        plt.close(fig)

    # Fig: Label Noise Robustness
    if 'noise' in all_results:
        logger.info("  Fig: Noise robustness")
        data = all_results['noise']
        levels = sorted(data.keys(), key=lambda x: float(x.replace('%', '')))
        noise_pcts = [float(l.replace('%', '')) for l in levels]
        f1s = [data[l]['f1'] for l in levels]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(noise_pcts, f1s, 'ro-', markersize=8, linewidth=2)
        ax.fill_between(noise_pcts, f1s, alpha=0.1, color='red')
        ax.set_xlabel('Label Noise (%)')
        ax.set_ylabel('Macro F1')
        ax.set_title('Robustness to Label Noise')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(noise_pcts)
        fig.savefig(FIGURES_DIR / 'noise_robustness.pdf')
        fig.savefig(FIGURES_DIR / 'noise_robustness.png')
        plt.close(fig)

    # Fig: MAX_LENGTH
    if 'maxlen' in all_results:
        logger.info("  Fig: Max length ablation")
        data = all_results['maxlen']
        lengths = sorted([int(k) for k in data.keys()])
        f1s = [data[str(l)]['f1'] for l in lengths]

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.bar([str(l) for l in lengths], f1s, color=['#2196F3', '#4CAF50', '#FF9800'],
               edgecolor='white', width=0.5)
        for i, (l, f1) in enumerate(zip(lengths, f1s)):
            ax.text(i, f1 + 0.005, f'{f1:.4f}', ha='center', fontsize=10)
        ax.set_xlabel('Max Sequence Length')
        ax.set_ylabel('Macro F1')
        ax.set_title('Effect of Input Length')
        ax.set_ylim(min(f1s) - 0.05, max(f1s) + 0.03)
        ax.grid(True, axis='y', alpha=0.3)
        fig.savefig(FIGURES_DIR / 'maxlen_ablation.pdf')
        fig.savefig(FIGURES_DIR / 'maxlen_ablation.png')
        plt.close(fig)

    # Fig: Can't-tell
    if 'canttell' in all_results:
        logger.info("  Fig: Can't-tell inclusion")
        data = all_results['canttell']
        configs = ['Without CT', 'CT → No', 'CT → Yes']
        keys = ['without_canttell', 'canttell_as_no', 'canttell_as_yes']
        f1s = [data[k]['f1'] for k in keys]

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        colors = ['#4CAF50', '#2196F3', '#F44336']
        ax.bar(configs, f1s, color=colors, edgecolor='white', width=0.5)
        for i, f1 in enumerate(f1s):
            ax.text(i, f1 + 0.005, f'{f1:.4f}', ha='center', fontsize=10)
        ax.set_ylabel('Macro F1')
        ax.set_title("Effect of Can't-Tell Training Data")
        ax.set_ylim(min(f1s) - 0.05, max(f1s) + 0.03)
        ax.grid(True, axis='y', alpha=0.3)
        fig.savefig(FIGURES_DIR / 'canttell_experiment.pdf')
        fig.savefig(FIGURES_DIR / 'canttell_experiment.png')
        plt.close(fig)

    # Fig: Ensemble comparison
    if 'ensemble' in all_results:
        logger.info("  Fig: Ensemble comparison")
        data = all_results['ensemble']
        sr = data['seed_results']

        names = [f"Seed {s}" for s in data['seeds']] + ['Avg Ensemble', 'Vote Ensemble']
        f1s = [sr[str(s)]['f1'] for s in data['seeds']] + \
              [data['ensemble_avg_f1'], data['majority_vote_f1']]
        colors = ['#9E9E9E'] * 3 + ['#4CAF50', '#2196F3']

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        y_pos = range(len(names))
        bars = ax.barh(y_pos, f1s, color=colors, edgecolor='white', height=0.6)
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{f1:.4f}', va='center', fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Macro F1')
        ax.set_title('Multi-Seed Ensemble')
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)
        fig.savefig(FIGURES_DIR / 'ensemble_comparison.pdf')
        fig.savefig(FIGURES_DIR / 'ensemble_comparison.png')
        plt.close(fig)

    logger.info(f"  Figures saved to {FIGURES_DIR}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--canttell', action='store_true')
    parser.add_argument('--maxlen', action='store_true')
    parser.add_argument('--subreddit', action='store_true')
    parser.add_argument('--probing', action='store_true')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("SemEval 2026 — Advanced Experiments")

    run_all = args.all or not any([
        args.ensemble, args.canttell, args.maxlen,
        args.subreddit, args.probing, args.noise, args.figures
    ])

    # Load partial results
    results_file = RESULTS_DIR / 'advanced.json'
    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)

    t_start = time.time()

    if run_all or args.ensemble:
        if 'ensemble' not in all_results:
            all_results['ensemble'] = run_ensemble(logger)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    if run_all or args.canttell:
        if 'canttell' not in all_results:
            all_results['canttell'] = run_canttell(logger)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    if run_all or args.maxlen:
        if 'maxlen' not in all_results:
            all_results['maxlen'] = run_maxlen(logger)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    if run_all or args.subreddit:
        if 'subreddit' not in all_results:
            all_results['subreddit'] = run_subreddit_generalization(logger)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    if run_all or args.probing:
        if 'probing' not in all_results:
            all_results['probing'] = run_layer_probing(logger)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    if run_all or args.noise:
        if 'noise' not in all_results:
            all_results['noise'] = run_noise_robustness(logger)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Always generate figures at end
    if run_all or args.figures:
        generate_advanced_figures(all_results, logger)

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    logger.info("DONE ✓")


if __name__ == '__main__':
    main()
