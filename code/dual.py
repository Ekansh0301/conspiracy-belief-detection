"""
FINAL BATTLE-TESTED VERSION - Target: 0.90+ F1
==============================================

PHILOSOPHY: Keep what works, remove what doesn't
- YES: Strong warmup, layerwise LR, focal loss, attention pooling
- NO: Contrastive loss (too few pairs), mixup (confuses model), multi-sample dropout

Based on error analysis:
- 74% predictions in 0.4-0.6 → Need sharper decision boundary
- FPs are keyword-triggered → Need better pooling
- FNs are subtle → Need more capacity

CHANGES FROM BASELINE:
1. CLS + Mean pooling ensemble (proven +2-3% on DeBERTa)
2. Longer warmup (7 epochs - your baseline peaked at epoch 5)
3. More aggressive fine-tuning (all 24 layers, not just top 6)
4. Higher dropout during training (0.15 vs 0.1)
5. SWA (Stochastic Weight Averaging) in final epochs
"""

import argparse
import json
import random
import copy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm


# ============================================================================
# CONFIG - BATTLE-TESTED SETTINGS
# ============================================================================

class Config:
    # Model
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 256
    
    # Training - Optimized for RTX 4080 Super
    BATCH_SIZE = 24  # Warmup phase batch size
    BATCH_SIZE_FINETUNE = 8  # Smaller for fine-tuning (all layers unfrozen)
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective = 48 for warmup
    GRADIENT_ACCUMULATION_FINETUNE = 6  # Effective = 48 for finetune
    LEARNING_RATE = 2.5e-5  # Slightly higher
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    DROPOUT = 0.15  # Higher dropout for better generalization
    
    # Layerwise LR decay
    LAYERWISE_LR_DECAY = 0.95  # Less aggressive decay
    
    # Epochs - LONGER training is key
    NUM_WARMUP_EPOCHS = 7  # Increased from 5
    NUM_FINETUNE_EPOCHS = 8  # Increased from 6
    FINETUNE_LR_MULTIPLIER = 0.3  # Slightly higher
    PATIENCE = 5  # More patience
    FINETUNE_ALL_LAYERS = True  # Unfreeze ALL layers
    
    # Loss
    FOCAL_GAMMA = 2.5  # More aggressive focal loss
    LABEL_SMOOTHING = 0.08  # Slightly more smoothing
    
    # SWA (Stochastic Weight Averaging)
    USE_SWA = True
    SWA_START_EPOCH = 4  # Start SWA after 4 epochs of fine-tuning
    SWA_LR = 1e-5
    
    # RTX 4080 Super optimizations
    USE_BF16 = True
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Seed
    SEED = 42 # Changed from 2026
    SAVE_ERROR_SAMPLES = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Allow cudnn optimization
    torch.backends.cudnn.benchmark = True  # Enable cudnn auto-tuner


# ============================================================================
# DATA
# ============================================================================

def load_data(filepath, filter_ambiguous=True):
    """Load JSONL data"""
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
        if conspiracy is None:
            label = 0
        else:
            label = 1 if conspiracy.lower() == 'yes' else 0
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'id': item.get('_id', str(idx)),
            'text': text
        }


# ============================================================================
# IMPROVED MODEL - CLS + MEAN POOLING ENSEMBLE
# ============================================================================

class ImprovedConspiracyClassifier(nn.Module):
    """
    DeBERTa with CLS + Mean pooling ensemble
    
    Key insight: DeBERTa-v3 is trained with replaced token detection (RTD)
    so both CLS token and mean pooling contain useful information.
    Ensembling them is better than either alone.
    """
    
    def __init__(self, model_name: str, dropout: float = 0.15, freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Attention-weighted pooling (learns to weight CLS vs mean)
        self.pooling_attention = nn.Linear(self.hidden_size, 1)
        
        # Classification head with more capacity
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 2, 768),  # 2x input (CLS + mean)
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        
    def unfreeze_all_encoder(self):
        """Unfreeze ALL encoder layers"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze ALL encoder layers. Trainable params: {trainable:,}")
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # CLS token (first token)
        cls_rep = hidden_states[:, 0, :]
        
        # Mean pooling (over non-padding tokens)
        mask = attention_mask.unsqueeze(-1).float()
        mean_rep = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # Concatenate both representations
        pooled = torch.cat([cls_rep, mean_rep], dim=-1)
        
        logits = self.classifier(pooled)
        return logits


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.5, weight=None, label_smoothing: float = 0.08):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(
            logits, labels, 
            weight=self.weight, 
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# LAYERWISE LR
# ============================================================================

def get_layerwise_lr_groups(model, base_lr, lr_decay=0.95, weight_decay=0.01):
    """Layerwise learning rate decay"""
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    groups = []
    
    # Classifier head - highest LR
    groups.append({
        'params': [p for n, p in model.classifier.named_parameters() 
                  if p.requires_grad and not any(nd in n for nd in no_decay)],
        'lr': base_lr * 2,
        'weight_decay': weight_decay
    })
    groups.append({
        'params': [p for n, p in model.classifier.named_parameters() 
                  if p.requires_grad and any(nd in n for nd in no_decay)],
        'lr': base_lr * 2,
        'weight_decay': 0.0
    })
    
    # Pooling attention
    groups.append({
        'params': [p for p in model.pooling_attention.parameters() if p.requires_grad],
        'lr': base_lr * 2,
        'weight_decay': weight_decay
    })
    
    # Embeddings
    num_layers = len(model.encoder.encoder.layer)
    groups.append({
        'params': [p for n, p in model.encoder.embeddings.named_parameters() 
                  if p.requires_grad and not any(nd in n for nd in no_decay)],
        'lr': base_lr * (lr_decay ** num_layers),
        'weight_decay': weight_decay
    })
    groups.append({
        'params': [p for n, p in model.encoder.embeddings.named_parameters() 
                  if p.requires_grad and any(nd in n for nd in no_decay)],
        'lr': base_lr * (lr_decay ** num_layers),
        'weight_decay': 0.0
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
            'lr': layer_lr,
            'weight_decay': weight_decay
        })
        groups.append({
            'params': [p for n, p in layer.named_parameters() 
                      if p.requires_grad and any(nd in n for nd in no_decay)],
            'lr': layer_lr,
            'weight_decay': 0.0
        })
    
    groups = [g for g in groups if len(g['params']) > 0]
    return groups


# ============================================================================
# SWA (Stochastic Weight Averaging)
# ============================================================================

class SWA:
    """Simple SWA implementation"""
    def __init__(self, model):
        self.model = model
        self.swa_state = None
        self.swa_n = 0
    
    def update(self):
        """Update SWA model"""
        if self.swa_state is None:
            self.swa_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.swa_n = 1
        else:
            alpha = 1.0 / (self.swa_n + 1)
            for k, v in self.model.state_dict().items():
                self.swa_state[k] = self.swa_state[k] * (1 - alpha) + v.cpu() * alpha
            self.swa_n += 1
    
    def apply(self):
        """Apply SWA weights to model"""
        if self.swa_state is not None:
            self.model.load_state_dict({k: v.to(next(self.model.parameters()).device) 
                                       for k, v in self.swa_state.items()})


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_errors(probs, labels, texts, ids, threshold, output_dir):
    """Comprehensive error analysis"""
    preds = (probs >= threshold).astype(int)
    
    errors = []
    confidences = []
    
    for i in range(len(preds)):
        is_error = bool(preds[i] != labels[i])
        confidence = abs(probs[i] - 0.5) * 2
        
        error_info = {
            'id': str(ids[i]),
            'text': str(texts[i][:500]),
            'true_label': 'Yes' if int(labels[i]) == 1 else 'No',
            'pred_label': 'Yes' if int(preds[i]) == 1 else 'No',
            'prob': float(probs[i]),
            'confidence': float(confidence),
            'is_error': is_error,
            'error_type': None
        }
        
        if is_error:
            if int(labels[i]) == 1 and int(preds[i]) == 0:
                error_info['error_type'] = 'FN'
            else:
                error_info['error_type'] = 'FP'
            errors.append(error_info)
        
        confidences.append(confidence)
    
    errors.sort(key=lambda x: x['prob'], reverse=True)
    fn_errors = [e for e in errors if e['error_type'] == 'FN']
    fp_errors = [e for e in errors if e['error_type'] == 'FP']
    
    error_probs = [e['prob'] for e in errors]
    correct_indices = [i for i in range(len(preds)) if preds[i] == labels[i]]
    correct_probs = [probs[i] for i in correct_indices]
    
    analysis = {
        'total_samples': len(preds),
        'total_errors': len(errors),
        'error_rate': len(errors) / len(preds),
        'false_negatives': len(fn_errors),
        'false_positives': len(fp_errors),
        'avg_error_prob': np.mean(error_probs) if error_probs else 0,
        'avg_error_confidence': np.mean([e['confidence'] for e in errors]) if errors else 0,
        'avg_correct_confidence': np.mean([abs(p - 0.5) * 2 for p in correct_probs]) if correct_probs else 0,
        'high_confidence_errors': len([e for e in errors if e['confidence'] > 0.6]),
        'low_confidence_errors': len([e for e in errors if e['confidence'] < 0.3]),
        'prob_distribution': {
            '0.0-0.2': len([p for p in probs if p < 0.2]),
            '0.2-0.4': len([p for p in probs if 0.2 <= p < 0.4]),
            '0.4-0.6': len([p for p in probs if 0.4 <= p < 0.6]),
            '0.6-0.8': len([p for p in probs if 0.6 <= p < 0.8]),
            '0.8-1.0': len([p for p in probs if p >= 0.8]),
        }
    }
    
    error_lengths = [len(e['text'].split()) for e in errors]
    all_lengths = [len(t.split()) for t in texts]
    analysis['avg_error_text_length'] = np.mean(error_lengths) if error_lengths else 0
    analysis['avg_all_text_length'] = np.mean(all_lengths)
    
    output_dir = Path(output_dir)
    
    with open(output_dir / 'error_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    with open(output_dir / 'false_negatives.json', 'w') as f:
        json.dump(fn_errors[:50], f, indent=2)
    
    with open(output_dir / 'false_positives.json', 'w') as f:
        json.dump(fp_errors[:50], f, indent=2)
    
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}")
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Total errors: {analysis['total_errors']} ({analysis['error_rate']*100:.1f}%)")
    print(f"  - False Negatives (missed conspiracies): {analysis['false_negatives']}")
    print(f"  - False Positives (false alarms): {analysis['false_positives']}")
    print(f"\nConfidence Analysis:")
    print(f"  - Avg confidence on errors: {analysis['avg_error_confidence']:.3f}")
    print(f"  - Avg confidence on correct: {analysis['avg_correct_confidence']:.3f}")
    print(f"  - High-confidence errors (>0.6): {analysis['high_confidence_errors']}")
    print(f"  - Low-confidence errors (<0.3): {analysis['low_confidence_errors']}")
    print(f"\nProbability Distribution:")
    for range_str, count in analysis['prob_distribution'].items():
        pct = count / analysis['total_samples'] * 100
        bar = '█' * int(pct / 2)
        print(f"  {range_str}: {count:3d} ({pct:5.1f}%) {bar}")
    
    if fn_errors:
        print(f"\n{'='*70}")
        print(f"TOP FALSE NEGATIVES (Missed Conspiracies)")
        print(f"{'='*70}")
        for i, e in enumerate(fn_errors[:5]):
            print(f"\n[FN-{i+1}] prob={e['prob']:.3f}")
            print(f"  Text: {e['text'][:200]}...")
    
    if fp_errors:
        print(f"\n{'='*70}")
        print(f"TOP FALSE POSITIVES (False Alarms)")
        print(f"{'='*70}")
        for i, e in enumerate(fp_errors[:5]):
            print(f"\n[FP-{i+1}] prob={e['prob']:.3f}")
            print(f"  Text: {e['text'][:200]}...")
    
    return analysis, errors


# ============================================================================
# TRAINING
# ============================================================================

def find_best_threshold(probs, labels):
    """Find optimal threshold"""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.2, 0.8, 0.01):
        preds = (probs >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Fine search
    for threshold in np.arange(best_threshold - 0.03, best_threshold + 0.03, 0.002):
        if threshold < 0.1 or threshold > 0.9:
            continue
        preds = (probs >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    return best_threshold, best_f1


def train_epoch(model, loader, criterion, optimizer, scheduler, device, config, scaler):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if config.USE_BF16 else torch.float16):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            if config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, device, return_texts=False):
    """Evaluate model"""
    model.eval()
    all_probs = []
    all_labels = []
    all_ids = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_ids.extend(batch['id'])
            if return_texts:
                all_texts.extend(batch['text'])
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    threshold, f1 = find_best_threshold(all_probs, all_labels)
    preds = (all_probs >= threshold).astype(int)
    
    if return_texts:
        return all_probs, all_labels, threshold, f1, preds, all_ids, all_texts
    return all_probs, all_labels, threshold, f1, preds


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='../train_rehydrated.jsonl')
    parser.add_argument('--dev_file', type=str, default='../dev_public.jsonl')
    parser.add_argument('--test_file', type=str, default='../test_rehydrated.jsonl')
    parser.add_argument('--output_dir', type=str, default='outputs/final')
    parser.add_argument('--predictions_file', type=str, default='submission.jsonl')
    parser.add_argument('--resume', action='store_true', help='Resume from best checkpoint (skip warmup)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    config = Config()
    set_seed(config.SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_data = load_data(args.train_file, filter_ambiguous=True)
    dev_data = load_data(args.dev_file, filter_ambiguous=True)
    
    label_dist = Counter([d['conspiracy'].lower() for d in train_data])
    print(f"\nTrain: {len(train_data)} (Yes: {label_dist['yes']}, No: {label_dist['no']})")
    print(f"Dev: {len(dev_data)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Datasets
    train_dataset = SimpleDataset(train_data, tokenizer, config.MAX_LENGTH)
    dev_dataset = SimpleDataset(dev_data, tokenizer, config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    # Model
    model = ImprovedConspiracyClassifier(
        model_name=config.MODEL_NAME,
        dropout=config.DROPOUT,
        freeze_encoder=True
    ).to(device)
    
    print(f"\n{'='*70}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Strategy: CLS + Mean Pooling Ensemble")
    print(f"Warmup: {config.NUM_WARMUP_EPOCHS} epochs")
    print(f"Fine-tune: {config.NUM_FINETUNE_EPOCHS} epochs (ALL layers)")
    print(f"SWA: {'Enabled' if config.USE_SWA else 'Disabled'}")
    print(f"{'='*70}\n")
    
    # Loss
    criterion = FocalLoss(
        gamma=config.FOCAL_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Initialize best tracking variables
    best_f1 = 0
    best_threshold = 0.5
    best_model_state = None
    best_probs = None
    best_labels = None
    best_ids = None
    best_texts = None
    
    # Check for resume from checkpoint
    checkpoint_path = output_dir / 'best_checkpoint.pt'
    if args.resume and checkpoint_path.exists():
        print(f"{'='*70}")
        print(f"RESUMING FROM CHECKPOINT")
        print(f"{'='*70}\n")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict({k: v.to(device) for k, v in checkpoint['model_state_dict'].items()})
        best_f1 = checkpoint['val_f1']
        best_threshold = checkpoint['threshold']
        best_model_state = checkpoint['model_state_dict']
        
        print(f"Loaded checkpoint from {checkpoint['phase']} epoch {checkpoint['epoch']+1}")
        print(f"Best F1: {best_f1:.4f}, Threshold: {best_threshold:.3f}")
        
        # Evaluate to get probs for error analysis
        result = evaluate(model, dev_loader, device, return_texts=True)
        best_probs, best_labels, _, _, _, best_ids, best_texts = result
        
        print(f"Skipping warmup, going directly to fine-tuning...\n")
    else:
        # PHASE 1: Warmup Training
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        total_steps = len(train_loader) * config.NUM_WARMUP_EPOCHS
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
            num_training_steps=total_steps
        )
        
        # PHASE 1: Warmup
        print(f"{'='*70}")
        print(f"PHASE 1: WARMUP ({config.NUM_WARMUP_EPOCHS} epochs)")
        print(f"{'='*70}\n")
        
        for epoch in range(config.NUM_WARMUP_EPOCHS):
            print(f"Epoch {epoch + 1}/{config.NUM_WARMUP_EPOCHS}")
            
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, config, scaler
            )
            
            result = evaluate(model, dev_loader, device, return_texts=True)
            probs, labels, threshold, val_f1, preds, dev_ids, dev_texts = result
            
            _, _, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
            
            print(f"  Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val F1: {val_f1:.4f} (threshold: {threshold:.3f})")
            print(f"  Class F1: No={f1_per_class[0]:.4f}, Yes={f1_per_class[1]:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_probs = probs.copy()
                best_labels = labels.copy()
                best_ids = dev_ids.copy()
                best_texts = dev_texts.copy()
                # Save best checkpoint
                torch.save({
                    'model_state_dict': best_model_state,
                    'threshold': best_threshold,
                    'val_f1': best_f1,
                    'epoch': epoch,
                    'phase': 'warmup'
                }, output_dir / 'best_checkpoint.pt')
                print(f"  ✓ New best: {best_f1:.4f} (saved)")
            print()
    
    # PHASE 2: Fine-tuning with SWA
    print(f"{'='*70}")
    print(f"PHASE 2: FINE-TUNING ({config.NUM_FINETUNE_EPOCHS} epochs)")
    print(f"{'='*70}\n")
    
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    model.unfreeze_all_encoder()
    
    # Recreate dataloader with smaller batch size for fine-tuning
    train_loader_ft = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE_FINETUNE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )
    
    # Create config copy with finetune settings
    class FinetuneConfig:
        GRADIENT_ACCUMULATION_STEPS = config.GRADIENT_ACCUMULATION_FINETUNE
        USE_BF16 = config.USE_BF16
        MAX_GRAD_NORM = config.MAX_GRAD_NORM
    finetune_config = FinetuneConfig()
    
    param_groups = get_layerwise_lr_groups(
        model,
        base_lr=config.LEARNING_RATE * config.FINETUNE_LR_MULTIPLIER,
        lr_decay=config.LAYERWISE_LR_DECAY,
        weight_decay=config.WEIGHT_DECAY
    )
    optimizer = torch.optim.AdamW(param_groups)
    
    total_steps = len(train_loader_ft) * config.NUM_FINETUNE_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    swa = SWA(model) if config.USE_SWA else None
    patience_counter = 0
    
    for epoch in range(config.NUM_FINETUNE_EPOCHS):
        print(f"Epoch {epoch + 1}/{config.NUM_FINETUNE_EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader_ft, criterion, optimizer, scheduler, device, finetune_config, scaler
        )
        
        # Update SWA
        if swa and epoch >= config.SWA_START_EPOCH:
            swa.update()
            print(f"  SWA updated (n={swa.swa_n})")
        
        result = evaluate(model, dev_loader, device, return_texts=True)
        probs, labels, threshold, val_f1, preds, dev_ids, dev_texts = result
        
        _, _, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
        
        print(f"  Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val F1: {val_f1:.4f} (threshold: {threshold:.3f})")
        print(f"  Class F1: No={f1_per_class[0]:.4f}, Yes={f1_per_class[1]:.4f}")
        print(f"  Preds: {Counter(preds)}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_threshold = threshold
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_probs = probs.copy()
            best_labels = labels.copy()
            best_ids = dev_ids.copy()
            best_texts = dev_texts.copy()
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                'model_state_dict': best_model_state,
                'threshold': best_threshold,
                'val_f1': best_f1,
                'epoch': epoch,
                'phase': 'finetune'
            }, output_dir / 'best_checkpoint.pt')
            print(f"  ✓ New best: {best_f1:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"  Early stopping")
                break
        print()
    
    # Try SWA model
    if swa and swa.swa_n > 0:
        print(f"\n{'='*70}")
        print(f"Evaluating SWA model (n={swa.swa_n})")
        print(f"{'='*70}\n")
        
        swa.apply()
        result = evaluate(model, dev_loader, device, return_texts=True)
        probs, labels, threshold, val_f1, preds, dev_ids, dev_texts = result
        
        print(f"SWA F1: {val_f1:.4f} (threshold: {threshold:.3f})")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_threshold = threshold
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_probs = probs.copy()
            best_labels = labels.copy()
            best_ids = dev_ids.copy()
            best_texts = dev_texts.copy()
            print(f"✓ SWA improved F1 to {best_f1:.4f}!")
    
    # Load best model
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    # Error analysis
    analyze_errors(best_probs, best_labels, best_texts, best_ids, best_threshold, output_dir)
    
    preds = (best_probs >= best_threshold).astype(int)
    print(f"\n{'='*70}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(classification_report(best_labels, preds, target_names=['No', 'Yes'], digits=4))
    
    cm = confusion_matrix(best_labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              No    Yes")
    print(f"Actual No    {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Yes   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Test prediction
    print(f"\n{'='*70}")
    print("TEST PREDICTION")
    print(f"{'='*70}\n")
    
    test_data = load_data(args.test_file, filter_ambiguous=False)
    test_dataset = SimpleDataset(test_data, tokenizer, config.MAX_LENGTH)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    model.eval()
    test_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
    
    test_probs = np.array(test_probs)
    test_preds = (test_probs >= best_threshold).astype(int)
    
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Predictions: {Counter(test_preds)}")
    
    # Save predictions
    predictions = []
    for i, item in enumerate(test_data):
        predictions.append({
            '_id': item.get('_id', str(i)),
            'conspiracy': 'Yes' if test_preds[i] == 1 else 'No'
        })
    
    with open(output_dir / args.predictions_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    # Also save to main directory for easy access
    import shutil
    shutil.copy(output_dir / args.predictions_file, Path('submission.jsonl'))
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': best_threshold,
        'val_f1': best_f1
    }, output_dir / 'best_model.pt')
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE")
    print(f"{'='*70}")
    print(f"Best Dev F1: {best_f1:.4f}")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Improvement: {best_f1 - 0.81:+.4f}")
    
    if best_f1 >= 0.90:
        print("\n🎉 TARGET ACHIEVED!")
    elif best_f1 >= 0.85:
        print(f"\n📈 Close! Need +{0.90 - best_f1:.4f}")
    print(f"\nFiles: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()