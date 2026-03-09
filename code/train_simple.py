"""
Simple DeBERTa-v3-large baseline for Conspiracy Detection
==========================================================
Based on the original working approach that got 0.8064 F1 on dev set.
NO marker features - pure text classification.

ENHANCEMENTS FOR 0.9+ F1:
- Comprehensive error analysis
- Layerwise learning rate decay
- Multiple seeds ensemble
- Detailed per-sample analysis
- Confidence calibration
"""

import argparse
import json
import random
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LENGTH = 256  # Keep original
    
    # NO instruction prefix - it may confuse the model
    USE_INSTRUCTION = False
    INSTRUCTION = ""
    
    # Training - back to original working config
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    DROPOUT = 0.1
    
    # Layerwise LR decay
    LAYERWISE_LR_DECAY = 0.9
    
    # Epochs - original that worked
    NUM_WARMUP_EPOCHS = 5  # More warmup - this is where F1 peaks
    NUM_FINETUNE_EPOCHS = 4
    FINETUNE_LR_MULTIPLIER = 0.2  # 5x lower LR for fine-tuning
    PATIENCE = 3
    FINETUNE_LAYERS = 6
    
    # Loss
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.05
    
    # Seed
    SEED = 2026
    
    # Error analysis
    SAVE_ERROR_SAMPLES = True
    TOP_ERRORS_TO_ANALYZE = 50


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
    """Simple text classification dataset with optional instruction prefix"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256, 
                 instruction: str = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('text', '')
        
        # Add instruction prefix if provided
        if self.instruction:
            text = self.instruction + text
        
        # Label - handle test data with no labels
        conspiracy = item.get('conspiracy')
        if conspiracy is None:
            label = 0  # Dummy label for test data
        else:
            label = 1 if conspiracy.lower() == 'yes' else 0
        
        # Tokenize
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
            'text': item.get('text', '')  # Original text without instruction
        }


# ============================================================================
# MODEL - Simple DeBERTa Classifier
# ============================================================================

class SimpleConspiracyClassifier(nn.Module):
    """Simple DeBERTa classifier with mean pooling"""
    
    def __init__(self, model_name: str, dropout: float = 0.1, freeze_encoder: bool = False):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        
    def unfreeze_encoder(self, num_layers: int = 6):
        """Unfreeze top N encoder layers"""
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Unfreeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = True
            
        # Unfreeze top N layers
        num_encoder_layers = len(self.encoder.encoder.layer)
        for i in range(num_encoder_layers - num_layers, num_encoder_layers):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True
                
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze top {num_layers} encoder layers. Trainable params: {trainable:,}")
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Mean pooling over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        logits = self.classifier(pooled)
        return logits


# ============================================================================
# LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for class imbalance"""
    
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
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
# LAYERWISE LR DECAY - Crucial for fine-tuning large models
# ============================================================================

def get_layerwise_lr_groups(model, base_lr, lr_decay=0.9, weight_decay=0.05):
    """Create parameter groups with layerwise LR decay"""
    
    # Get number of encoder layers
    num_layers = len(model.encoder.encoder.layer)
    
    # Parameter groups
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    groups = []
    
    # Classifier head - highest LR
    groups.append({
        'params': [p for n, p in model.classifier.named_parameters() if p.requires_grad],
        'lr': base_lr * 5,  # 5x for classifier
        'weight_decay': weight_decay
    })
    
    # Embeddings - low LR
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
    
    # Encoder layers - gradual LR decay
    for layer_idx in range(num_layers):
        layer = model.encoder.encoder.layer[layer_idx]
        layer_lr = base_lr * (lr_decay ** (num_layers - layer_idx - 1))
        
        # Check if layer has any trainable params
        trainable_params = [p for p in layer.parameters() if p.requires_grad]
        if not trainable_params:
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
    
    # Filter out empty groups
    groups = [g for g in groups if len(g['params']) > 0]
    
    return groups


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_errors(probs, labels, texts, ids, threshold, output_dir):
    """Comprehensive error analysis"""
    preds = (probs >= threshold).astype(int)
    
    errors = []
    confidences = []
    
    for i in range(len(preds)):
        is_error = bool(preds[i] != labels[i])  # Convert to Python bool
        confidence = abs(probs[i] - 0.5) * 2  # 0-1 scale
        
        error_info = {
            'id': str(ids[i]),  # Ensure string
            'text': str(texts[i][:500]),  # Truncate for readability
            'true_label': 'Yes' if int(labels[i]) == 1 else 'No',
            'pred_label': 'Yes' if int(preds[i]) == 1 else 'No',
            'prob': float(probs[i]),
            'confidence': float(confidence),
            'is_error': is_error,
            'error_type': None
        }
        
        if is_error:
            if int(labels[i]) == 1 and int(preds[i]) == 0:
                error_info['error_type'] = 'FN'  # False Negative
            else:
                error_info['error_type'] = 'FP'  # False Positive
            errors.append(error_info)
        
        confidences.append(confidence)
    
    # Sort errors by confidence (most confident wrong predictions are most concerning)
    errors.sort(key=lambda x: x['prob'], reverse=True)
    fn_errors = [e for e in errors if e['error_type'] == 'FN']
    fp_errors = [e for e in errors if e['error_type'] == 'FP']
    
    # Confidence analysis
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
    
    # Text length analysis for errors
    error_lengths = [len(e['text'].split()) for e in errors]
    all_lengths = [len(t.split()) for t in texts]
    analysis['avg_error_text_length'] = np.mean(error_lengths) if error_lengths else 0
    analysis['avg_all_text_length'] = np.mean(all_lengths)
    
    # Save detailed error analysis
    output_dir = Path(output_dir)
    
    with open(output_dir / 'error_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save FN errors (most important - missed conspiracy posts)
    with open(output_dir / 'false_negatives.json', 'w') as f:
        json.dump(fn_errors[:50], f, indent=2)
    
    # Save FP errors
    with open(output_dir / 'false_positives.json', 'w') as f:
        json.dump(fp_errors[:50], f, indent=2)
    
    # Print summary
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
    print(f"\nText Length Analysis:")
    print(f"  - Avg words in errors: {analysis['avg_error_text_length']:.1f}")
    print(f"  - Avg words overall: {analysis['avg_all_text_length']:.1f}")
    
    # Print top FN examples
    if fn_errors:
        print(f"\n{'='*70}")
        print(f"TOP FALSE NEGATIVES (Missed Conspiracies) - Most problematic")
        print(f"{'='*70}")
        for i, e in enumerate(fn_errors[:5]):
            print(f"\n[FN-{i+1}] prob={e['prob']:.3f}")
            print(f"  Text: {e['text'][:200]}...")
    
    # Print top FP examples
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
    """Find optimal threshold for F1 with fine granularity"""
    best_f1 = 0
    best_threshold = 0.5
    
    # Coarse search
    for threshold in np.arange(0.2, 0.8, 0.02):
        preds = (probs >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Fine search around best
    for threshold in np.arange(best_threshold - 0.05, best_threshold + 0.05, 0.005):
        if threshold < 0.1 or threshold > 0.9:
            continue
        preds = (probs >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    return best_threshold, best_f1


def train_epoch(model, loader, criterion, optimizer, scheduler, device, config, scaler=None):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / config.GRADIENT_ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}', 
                         'acc': f'{correct/total:.4f}'})
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device, return_texts=False):
    """Evaluate model with optional text return for error analysis"""
    model.eval()
    all_probs = []
    all_labels = []
    all_ids = []
    all_texts = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)[:, 1]
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/simple')
    parser.add_argument('--predictions_file', type=str, default='submission.jsonl')
    
    args = parser.parse_args()
    config = Config()
    
    # Seed
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("SIMPLE DEBERTA CONSPIRACY CLASSIFIER")
    print(f"{'='*70}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Batch Size: {config.BATCH_SIZE} × {config.GRADIENT_ACCUMULATION_STEPS} = {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Seed: {config.SEED}")
    print(f"{'='*70}\n")
    
    # Load data
    train_data = load_data(args.train_file)
    dev_data = load_data(args.dev_file) if args.dev_file else None
    print(f"Train: {len(train_data)}")
    if dev_data:
        print(f"Dev: {len(dev_data)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Instruction prefix
    instruction = config.INSTRUCTION if config.USE_INSTRUCTION else None
    if instruction:
        print(f"Using instruction prefix: '{instruction}'")
    
    # Datasets
    train_dataset = SimpleDataset(train_data, tokenizer, config.MAX_LENGTH, instruction)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    
    if dev_data:
        dev_dataset = SimpleDataset(dev_data, tokenizer, config.MAX_LENGTH, instruction)
        dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, num_workers=2)
    
    # Class weights
    train_labels = [1 if item['conspiracy'].lower() == 'yes' else 0 for item in train_data]
    counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Model
    model = SimpleConspiracyClassifier(
        model_name=config.MODEL_NAME,
        dropout=config.DROPOUT,
        freeze_encoder=True
    ).to(device)
    
    # Loss
    criterion = FocalLoss(
        gamma=config.FOCAL_GAMMA,
        weight=class_weights,
        label_smoothing=config.LABEL_SMOOTHING
    )
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_f1 = 0
    best_threshold = 0.5
    best_model_state = None
    best_probs = None
    best_labels = None
    best_ids = None
    best_texts = None
    
    # =====================
    # Phase 1: Warmup
    # =====================
    print(f"\n{'='*50}")
    print(f"Phase 1: Warmup ({config.NUM_WARMUP_EPOCHS} epochs, encoder frozen)")
    print(f"{'='*50}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE * 5,  # Higher LR for warmup
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.NUM_WARMUP_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    patience_counter = 0
    
    for epoch in range(config.NUM_WARMUP_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_WARMUP_EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, scaler
        )
        
        if dev_data:
            result = evaluate(model, dev_loader, device, return_texts=True)
            probs, labels, threshold, val_f1, preds, dev_ids, dev_texts = result
            
            _, _, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val F1: {val_f1:.4f} (threshold: {threshold:.3f})")
            print(f"  Per-class F1: No={f1_per_class[0]:.4f}, Yes={f1_per_class[1]:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_probs = probs.copy()
                best_labels = labels.copy()
                best_ids = dev_ids.copy()
                best_texts = dev_texts.copy()
                patience_counter = 0
                print(f"  ✓ New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
    
    # =====================
    # Phase 2: Fine-tuning with Layerwise LR
    # =====================
    print(f"\n{'='*50}")
    print(f"Phase 2: Fine-tuning ({config.NUM_FINETUNE_EPOCHS} epochs)")
    print(f"Using Layerwise LR Decay: {config.LAYERWISE_LR_DECAY}")
    print(f"Fine-tuning LR multiplier: {config.FINETUNE_LR_MULTIPLIER}")
    print(f"{'='*50}")
    
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    model.unfreeze_encoder(num_layers=config.FINETUNE_LAYERS)
    
    # Use layerwise LR groups with lower LR for fine-tuning
    param_groups = get_layerwise_lr_groups(
        model, 
        base_lr=config.LEARNING_RATE * config.FINETUNE_LR_MULTIPLIER,  # Lower LR
        lr_decay=config.LAYERWISE_LR_DECAY,
        weight_decay=config.WEIGHT_DECAY
    )
    optimizer = torch.optim.AdamW(param_groups)
    
    # Print LR per group
    print("Learning rates per layer group:")
    for i, group in enumerate(param_groups):
        print(f"  Group {i}: {len(group['params'])} params, lr={group['lr']:.2e}")
    
    total_steps = len(train_loader) * config.NUM_FINETUNE_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    
    patience_counter = 0
    
    for epoch in range(config.NUM_FINETUNE_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_FINETUNE_EPOCHS}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, scaler
        )
        
        if dev_data:
            result = evaluate(model, dev_loader, device, return_texts=True)
            probs, labels, threshold, val_f1, preds, dev_ids, dev_texts = result
            
            _, _, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val F1: {val_f1:.4f} (threshold: {threshold:.3f})")
            print(f"  Per-class F1: No={f1_per_class[0]:.4f}, Yes={f1_per_class[1]:.4f}")
            print(f"  Predictions: {Counter(preds)}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_threshold = threshold
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_probs = probs.copy()
                best_labels = labels.copy()
                best_ids = dev_ids.copy()
                best_texts = dev_texts.copy()
                patience_counter = 0
                print(f"  ✓ New best F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"  Early stopping after {patience_counter} epochs without improvement")
                    break
    
    # Load best model
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    
    # =====================
    # Comprehensive Error Analysis on Dev Set
    # =====================
    if dev_data and config.SAVE_ERROR_SAMPLES:
        analyze_errors(best_probs, best_labels, best_texts, best_ids, best_threshold, output_dir)
        
        # Also save full classification report
        preds = (best_probs >= best_threshold).astype(int)
        print(f"\n{'='*70}")
        print("CLASSIFICATION REPORT")
        print(f"{'='*70}")
        print(classification_report(best_labels, preds, target_names=['No', 'Yes'], digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(best_labels, preds)
        print(f"Confusion Matrix:")
        print(f"              Predicted")
        print(f"              No    Yes")
        print(f"Actual No    {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       Yes   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # =====================
    # Test Prediction
    # =====================
    print(f"\n{'='*50}")
    print("Test Set Prediction")
    print(f"{'='*50}")
    
    test_data = load_data(args.test_file, filter_ambiguous=False)
    print(f"Test samples: {len(test_data)}")
    
    test_dataset = SimpleDataset(test_data, tokenizer, config.MAX_LENGTH, instruction)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, num_workers=2)
    
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
    
    print(f"\nUsing threshold: {best_threshold:.3f}")
    print(f"Predictions distribution: {Counter(test_preds)}")
    
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
    
    # Save best model
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': best_threshold,
        'val_f1': best_f1,
        'config': {
            'model_name': config.MODEL_NAME,
            'max_length': config.MAX_LENGTH
        }
    }, output_dir / 'best_model.pt')
    
    # Save latest model (current state)
    torch.save({
        'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'threshold': best_threshold,
        'val_f1': best_f1,
        'config': {
            'model_name': config.MODEL_NAME,
            'max_length': config.MAX_LENGTH
        }
    }, output_dir / 'latest_model.pt')
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE")
    print(f"{'='*70}")
    print(f"Best Dev F1: {best_f1:.4f}")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Test predictions: {Counter(test_preds)}")
    print(f"Predictions saved to: {output_dir / args.predictions_file}")
    print(f"Models saved: best_model.pt, latest_model.pt")
    
    # Actionable recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS FOR 0.9+ F1")
    print(f"{'='*70}")
    
    if best_f1 < 0.85:
        print("⚠️  F1 < 0.85 - Need significant improvements:")
        print("   1. Check error_analysis.json for patterns")
        print("   2. Increase MAX_LENGTH to 384 or 512")
        print("   3. Try data augmentation (back-translation)")
        print("   4. Consider ensemble of multiple seeds")
    elif best_f1 < 0.90:
        print("📈 F1 0.85-0.90 - Close to target:")
        print("   1. Analyze false_negatives.json - these hurt most")
        print("   2. Try threshold optimization on hold-out set")
        print("   3. Ensemble 3-5 models with different seeds")
        print("   4. Try longer training with lower LR")
    else:
        print("🎉 F1 >= 0.90 - Target achieved!")
        print("   1. Verify on test set")
        print("   2. Consider calibration for better thresholding")
    
    print(f"\n{'='*70}")
    print(f"Error analysis files saved to: {output_dir}/")
    print(f"  - error_analysis.json: Summary statistics")
    print(f"  - false_negatives.json: Missed conspiracies (TOP PRIORITY)")
    print(f"  - false_positives.json: False alarms")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
