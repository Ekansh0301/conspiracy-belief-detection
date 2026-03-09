"""
Proper CV comparison: mean pooling vs CLS pooling, both with identical H_conservative config.
Also runs 5-seed multi-seed dev eval with CLS to compare with mean's 0.829 ensemble.

H_conservative config:
  single-phase, label_smoothing=0.10, lr=2e-5, epochs=9, patience=4,
  unfreeze_layers=6, max_length=256, batch_size=16, dropout=0.1
"""

import json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# ── Config (H_conservative) ──────────────────────────────────
TRAIN_PATH = "/home/divya/Desktop/Conspiracy/train_rehydrated.jsonl"
DEV_PATH   = "/home/divya/Desktop/Conspiracy/dev_public.jsonl"
OUT_PATH   = "/home/divya/Desktop/Conspiracy/subtask2/results/proper_pooling_cv.json"
MODEL_NAME = "microsoft/deberta-v3-large"

MAX_LENGTH    = 256
BATCH_SIZE    = 16
GRAD_ACCUM    = 2
LR            = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
MAX_GRAD_NORM = 1.0
DROPOUT       = 0.1
NUM_EPOCHS    = 9
PATIENCE      = 4
LABEL_SMOOTH  = 0.10
UNFREEZE_LAYERS = 6
SEEDS = [2026, 42, 1337, 7, 2024]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def label_of(item):
    c = item.get("conspiracy", "no")
    if isinstance(c, str):
        if c.lower() in ("yes", "can't tell"):
            return 1
        return 0
    return 0

class TextDataset(Dataset):
    def __init__(self, items, tokenizer):
        self.items = items
        self.tok = tokenizer
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        enc = self.tok(it["text"], max_length=MAX_LENGTH, truncation=True,
                       padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k,v in enc.items()}, label_of(it)

class PoolClassifier(nn.Module):
    def __init__(self, use_cls=False):
        super().__init__()
        self.use_cls = use_cls
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, 1)
        )
        # Freeze all, then unfreeze top layers
        for p in self.encoder.parameters():
            p.requires_grad = False
        layers = getattr(self.encoder, 'encoder', self.encoder).layer
        for layer in layers[-UNFREEZE_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        if self.use_cls:
            rep = out.last_hidden_state[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            rep = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return self.classifier(rep).squeeze(-1)

def label_smooth_bce(logits, labels, smooth=LABEL_SMOOTH):
    soft = labels.float() * (1 - smooth) + smooth * 0.5
    return F.binary_cross_entropy_with_logits(logits, soft)

def train_and_eval(train_items, val_items, tokenizer, use_cls, seed):
    seed_everything(seed)
    ds_train = TextDataset(train_items, tokenizer)
    ds_val   = TextDataset(val_items,   tokenizer)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=32)

    model = PoolClassifier(use_cls=use_cls).to(DEVICE)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    total_steps = (len(dl_train) // GRAD_ACCUM) * NUM_EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(WARMUP_RATIO * total_steps), total_steps)

    best_f1, patience_ctr = 0.0, 0
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        model.train()
        opt.zero_grad()
        for step, (batch, labs) in enumerate(dl_train):
            batch = {k: v.to(DEVICE) for k,v in batch.items()}
            labs  = labs.float().to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(**batch)
                loss   = label_smooth_bce(logits, labs) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch, labs in dl_val:
                batch = {k: v.to(DEVICE) for k,v in batch.items()}
                logits = model(**batch)
                preds.extend((torch.sigmoid(logits) > 0.5).cpu().int().tolist())
                trues.extend(labs.tolist())
        f1 = f1_score(trues, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    del model; torch.cuda.empty_cache()
    return best_f1

# ── Main ─────────────────────────────────────────────────────
train_data = load_jsonl(TRAIN_PATH)
dev_data   = load_jsonl(DEV_PATH)
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

labels = np.array([label_of(x) for x in train_data])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# ── 1. CV: mean pooling (H_conservative) ─────────────────────
print("\n" + "="*60)
print("CV: MEAN POOLING (H_conservative config)")
print("="*60)
fold_f1s_mean = []
for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_data, labels)):
    tr = [train_data[i] for i in tr_idx]
    va = [train_data[i] for i in va_idx]
    seed = SEEDS[fold_idx % len(SEEDS)]
    t0 = time.time()
    f1 = train_and_eval(tr, va, tokenizer, use_cls=False, seed=seed)
    fold_f1s_mean.append(f1)
    print(f"  Fold {fold_idx}: F1={f1:.4f}  seed={seed}  elapsed={time.time()-t0:.0f}s")

mean_cv_mean = float(np.mean(fold_f1s_mean))
mean_cv_std  = float(np.std(fold_f1s_mean))
print(f"\nMEAN POOL CV: {mean_cv_mean:.4f} ± {mean_cv_std:.4f}")
results["mean_pooling_cv"] = {
    "fold_f1s": [float(f) for f in fold_f1s_mean],
    "mean_f1": mean_cv_mean,
    "std_f1": mean_cv_std
}

# ── 2. CV: CLS pooling (same H_conservative config) ──────────
print("\n" + "="*60)
print("CV: CLS POOLING (H_conservative config)")
print("="*60)
fold_f1s_cls = []
for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(train_data, labels)):
    tr = [train_data[i] for i in tr_idx]
    va = [train_data[i] for i in va_idx]
    seed = SEEDS[fold_idx % len(SEEDS)]
    t0 = time.time()
    f1 = train_and_eval(tr, va, tokenizer, use_cls=True, seed=seed)
    fold_f1s_cls.append(f1)
    print(f"  Fold {fold_idx}: F1={f1:.4f}  seed={seed}  elapsed={time.time()-t0:.0f}s")

cls_cv_mean = float(np.mean(fold_f1s_cls))
cls_cv_std  = float(np.std(fold_f1s_cls))
print(f"\nCLS POOL CV: {cls_cv_mean:.4f} ± {cls_cv_std:.4f}")
results["cls_pooling_cv"] = {
    "fold_f1s": [float(f) for f in fold_f1s_cls],
    "mean_f1": cls_cv_mean,
    "std_f1": cls_cv_std
}

gap = mean_cv_mean - cls_cv_mean
print(f"\nGAP (mean - CLS): {gap:.4f} pts")
mean_wins = sum(m > c for m,c in zip(fold_f1s_mean, fold_f1s_cls))
print(f"Mean pooling wins in {mean_wins}/5 folds")

results["gap_mean_minus_cls"] = float(gap)
results["mean_wins_folds"] = mean_wins

# ── 3. 5-seed dev: CLS pooling (for ensemble comparison) ──────
print("\n" + "="*60)
print("5-SEED DEV: CLS POOLING (compare with mean's 0.829)")
print("="*60)
seed_f1s_cls = []
for s in SEEDS:
    t0 = time.time()
    f1 = train_and_eval(train_data, dev_data, tokenizer, use_cls=True, seed=s)
    seed_f1s_cls.append(f1)
    print(f"  seed={s}: F1={f1:.4f}  elapsed={time.time()-t0:.0f}s")

seed_mean_cls = float(np.mean(seed_f1s_cls))
seed_std_cls  = float(np.std(seed_f1s_cls))
print(f"\nCLS 5-seed dev: {seed_mean_cls:.4f} ± {seed_std_cls:.4f}")
results["cls_5seed_dev"] = {
    "seed_f1s": {str(s): float(f) for s,f in zip(SEEDS, seed_f1s_cls)},
    "mean_f1": seed_mean_cls,
    "std_f1": seed_std_cls
}

# ── Save ─────────────────────────────────────────────────────
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT_PATH, "w"), indent=2)
print(f"\n✓ Results saved to {OUT_PATH}")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Mean pooling CV:  {mean_cv_mean:.4f} ± {mean_cv_std:.4f}")
print(f"CLS pooling CV:   {cls_cv_mean:.4f} ± {cls_cv_std:.4f}")
print(f"Gap:              {gap:+.4f} pts ({'mean wins' if gap > 0 else 'CLS wins'})")
print(f"Mean wins folds:  {mean_wins}/5")
print(f"CLS 5-seed dev:   {seed_mean_cls:.4f} ± {seed_std_cls:.4f}")
print(f"Mean 5-seed dev:  0.7848 (from improved_ensemble_H.json — for reference)")
