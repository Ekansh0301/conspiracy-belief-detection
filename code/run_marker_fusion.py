"""
Marker Feature Fusion experiment.
Appends 6 normalized marker count features (actor, action, effect, evidence,
victim, total) to the mean-pooled DeBERTa representation before the classifier.

Architecture: DeBERTa-v3-large mean pool (1024) → concat marker counts (6) → MLP → binary

Runs:
  1. Mean pool (baseline, H_conservative) – for direct comparison
  2. Mean pool + marker counts (fusion)

Evaluates both on dev set (single seed 2026) and reports F1, P, R, Acc.
Results saved to results/marker_fusion.json
"""

import json, random, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support

TRAIN_PATH = "/home/divya/Desktop/Conspiracy/train_rehydrated.jsonl"
DEV_PATH   = "/home/divya/Desktop/Conspiracy/dev_public.jsonl"
OUT_PATH   = "/home/divya/Desktop/Conspiracy/subtask2/results/marker_fusion.json"

# H_conservative config
SEED            = 2026
MAX_LENGTH      = 256
BATCH_SIZE      = 16
GRAD_ACCUM      = 2
LR              = 2e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
MAX_GRAD_NORM   = 1.0
DROPOUT         = 0.1
NUM_EPOCHS      = 9
PATIENCE        = 4
LABEL_SMOOTH    = 0.10
UNFREEZE_LAYERS = 6
MODEL_NAME      = "microsoft/deberta-v3-large"
MARKER_TYPES    = ["actor", "action", "effect", "evidence", "victim"]
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def label_of(item):
    c = item.get("conspiracy", "no")
    if isinstance(c, str) and c.lower() in ("yes", "can't tell"):
        return 1
    return 0

def get_marker_vec(item):
    """Return normalized 6-dim vector [actor, action, effect, evidence, victim, total]."""
    spans = item.get("markers", [])
    counts = Counter(s.get("type", "").lower() for s in spans)
    vec = [counts.get(m, 0) for m in MARKER_TYPES]
    total = sum(vec)
    vec.append(total)
    # Normalize by /10 to keep values in ~0-1 range
    return [v / 10.0 for v in vec]

class TextDataset(Dataset):
    def __init__(self, items, tokenizer, use_markers=False):
        self.items = items
        self.tok = tokenizer
        self.use_markers = use_markers

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        enc = self.tok(it["text"], max_length=MAX_LENGTH, truncation=True,
                       padding="max_length", return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        marker_vec = torch.tensor(get_marker_vec(it), dtype=torch.float32) if self.use_markers else None
        return enc, marker_vec, label_of(it)

def collate_fn(batch):
    encs, markers, labels = zip(*batch)
    enc_batch = {k: torch.stack([e[k] for e in encs]) for k in encs[0]}
    labels = torch.tensor(labels, dtype=torch.long)
    if markers[0] is not None:
        markers = torch.stack(markers)
    else:
        markers = None
    return enc_batch, markers, labels

class MeanPoolClassifier(nn.Module):
    def __init__(self, use_markers=False):
        super().__init__()
        self.use_markers = use_markers
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size  # 1024
        marker_dim = len(MARKER_TYPES) + 1 if use_markers else 0
        in_dim = hidden + marker_dim
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, 1)
        )
        # Freeze all except top UNFREEZE_LAYERS
        for param in self.encoder.parameters():
            param.requires_grad = False
        for layer in self.encoder.encoder.layer[-UNFREEZE_LAYERS:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None, marker_vec=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        if self.use_markers and marker_vec is not None:
            pooled = torch.cat([pooled, marker_vec], dim=-1)
        return self.classifier(pooled).squeeze(-1)

def label_smooth_bce(logits, labels, eps=LABEL_SMOOTH):
    soft = labels.float() * (1 - eps) + eps / 2
    return nn.functional.binary_cross_entropy_with_logits(logits, soft)

def train_and_eval(tr_data, de_data, tokenizer, use_markers, seed):
    seed_everything(seed)
    ds_tr = TextDataset(tr_data, tokenizer, use_markers)
    ds_de = TextDataset(de_data, tokenizer, use_markers)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dl_de = DataLoader(ds_de, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = MeanPoolClassifier(use_markers=use_markers).to(DEVICE)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(dl_tr) // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda')

    best_f1, best_preds, patience_ctr, best_epoch = 0, [], 0, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        opt.zero_grad()
        for step, (enc, mvec, labs) in enumerate(dl_tr):
            input_ids = enc["input_ids"].to(DEVICE)
            attn_mask = enc["attention_mask"].to(DEVICE)
            ttids = enc.get("token_type_ids")
            if ttids is not None: ttids = ttids.to(DEVICE)
            mvec_d = mvec.to(DEVICE) if mvec is not None else None
            labs = labs.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attn_mask, ttids, mvec_d)
                loss = label_smooth_bce(logits, labs) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for enc, mvec, labs in dl_de:
                input_ids = enc["input_ids"].to(DEVICE)
                attn_mask = enc["attention_mask"].to(DEVICE)
                ttids = enc.get("token_type_ids")
                if ttids is not None: ttids = ttids.to(DEVICE)
                mvec_d = mvec.to(DEVICE) if mvec is not None else None
                logits = model(input_ids, attn_mask, ttids, mvec_d)
                all_probs.extend(torch.sigmoid(logits).cpu().tolist())
                all_labels.extend(labs.tolist())

        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        f1 = f1_score(all_labels, preds, average="macro")
        print(f"  Epoch {epoch+1}: F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_preds = f1, preds
            patience_ctr, best_epoch = 0, epoch + 1
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}. Best F1={best_f1:.4f} at epoch {best_epoch}")
                break

    p, r, _, _ = precision_recall_fscore_support(all_labels, best_preds, average="macro")
    acc = sum(a == b for a, b in zip(all_labels, best_preds)) / len(all_labels)
    return {"f1": float(best_f1), "precision": float(p), "recall": float(r), "accuracy": float(acc)}


print("Loading data and tokenizer...")
train_data = load_jsonl(TRAIN_PATH)
dev_data   = load_jsonl(DEV_PATH)
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\n" + "="*60)
print("RUN 1: Mean pooling ONLY (H_conservative, baseline)")
print("="*60)
t0 = time.time()
res_baseline = train_and_eval(train_data, dev_data, tokenizer, use_markers=False, seed=SEED)
print(f"BASELINE: {res_baseline}  ({time.time()-t0:.0f}s)")

print("\n" + "="*60)
print("RUN 2: Mean pooling + MARKER COUNTS (fusion)")
print("="*60)
t0 = time.time()
res_fusion = train_and_eval(train_data, dev_data, tokenizer, use_markers=True, seed=SEED)
print(f"FUSION:   {res_fusion}  ({time.time()-t0:.0f}s)")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
gain = res_fusion["f1"] - res_baseline["f1"]
print(f"Baseline (mean pool)   : F1={res_baseline['f1']:.4f}")
print(f"Fusion  (mean+markers) : F1={res_fusion['f1']:.4f}")
print(f"Delta                  : {gain:+.4f} pts ({'improvement' if gain>0 else 'no improvement'})")

results = {
    "baseline_mean": res_baseline,
    "fusion_mean_plus_markers": res_fusion,
    "delta_f1": float(gain)
}
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(OUT_PATH, "w"), indent=2)
print(f"\n✓ Saved to {OUT_PATH}")
