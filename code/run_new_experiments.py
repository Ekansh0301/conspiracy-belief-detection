"""
New experiments for paper:
  1. Marker-masked ablation  -- CLS vs mean pooling on text with marker spans removed
  2. RoBERTa-large baseline  -- mean pooling, same hyperparams
  3. DeBERTa-base baseline   -- mean pooling, same hyperparams

All single-seed (fast), evaluated on dev set.
Results saved to results/new_experiments.json
"""

import json, random, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
TRAIN_PATH = "/home/divya/Desktop/Conspiracy/train_rehydrated.jsonl"
DEV_PATH   = "/home/divya/Desktop/Conspiracy/dev_public.jsonl"
OUT_PATH   = "/home/divya/Desktop/Conspiracy/subtask2/results/new_experiments.json"

SEED          = 2026
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ─────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

def mask_markers(item):
    """Return item text with all marker character spans replaced by spaces."""
    text = item["text"]
    spans = item.get("markers", [])
    if not spans:
        return text
    chars = list(text)
    for m in spans:
        s, e = m["startIndex"], m["endIndex"]
        for i in range(min(s, len(chars)), min(e, len(chars))):
            chars[i] = " "
    return re.sub(r" {2,}", " ", "".join(chars)).strip()

def get_class_weights(labels):
    counts = np.bincount(labels)
    w = len(labels) / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32).to(DEVICE)

def label_of(item):
    c = item.get("conspiracy", "no")
    if isinstance(c, str):
        if c.lower() == "yes" or c.lower() == "can't tell":
            return 1
        return 0
    return 0

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, items, tokenizer, mask_marker_spans=False):
        self.items = items
        self.tokenizer = tokenizer
        self.mask_spans = mask_marker_spans

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        text = mask_markers(item) if self.mask_spans else item["text"]
        label = label_of(item)
        enc = self.tokenizer(text, max_length=MAX_LENGTH, padding="max_length",
                             truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class Classifier(nn.Module):
    def __init__(self, model_name, use_cls=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.use_cls = use_cls
        h = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(h, 512),
            nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(512, 2)
        )
        # Freeze all except top UNFREEZE_LAYERS
        for p in self.encoder.parameters():
            p.requires_grad = False
        nlayers = len(self.encoder.encoder.layer)
        for i in range(nlayers - UNFREEZE_LAYERS, nlayers):
            for p in self.encoder.encoder.layer[i].parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_cls:
            pooled = out.last_hidden_state[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled)

# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_and_eval(model_name, use_cls, train_items, dev_items, mask_spans=False, tag=""):
    seed_everything(SEED)
    print(f"\n{'='*60}\n[{tag}] model={model_name} cls={use_cls} mask={mask_spans}\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = TextDataset(train_items, tokenizer, mask_spans)
    dev_ds   = TextDataset(dev_items,   tokenizer, mask_spans)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=32,         shuffle=False, num_workers=2, pin_memory=True)

    model = Classifier(model_name, use_cls).to(DEVICE)
    labels_train = [label_of(x) for x in train_items]
    cw = get_class_weights(labels_train)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps
    )

    best_f1, best_ep, no_improve = 0.0, 0, 0
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            iids = batch["input_ids"].to(DEVICE)
            amask = batch["attention_mask"].to(DEVICE)
            labs = batch["label"].to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(iids, amask)
                loss = F.cross_entropy(logits, labs, weight=cw,
                                       label_smoothing=LABEL_SMOOTH) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()

        # Eval
        model.eval()
        all_preds, all_labs = [], []
        with torch.no_grad():
            for batch in dev_loader:
                iids = batch["input_ids"].to(DEVICE)
                amask = batch["attention_mask"].to(DEVICE)
                labs = batch["label"]
                with torch.cuda.amp.autocast():
                    logits = model(iids, amask)
                preds = logits.argmax(-1).cpu().tolist()
                all_preds.extend(preds); all_labs.extend(labs.tolist())
        f1 = f1_score(all_labs, all_preds, average="macro")
        print(f"  Epoch {ep}: dev macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_ep, no_improve = f1, ep, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {ep}.")
                break

    print(f"  Best F1 = {best_f1:.4f} at epoch {best_ep}")
    del model; torch.cuda.empty_cache()
    return round(best_f1, 4)

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    train_all = load_jsonl(TRAIN_PATH)
    dev_all   = load_jsonl(DEV_PATH)

    # Remap can't-tell to yes  (matching main training setup)
    for x in train_all:
        if str(x.get("conspiracy","")).lower() == "can't tell":
            x["conspiracy"] = "yes"

    # Only keep subtask2 items with conspiracy label
    train_items = [x for x in train_all if x.get("conspiracy","").lower() in ("yes","no")]
    dev_items   = [x for x in dev_all   if x.get("conspiracy","").lower() in ("yes","no")]

    print(f"Train: {len(train_items)}  Dev: {len(dev_items)}")

    results = {}

    # ── Experiment 1: Marker masking ablation ─────────────────
    # We need: (a) mean/original, (b) cls/original, (c) mean/masked, (d) cls/masked
    # (a) and (b) we essentially have from earlier, but re-run for a clean comparison 
    # under identical conditions in this script.
    deberta = "microsoft/deberta-v3-large"

    print("\n>>> MARKER MASKING ABLATION <<<")
    r_mean_orig = train_and_eval(deberta, use_cls=False, train_items=train_items,
                                  dev_items=dev_items, mask_spans=False,
                                  tag="mean-original")
    r_cls_orig  = train_and_eval(deberta, use_cls=True,  train_items=train_items,
                                  dev_items=dev_items, mask_spans=False,
                                  tag="cls-original")
    r_mean_mask = train_and_eval(deberta, use_cls=False, train_items=train_items,
                                  dev_items=dev_items, mask_spans=True,
                                  tag="mean-masked")
    r_cls_mask  = train_and_eval(deberta, use_cls=True,  train_items=train_items,
                                  dev_items=dev_items, mask_spans=True,
                                  tag="cls-masked")

    gap_orig = round(r_mean_orig - r_cls_orig, 4)
    gap_mask = round(r_mean_mask - r_cls_mask, 4)
    results["marker_masking"] = {
        "mean_original":  r_mean_orig,
        "cls_original":   r_cls_orig,
        "mean_masked":    r_mean_mask,
        "cls_masked":     r_cls_mask,
        "pooling_gap_original": gap_orig,
        "pooling_gap_masked":   gap_mask,
        "gap_reduction": round(gap_orig - gap_mask, 4),
    }
    print(f"\nPooling gap (original text): {gap_orig:+.4f}")
    print(f"Pooling gap (masked text):   {gap_mask:+.4f}")
    print(f"Gap reduction after masking: {gap_orig - gap_mask:+.4f}")

    # ── Experiment 2: RoBERTa-large baseline ──────────────────
    print("\n>>> ROBERTA-LARGE BASELINE <<<")
    r_roberta = train_and_eval("roberta-large", use_cls=False, train_items=train_items,
                                dev_items=dev_items, mask_spans=False,
                                tag="roberta-large-mean")
    results["roberta_large_mean"] = r_roberta

    # ── Experiment 3: DeBERTa-base baseline ───────────────────
    print("\n>>> DEBERTA-BASE BASELINE <<<")
    r_deberta_base = train_and_eval("microsoft/deberta-v3-base", use_cls=False,
                                     train_items=train_items, dev_items=dev_items,
                                     mask_spans=False, tag="deberta-base-mean")
    results["deberta_base_mean"] = r_deberta_base

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(json.dumps(results, indent=2))

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
