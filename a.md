You are an ML researcher asked to improve a **binary text classification** system for conspiracy theory detection.

# TASK: SemEval-2026 Task 10 Subtask 2

**Goal**: Classify Reddit posts as "Yes" (promotes/believes conspiracy) or "No" (does not).

**Key Distinction**: The model must distinguish between:

- Posts that **discuss/report** conspiracy topics -> "No"
- Posts that **promote/believe** conspiracy theories -> "Yes"

---

# BASELINE (READ CAREFULLY)

The current system includes:

- **DeBERTa-v3-large encoder** (434M params)
- **Mean pooling** over non-padding tokens
- **2-layer MLP classifier**: Linear(1024->512) -> GELU -> Linear(512->2)
- **Focal Loss** (gamma=2.0) with label smoothing (0.05)
- **2-phase training**: Warmup (frozen encoder) + Fine-tuning (top 6 layers)
- **Layerwise LR decay** (0.9x per layer)
- **Threshold optimization** on dev set

**Current Performance**: 0.81 F1 on dev (77 samples)
**Target**: 0.90+ F1

---

# HARD CONSTRAINTS

You are NOT allowed to:

- Change the base encoder (DeBERTa-v3-large)
- Use external data or APIs
- Use complex multi-task learning (already tried, hurt performance)
- Use marker-based features (test set has no markers)

You MUST propose improvements that are:

- Simple and additive to current architecture
- Empirically likely to improve F1
- Fast to train (<15 min on RTX 4080)

---

# ERROR ANALYSIS (CRITICAL)

## False Negatives (5 errors) - Missed Conspiracies

Posts with **implicit conspiracy content**, no explicit keywords:

1. "Weinstein army of spies covering up crimes" - written factually
2. "Germans labeled as Nazis by power structure" - persecution narrative
3. "UFO shot by military pilot" - factual reporting style
4. "Government deporting disabled children" - critique without conspiracy words

**Pattern**: Subtle implications of hidden agendas, elite malfeasance, cover-ups.

## False Positives (10 errors) - Wrong Predictions

Model confuses **discussing** with **believing**:

1. "James Alefantis Rothschild..." - describes Pizzagate, doesn't promote it
2. "Jeffrey Epstein's black book..." - reports on topic
3. "Are you SURE you're going to heaven?" - religious, not conspiracy
4. "Jesus and word of God is real" - religious content
5. "Tax havens where wealthy park money" - social commentary

**Pattern**: Keyword-triggered without understanding author intent.

## Calibration Issue

- 71% of predictions in 0.4-0.6 probability range
- Model is uncertain on most samples
- All errors are low-confidence (<0.3)

## Confusion Matrix (Dev Set)

```
                Predicted
                Yes    No
Actual  Yes     22     5   (FN)
        No      10    40   (FP)
```

**Metrics**:

- Precision: 22/(22+10) = 0.688
- Recall: 22/(22+5) = 0.815
- F1: 0.81
- Accuracy: 62/77 = 0.805

**Key Observation**: False Positives (10) > False Negatives (5). The model is biased toward predicting "Yes" when unsure.

---

# OBJECTIVE

Design ONE modification that would raise macro F1 from 0.81 to 0.90+ by:

1. Better distinguishing "discussing" vs "believing" conspiracy content
2. Improving detection of implicit/subtle conspiracies
3. Reducing false positives on religious and news-reporting content
4. Better calibration (push probabilities away from 0.5)

---

# REQUIRED OUTPUT FORMAT (STRICT)

## 1. Identified Bottleneck (Specific)

Name the exact limitation of the current approach.
Reference which component causes it (encoder, pooling, classifier, loss, threshold).

## 2. Proposed Improvement (Single Idea Only)

Pick exactly ONE:

- A contrastive learning objective
- An auxiliary classification task
- A different pooling strategy (CLS, attention-weighted)
- A calibration technique
- Data augmentation strategy
- Ensemble method

Explain how it composes with the existing system.

## 3. Implementation Details

Provide:

- Code changes needed
- New hyperparameters
- Training procedure modifications

## 4. Expected Impact

- Which error type does this fix? (FN, FP, calibration)
- Why will this work?
- What's the expected F1 improvement?

---

# DATASET INFO

| Split | Samples | Yes      | No       |
| ----- | ------- | -------- | -------- |
| Train | 3,531   | ~44%     | ~56%     |
| Dev   | 77      | 27 (35%) | 50 (65%) |
| Test  | 938     | ?        | ?        |

---

# WHAT HAS BEEN TRIED

| Approach                                 | Dev F1 | Notes          |
| ---------------------------------------- | ------ | -------------- |
| DeBERTa baseline (3 warmup + 8 finetune) | 0.81   | Current best   |
| More warmup epochs (5)                   | 0.79   | Worse          |
| Frozen encoder only                      | 0.77   | Worse          |
| Lower fine-tuning LR (0.2x)              | 0.79   | Worse          |
| Complex features (R-Drop, FGM, MTL)      | 0.74   | Much worse     |
| Higher dropout (0.2)                     | 0.80   | No improvement |

---

# KEY INSIGHT

The fundamental challenge is:

1. Dev set is tiny (77 samples) - high variance in eval
2. Model detects **keywords** not **intent**
3. Fine-tuning doesn't help much over warmup alone

The improvement must address the semantic understanding of "promoting" vs "discussing" conspiracy content.

---

# STYLE RULES

- No vague claims ("captures structure", "models context")
- No encoder swaps
- No hyperparameter fiddling disguised as novelty
- If it's not implementable in ~200 lines of PyTorch, reject it
- Answer as if your proposal will be implemented tomorrow

**PROMPT END**
