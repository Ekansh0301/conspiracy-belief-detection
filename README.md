# SemEval-2026 Task 10 — Subtask 2: Conspiracy Belief Detection

**Team:** Truth Gradient · Ekansh Goyal, IIIT Hyderabad  
**Task:** [PsyCoMark — SemEval-2026 Task 10](https://sites.google.com/view/semeval2026-task10)  
**Paper:** [`paper/latex/main.pdf`](paper/latex/main.pdf)

| Metric | Score |
|---|---|
| **Official Test Macro F1** | **0.750** |
| Dev Macro F1 (τ = 0.595) | 0.829 |
| Dev Precision / Recall / Accuracy | 0.829 / 0.829 / 0.844 |
| Dev bootstrap 95% CI | [0.712, 0.893] |
| 5-fold CV | 0.734 ± 0.007 |

---

## Task

Binary classification: given a Reddit post, determine whether the author **believes** a conspiracy theory (`Yes`) or merely **discusses / reports** it (`No`). Metric: macro F1.

---

## Key Finding: Narrative Density

Believers use significantly more psycholinguistic markers per post (6.53 ± 3.53) than non-believers (4.35 ± 4.29) — Cohen's *d* = 0.56, Mann-Whitney *p* < 10⁻⁸⁰, across all five marker categories (Actor, Action, Effect, Evidence, Victim). We call this **narrative density**: conspiracy believers construct structurally *more complete* conspiratorial frames, not just more topically relevant text.

Two causal tests support this:

| Test | Result |
|---|---|
| Marker masking (replace spans → whitespace) | F1 drops .723 → .670 (−5.3 pp) |
| Marker-count fusion (+6 normalized features) | F1 rises .670 → .679 (+0.9 pp) |

This motivates **sequence-level mean pooling** over the full post rather than a single `[CLS]` token.

---

## Architecture

| Component | Choice |
|---|---|
| Encoder | `microsoft/deberta-v3-large` (434M params) |
| Pooling | Mean pool over all non-padding tokens |
| Classifier | Linear (hidden → 2) + label smoothing 0.10 |
| Training | Frozen warmup → unfreeze top 6 layers; cosine LR |
| Hyperparameters | lr = 2e-5, batch = 16 × 2 grad-accum, max_len = 256, epochs = 9, patience = 4 |
| Ensemble | 5-seed probability averaging (seeds: 2026, 42, 1337, 7, 2024) |
| Threshold | τ = 0.595 (dev-optimized) |
| can't-tell | Ambiguous training labels remapped to `Yes` |

---

## Results

### Main Results (dev set, τ = 0.595)

| System | F1 | P | R | Acc |
|---|---|---|---|---|
| Majority class | 0.394 | — | — | — |
| TF-IDF + LR | 0.690 | 0.684 | 0.669 | 0.714 |
| TF-IDF + SVM | 0.656 | 0.668 | 0.651 | 0.701 |
| DeBERTa-v3-large, CLS | 0.776 | — | — | — |
| DeBERTa-v3-large, single seed | 0.780 | — | — | — |
| **DeBERTa-v3-large, mean pool, 5-seed** | **0.829** | **0.829** | **0.829** | **0.844** |

**Official test F1: 0.750** (leaderboard rankings not yet public at time of writing).

### Ablation (5-fold CV)

| Config | CV F1 | Δ |
|---|---|---|
| Full system | 0.734 | — |
| − Encoder fine-tuning | 0.662 | **−0.072** |

### Dev → Test Gap

The 7.9-point gap (dev 0.829 → test 0.750) reflects compounding factors on the 77-sample dev set: wide bootstrap uncertainty (CI [0.712, 0.893]); τ and can't-tell remapping both selected on the same small set; ensemble averaging optimized to dev boundary cases. The CV estimate (0.734) was the more honest predictor — use CV-based model selection in low-resource shared tasks.

---

## Project Structure

```
subtask2/
├── README.md
├── requirements.txt
├── .gitignore
│
├── code/
│   ├── train_simple.py          # Simple DeBERTa + mean-pool trainer (original)
│   ├── train_improved.py        # ★ Ensemble trainer used for submission
│   ├── dual.py                  # Dual-head (CLS + mean pool) baseline model
│   ├── run_experiments.py       # Baselines, single-seed ablations, k-fold
│   ├── run_advanced.py          # Layer probing, subreddit analysis, noise robustness
│   ├── run_kfold.py             # 5-fold cross-validation
│   ├── run_cv_ablations.py      # CV ablation study
│   ├── run_new_experiments.py   # Marker masking & pooling comparison
│   ├── run_marker_fusion.py     # Marker-count feature fusion
│   ├── run_supplementary.py     # Calibration, statistical significance
│   ├── run_proper_pooling_cv.py # Pooling strategy head-to-head CV
│   ├── run_analysis.py          # Post-hoc analysis pipeline
│   └── generate_pub_figures.py  # ★ All paper figures (outputs to figures/)
│
├── figures/                     # Figure PNGs for GitHub preview
│   ├── pub_markers.png          # Narrative density / marker analysis
│   ├── pub_probing.png          # Layer-wise probing F1
│   ├── error_analysis.png       # Error analysis breakdown
│   ├── kfold_cv.png             # K-fold CV config comparison
│   └── ...                      # 22 more exploratory figures
│
├── results/
│   ├── baselines.json           # Majority, TF-IDF, DeBERTa-base
│   ├── models.json              # Main model comparison
│   ├── ablations.json           # Component ablation
│   ├── kfold.json               # 5-fold CV
│   ├── improved_cv.json         # CV across ensemble configs
│   ├── improved_ensemble_H.json # ★ Submitted H_conservative ensemble results
│   ├── improved_ensemble.json   # Alt-config ensemble
│   ├── improved_search.json     # Hyperparameter search grid
│   ├── cv_ablations.json        # CV ablation study
│   ├── new_experiments.json     # Marker masking & pooling numbers
│   ├── marker_fusion.json       # Fusion experiment
│   ├── proper_pooling_cv.json   # Pooling CV comparison
│   ├── multi_seed.json          # Per-seed variance
│   ├── ensemble_pr.json         # Ensemble P/R
│   ├── missing_pr.json          # P/R for all table entries
│   ├── advanced.json            # Probing, subreddit, noise
│   ├── supplementary.json       # Calibration, significance
│   ├── full_results.txt         # Human-readable summary
│   └── analysis/
│       ├── bootstrap_ci.json
│       ├── error_analysis.json
│       ├── false_positives.json
│       ├── false_negatives.json
│       ├── full_analysis.json
│       ├── full_dev_predictions.json
│       ├── probability_data.json
│       ├── seed_comparison.json
│       └── threshold_sweep.json
│
├── models/                      # Checkpoints — gitignored (~1.7 GB each)
│   ├── improved_H_seed{2026,42,1337,7,2024}.pt  # ★ Submitted 5-seed ensemble
│   └── best_dual_checkpoint.pt                  # Dual-head baseline
│
├── paper/
│   └── latex/
│       ├── main.tex             # ★ Final paper source
│       ├── main.pdf             # ★ Compiled paper (PDFs in figures/ for LaTeX)
│       ├── references.bib
│       ├── acl.sty
│       └── acl_natbib.bst
│
└── submission/
    └── submission.jsonl         # Official task submission
```

> **Model checkpoints** are gitignored (`.pt` files, ~1.7 GB each). Retrain with the commands below, or track with Git LFS:
> ```bash
> git lfs install && git lfs track "models/*.pt" && git add .gitattributes
> ```

> **Figure PDFs** are gitignored — regenerated by `generate_pub_figures.py`. They are required only if you want to recompile `paper/latex/main.tex` (which uses `\graphicspath{{../../figures/}}`).

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
# Python 3.10+, NVIDIA GPU >= 16 GB VRAM (tested on RTX 4080 Super)
```

### Data

Request access from the [task organizers](https://sites.google.com/view/semeval2026-task10). Place files as:

```
data/
├── train.jsonl
├── dev.jsonl
└── test.jsonl
```

### Train

```bash
# Full submitted ensemble (5 seeds, H_conservative config)
python code/train_improved.py

# Quick single-seed run
python code/train_simple.py --seed 2026
```

### Reproduce experiments

```bash
python code/run_experiments.py        # Baselines + ablations + k-fold
python code/run_advanced.py           # Layer probing, subreddit, noise
python code/run_cv_ablations.py       # CV ablation study
python code/run_new_experiments.py    # Marker masking
python code/run_marker_fusion.py      # Marker-count fusion
python code/run_supplementary.py      # Calibration, significance
python code/run_analysis.py           # Post-hoc analysis
python code/generate_pub_figures.py   # Regenerate all figures
```

---

## Citation

```bibtex
@inproceedings{goyal2026truthgradient,
  title     = {Truth Gradient at {SemEval}-2026 Task 10: Mean Pooling and
               Narrative Density for Conspiracy Belief Detection},
  author    = {Goyal, Ekansh},
  booktitle = {Proceedings of the 20th International Workshop on Semantic
               Evaluation ({SemEval}-2026)},
  year      = {2026},
  note      = {To appear}
}
```
