<div align="center">

# Conspiracy Belief Detection
### SemEval-2026 Task 10, Subtask 2 · PsyCoMark

**Ekansh Goyal** · IIIT Hyderabad · Team: Truth Gradient

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/latex/main.pdf)
[![Task](https://img.shields.io/badge/SemEval--2026-Task%2010-blue)](https://sites.google.com/view/semeval2026-task10)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](requirements.txt)
[![Model](https://img.shields.io/badge/Model-DeBERTa--v3--large-orange)](https://huggingface.co/microsoft/deberta-v3-large)

| Metric | Score |
|:---|:---:|
| **Official Test Macro F1** | **0.750** |
| Dev Macro F1 (τ = 0.595) | 0.829 |
| Dev Precision / Recall / Accuracy | 0.829 / 0.829 / 0.844 |
| Dev Bootstrap 95% CI | [0.712, 0.893] |
| 5-Fold Cross-Validation F1 | 0.734 ± 0.007 |

</div>

---

## About

This repository contains the full system for [PsyCoMark (SemEval-2026 Task 10)](https://sites.google.com/view/semeval2026-task10), Subtask 2: given a Reddit post, classify whether the author **believes** a conspiracy theory (`Yes`) or merely **discusses or reports** it (`No`). Evaluation is by macro F1.

The task is hard because the surface-level vocabulary of believers and non-believers is nearly identical — both groups write about the same topics. What differs is *how* they write.

---

## Key Idea: Narrative Density

Our central finding is that conspiracy believers do not just use different words; they construct *structurally denser* conspiratorial narratives. Believers pack significantly more psycholinguistic markers into each post than non-believers:

| | Believers | Non-believers |
|:---|:---:|:---:|
| Markers per post (mean ± std) | **6.53 ± 3.53** | 4.35 ± 4.29 |
| Cohen's *d* | **0.56** | — |
| Mann-Whitney *p* | **< 10⁻⁸⁰** | — |
| Full narrative rate (all 5 types) | **41.3%** | 22.2% |

The five marker categories — **Actor**, **Action**, **Effect**, **Evidence**, **Victim** — together form the complete schema of a conspiracy narrative. Believers consistently populate *all five* slots; non-believers leave gaps.

We call this pattern **narrative density**, and it has a direct architectural implication: because the belief signal is spread across the entire post rather than concentrated at any single token, **sequence-level mean pooling** captures it better than a `[CLS]` representation.

Two experiments confirm the markers carry genuine signal:

| Test | Finding |
|:---|:---|
| **Marker masking** — replace all annotated spans with whitespace | F1 drops from 0.723 → 0.670 (−5.3 pp) |
| **Marker-count fusion** — append 6 normalised count features to encoder output | F1 rises from 0.670 → 0.679 (+0.9 pp) |

---

## System Architecture

Fine-tuned `microsoft/deberta-v3-large` with a lightweight linear classifier on top of mean-pooled token representations, trained as a 5-seed ensemble.

| Component | Configuration |
|:---|:---|
| Encoder | `microsoft/deberta-v3-large` — 434 M parameters, 24 layers |
| Pooling | Mean pool over all non-padding token representations |
| Classifier | Single linear layer (hidden dim → 2) |
| Label smoothing | 0.10 |
| Training | Two-phase: frozen warmup, then unfreeze top 6 encoder layers |
| Optimiser | AdamW · lr = 2×10⁻⁵ · batch = 16 × 2 gradient accumulation |
| Max sequence length | 256 tokens |
| Early stopping | Patience = 4 (max 9 epochs) |
| Ensemble | Probability averaging across 5 seeds: {42, 7, 2024, 1337, 2026} |
| can't-tell labels | Remapped to `Yes` during training |
| Decision threshold | τ = 0.595 (optimised on dev set) |

---

## Results

### Main Comparison

| System | F1 | P | R | Acc |
|:---|:---:|:---:|:---:|:---:|
| Majority class | 0.394 | — | — | — |
| TF-IDF + Logistic Regression | 0.690 | 0.684 | 0.669 | 0.714 |
| TF-IDF + SVM | 0.656 | 0.668 | 0.651 | 0.701 |
| DeBERTa-v3-large, CLS token | 0.776 | — | — | — |
| DeBERTa-v3-large, mean pool (single seed) | 0.780 | — | — | — |
| **Ours — mean pool, 5-seed ensemble** | **0.829** | **0.829** | **0.829** | **0.844** |

> **Official test F1: 0.750** — submitted through the official evaluation portal. Task leaderboard rankings were not yet published at time of writing.

### Ablation (5-Fold CV)

| Configuration | CV F1 | Δ |
|:---|:---:|:---:|
| Full system | 0.734 | — |
| − Encoder fine-tuning (all layers frozen) | 0.662 | **−0.072** |

Encoder fine-tuning is by far the dominant factor. All other components (label smoothing, can't-tell remapping, pooling choice) provide incremental gains.

### On the Dev → Test Gap

The 7.9-point drop from dev (0.829) to test (0.750) deserves context. Three factors compound on the tiny 77-sample dev set:

1. **Sampling noise** — the bootstrap 95% CI is [0.712, 0.893]; the dev point estimate is inherently unreliable.
2. **Selection overfit** — both the threshold τ = 0.595 and the can't-tell remapping strategy were chosen using the same 77 samples.
3. **Ensemble fit** — probability averaging was implicitly calibrated to dev boundary cases.

The 5-fold CV estimate (0.734 ± 0.007) was not used for selection and proved far more predictive of test performance. **We recommend CV-based model selection as standard practice in low-resource shared tasks.**

---

## Repository Structure

```
.
├── code/
│   ├── train_improved.py        # submitted ensemble trainer ★
│   ├── train_simple.py          # original single-seed mean-pool trainer
│   ├── dual.py                  # dual-head (CLS + mean pool) baseline
│   ├── generate_pub_figures.py  # generates all paper figures ★
│   ├── run_experiments.py       # baselines, single-seed ablations, k-fold
│   ├── run_advanced.py          # layer probing, subreddit analysis, noise
│   ├── run_kfold.py             # 5-fold cross-validation
│   ├── run_cv_ablations.py      # CV ablation study
│   ├── run_new_experiments.py   # marker masking & pooling comparison
│   ├── run_marker_fusion.py     # marker-count feature fusion
│   ├── run_proper_pooling_cv.py # pooling strategy head-to-head CV
│   ├── run_supplementary.py     # calibration, significance tests
│   └── run_analysis.py          # post-hoc analysis pipeline
│
├── figures/                     # PNG previews (PDFs are gitignored)
│   ├── pub_markers.png          # narrative density — paper Fig. 1
│   ├── pub_probing.png          # layer-wise probing — paper Fig. 2
│   ├── error_analysis.png       # error breakdown — paper Fig. 3
│   └── ...                      # 23 additional exploratory figures
│
├── results/
│   ├── improved_ensemble_H.json # submitted system results ★
│   ├── new_experiments.json     # marker masking numbers
│   ├── marker_fusion.json       # fusion experiment
│   ├── cv_ablations.json        # CV ablation
│   ├── full_results.txt         # human-readable summary of all experiments
│   ├── analysis/                # per-sample predictions, bootstrap CIs,
│   │   └── ...                  # FP/FN breakdowns, threshold sweep
│   └── ...                      # all other experiment JSONs
│
├── models/                      # gitignored — ~1.7 GB per .pt file
│   └── improved_H_seed{42,7,2024,1337,2026}.pt
│
├── paper/latex/
│   ├── main.tex                 # paper source ★
│   ├── main.pdf                 # compiled paper ★
│   └── references.bib
│
├── submission/
│   └── submission.jsonl         # official task submission file
│
├── requirements.txt
└── .gitignore
```

**Notes:**
- Model checkpoints (`models/*.pt`, ~1.7 GB each) are gitignored. Re-train locally or use [Git LFS](https://git-lfs.com): `git lfs track "models/*.pt"`.
- Figure PDFs (`figures/*.pdf`) are gitignored. Regenerate with `python code/generate_pub_figures.py`. They are only needed to recompile the LaTeX source, which resolves them via `\graphicspath{{../../figures/}}`.

---

## Reproducing the Results

### Prerequisites

```bash
git clone https://github.com/Ekansh0301/conspiracy-belief-detection.git
cd conspiracy-belief-detection
pip install -r requirements.txt
```

Python 3.10+ and an NVIDIA GPU with ≥ 16 GB VRAM are required (developed on an RTX 4080 Super).

### Data

Request access to the PsyCoMark dataset from the [task organisers](https://sites.google.com/view/semeval2026-task10) and place the files at:

```
data/
├── train.jsonl
├── dev.jsonl
└── test.jsonl
```

### Training

```bash
# Full submitted system — 5-seed ensemble with H_conservative configuration
python code/train_improved.py

# Lightweight single-seed run (for development / verification)
python code/train_simple.py --seed 2026
```

### Running Experiments

```bash
python code/run_experiments.py        # baselines, ablations, k-fold CV
python code/run_advanced.py           # layer probing, subreddit generalisation, noise robustness
python code/run_cv_ablations.py       # CV ablation study
python code/run_new_experiments.py    # marker masking experiment
python code/run_marker_fusion.py      # marker-count feature fusion
python code/run_supplementary.py      # calibration and significance tests
python code/run_analysis.py           # post-hoc error and probability analysis
python code/generate_pub_figures.py   # regenerate all paper figures
```

---

## Citation

```bibtex
@inproceedings{goyal2026truthgradient,
  title     = {Truth Gradient at {SemEval}-2026 Task~10:
               Mean Pooling and Narrative Density for Conspiracy Belief Detection},
  author    = {Goyal, Ekansh},
  booktitle = {Proceedings of the 20th International Workshop on
               Semantic Evaluation ({SemEval}-2026)},
  year      = {2026},
  note      = {To appear}
}
```
