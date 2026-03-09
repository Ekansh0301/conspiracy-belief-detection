# Conspiracy Belief Detection — SemEval-2026 Task 10, Subtask 2

> **Truth Gradient** · Ekansh Goyal · IIIT Hyderabad

Official system submission for [PsyCoMark: SemEval-2026 Task 10](https://sites.google.com/view/semeval2026-task10), Subtask 2 — detecting whether a Reddit author *believes* a conspiracy theory or merely discusses it.

📄 **Paper:** [Truth Gradient at SemEval-2026 Task 10: Mean Pooling and Narrative Density for Conspiracy Belief Detection](paper/latex/main.pdf)

---

## Results at a Glance

| | |
|---|---|
| **Official Test Macro F1** | **0.750** |
| Dev Macro F1 | 0.829 (P = R = 0.829, Acc = 0.844) |
| Dev 95% Bootstrap CI | [0.712, 0.893] |
| 5-Fold Cross-Validation | 0.734 ± 0.007 |
| Decision Threshold | τ = 0.595 (dev-optimised) |

---

## Overview

### Task

Binary classification over Reddit posts: does the author **believe** the conspiracy theory (`Yes`) or merely **discuss / report** it (`No`)? Evaluation metric is macro F1.

### Core Finding: Narrative Density

Conspiracy believers do not merely use more topically relevant words — they construct *structurally denser* conspiratorial frames. Believers average **6.53 ± 3.53** psycholinguistic markers per post versus **4.35 ± 4.29** for non-believers (Cohen's *d* = 0.56, Mann-Whitney *p* < 10⁻⁸⁰), consistently across all five marker categories:

| Marker Category | Captures |
|---|---|
| **Actor** | Who is behind it |
| **Action** | What they are doing |
| **Effect** | What harm results |
| **Evidence** | What proof is cited |
| **Victim** | Who is harmed |

We call this pattern **narrative density**. Two causal tests validate it:

| Experiment | Finding |
|---|---|
| Mask all annotated marker spans | F1: .723 → .670 (−5.3 pp) |
| Append 6 normalised marker-count features | F1: .670 → .679 (+0.9 pp) |

Because the belief signal is *distributed* across the full post rather than concentrated at one position, **sequence-level mean pooling** outperforms single-token `[CLS]` representations.

---

## Model

| Component | Details |
|---|---|
| **Encoder** | `microsoft/deberta-v3-large` (434 M params, 24 layers) |
| **Pooling** | Mean pool over all non-padding token representations |
| **Classifier** | Linear projection (hidden → 2 classes) |
| **Label smoothing** | 0.10 |
| **Training schedule** | Frozen warmup → unfreeze top 6 layers → cosine LR decay |
| **Optimiser** | AdamW, lr = 2 × 10⁻⁵, batch = 16 × 2 grad-accum |
| **Sequence length** | 256 tokens |
| **Early stopping** | Patience = 4 epochs (max 9) |
| **Ensemble** | Probability averaging over 5 seeds: {2026, 42, 1337, 7, 2024} |
| **can't-tell handling** | Ambiguous training labels remapped to `Yes` |

---

## Experimental Results

### Comparison with Baselines

| System | F1 | P | R | Acc |
|---|---|---|---|---|
| Majority class | 0.394 | — | — | — |
| TF-IDF + Logistic Regression | 0.690 | 0.684 | 0.669 | 0.714 |
| TF-IDF + SVM | 0.656 | 0.668 | 0.651 | 0.701 |
| DeBERTa-v3-large, CLS token | 0.776 | — | — | — |
| DeBERTa-v3-large, single seed | 0.780 | — | — | — |
| **DeBERTa-v3-large, mean pool, 5-seed (ours)** | **0.829** | **0.829** | **0.829** | **0.844** |

**Official test F1: 0.750.** Leaderboard rankings had not been published at the time of writing.

### Ablation (5-Fold CV)

| Configuration | CV F1 | Δ |
|---|---|---|
| Full system | 0.734 | — |
| − Encoder fine-tuning (all layers frozen) | 0.662 | **−0.072** |

### Note on the Dev → Test Gap

The 7.9-point gap (dev 0.829 → test 0.750) is explained by three compounding factors:
1. The 77-sample dev set produces wide uncertainty (bootstrap 95% CI: [0.712, 0.893]).
2. Both the decision threshold (τ = 0.595) and the can't-tell remapping strategy were selected on the same 77 samples.
3. The ensemble was implicitly tuned to dev boundary cases.

The 5-fold CV estimate (0.734) was the more reliable predictor of test performance and was not used for selection — we recommend CV-based model selection as the standard for low-resource shared tasks.

---

## Repository Structure

```
.
├── code/
│   ├── train_improved.py        ← submitted ensemble trainer
│   ├── train_simple.py          ← original single-seed trainer
│   ├── dual.py                  ← CLS + mean-pool dual-head baseline
│   ├── run_experiments.py       ← baselines, single-seed ablations, k-fold
│   ├── run_advanced.py          ← layer probing, subreddit analysis, noise
│   ├── run_kfold.py             ← 5-fold cross-validation
│   ├── run_cv_ablations.py      ← CV ablation study
│   ├── run_new_experiments.py   ← marker masking & pooling comparison
│   ├── run_marker_fusion.py     ← marker-count feature fusion
│   ├── run_supplementary.py     ← calibration, significance tests
│   ├── run_proper_pooling_cv.py ← pooling strategy head-to-head CV
│   ├── run_analysis.py          ← post-hoc analysis pipeline
│   └── generate_pub_figures.py  ← generates all paper figures
│
├── figures/                     ← PNG previews of all figures
│   ├── pub_markers.png          ← narrative density analysis (paper Fig. 1)
│   ├── pub_probing.png          ← layer-wise probing F1 (paper Fig. 2)
│   ├── error_analysis.png       ← error breakdown (paper Fig. 3)
│   └── ...                      ← 23 additional exploratory figures
│
├── results/
│   ├── improved_ensemble_H.json ← submitted ensemble results
│   ├── new_experiments.json     ← marker masking numbers
│   ├── marker_fusion.json       ← fusion experiment results
│   ├── cv_ablations.json        ← CV ablation study
│   ├── full_results.txt         ← human-readable summary
│   ├── analysis/                ← per-sample predictions, FP/FN examples,
│   │   └── ...                     bootstrap CIs, threshold sweep
│   └── ...                      ← all other experiment JSONs
│
├── models/                      ← gitignored — ~1.7 GB per checkpoint
│   └── improved_H_seed{2026,42,1337,7,2024}.pt
│
├── paper/
│   └── latex/
│       ├── main.tex             ← paper source
│       ├── main.pdf             ← compiled paper
│       └── references.bib
│
├── submission/
│   └── submission.jsonl         ← official task submission
│
├── requirements.txt
└── .gitignore
```

> **Model weights** (`.pt` files, ~1.7 GB each) are excluded from this repository. Retrain using the instructions below, or host with Git LFS:
> ```bash
> git lfs install && git lfs track "models/*.pt" && git add .gitattributes
> ```

> **Figure PDFs** are excluded (`figures/*.pdf`). They are regenerated automatically by `generate_pub_figures.py` and are only needed to recompile the LaTeX paper (which resolves them via `\graphicspath{{../../figures/}}`).

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and an NVIDIA GPU with ≥ 16 GB VRAM (tested on RTX 4080 Super).

### 2. Obtain the data

Request access from the [task organisers](https://sites.google.com/view/semeval2026-task10) and place the files as:

```
data/
├── train.jsonl
├── dev.jsonl
└── test.jsonl
```

### 3. Train

```bash
# Reproduce the submitted 5-seed ensemble (H_conservative config)
python code/train_improved.py

# Quick single-seed run for development
python code/train_simple.py --seed 2026
```

### 4. Run experiments

```bash
python code/run_experiments.py        # baselines + ablations + k-fold
python code/run_advanced.py           # layer probing, subreddit, noise robustness
python code/run_cv_ablations.py       # CV ablation study
python code/run_new_experiments.py    # marker masking
python code/run_marker_fusion.py      # marker-count fusion
python code/run_supplementary.py      # calibration + significance tests
python code/run_analysis.py           # post-hoc analysis
python code/generate_pub_figures.py   # regenerate all paper figures
```

---

## Citation

If you use this code or findings, please cite:

```bibtex
@inproceedings{goyal2026truthgradient,
  title     = {Truth Gradient at {SemEval}-2026 Task 10:
               Mean Pooling and Narrative Density for Conspiracy Belief Detection},
  author    = {Goyal, Ekansh},
  booktitle = {Proceedings of the 20th International Workshop on
               Semantic Evaluation ({SemEval}-2026)},
  year      = {2026},
  note      = {To appear}
}
```
