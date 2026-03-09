#!/usr/bin/env python3
"""
Supplementary Analyses for SemEval 2026 Task 10 Paper
=====================================================
Statistical significance, calibration, confidence analysis,
discriminative features, per-subreddit breakdown.

All from existing predictions — no GPU retraining needed.
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy import stats
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, brier_score_loss
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================================
# SETUP
# ============================================================================

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = BASE_DIR.parent
SEED = 2026


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
    """Load all data including Can't tell"""
    return load_data(filepath, filter_ambiguous=False)


# ============================================================================
# 1. STATISTICAL SIGNIFICANCE (McNemar's test)
# ============================================================================

def mcnemar_test(preds_a, preds_b, labels):
    """
    McNemar's test: are the two classifiers significantly different?
    Returns chi2 statistic and p-value.
    """
    # Build contingency table
    # b = A correct, B wrong
    # c = A wrong, B correct
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right

    # McNemar's with continuity correction
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p), int(b), int(c)


def run_significance_tests():
    print("=" * 70)
    print("1. STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)

    # Load dev data
    dev_data = load_data(DATA_DIR / "dev_public.jsonl")
    train_data = load_data(DATA_DIR / "train_rehydrated.jsonl")
    dev_texts = [d['text'] for d in dev_data]
    dev_labels = np.array([1 if d['conspiracy'].lower() == 'yes' else 0 for d in dev_data])
    train_texts = [d['text'] for d in train_data]
    train_labels = [1 if d['conspiracy'].lower() == 'yes' else 0 for d in train_data]

    # Our model predictions
    with open(RESULTS_DIR / 'models.json') as f:
        models = json.load(f)
    our_probs = np.array(models['simple']['probs'])
    our_preds = (our_probs >= models['simple']['threshold']).astype(int)

    # Baseline predictions
    # Majority
    majority = Counter(train_labels).most_common(1)[0][0]
    maj_preds = np.array([majority] * len(dev_labels))

    # TF-IDF + LR
    tfidf_lr = TfidfVectorizer(max_features=50000, ngram_range=(1, 3),
                                sublinear_tf=True, min_df=2)
    X_train = tfidf_lr.fit_transform(train_texts)
    X_dev = tfidf_lr.transform(dev_texts)
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=SEED)
    lr.fit(X_train, train_labels)
    lr_preds = lr.predict(X_dev)

    # CLS ablation predictions (reconstruct from ablation)
    # We'll use our model probs at CLS threshold
    with open(RESULTS_DIR / 'ablations.json') as f:
        ablations = json.load(f)

    results = {}

    # Our model vs Majority
    chi2, p, b, c = mcnemar_test(our_preds, maj_preds, dev_labels)
    results['ours_vs_majority'] = {'chi2': chi2, 'p': p, 'b': b, 'c': c,
                                    'significant': p < 0.05}
    print(f"\n  Ours vs Majority: χ²={chi2:.3f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

    # Our model vs TF-IDF+LR
    chi2, p, b, c = mcnemar_test(our_preds, lr_preds, dev_labels)
    results['ours_vs_tfidf_lr'] = {'chi2': chi2, 'p': p, 'b': b, 'c': c,
                                    'significant': p < 0.05}
    print(f"  Ours vs TF-IDF+LR: χ²={chi2:.3f}, p={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

    # Permutation test (more robust for small samples)
    print("\n  Permutation tests (10000 iterations):")
    our_f1 = f1_score_macro(our_preds, dev_labels)
    lr_f1 = f1_score_macro(lr_preds, dev_labels)
    maj_f1 = f1_score_macro(maj_preds, dev_labels)

    # Permutation: ours vs TF-IDF+LR
    observed_diff = our_f1 - lr_f1
    n_perms = 10000
    count = 0
    combined = np.column_stack([our_preds, lr_preds])
    rng = np.random.RandomState(SEED)
    for _ in range(n_perms):
        # Randomly swap predictions between systems
        swap = rng.randint(0, 2, size=len(dev_labels))
        perm_a = np.where(swap == 0, combined[:, 0], combined[:, 1])
        perm_b = np.where(swap == 0, combined[:, 1], combined[:, 0])
        perm_diff = f1_score_macro(perm_a, dev_labels) - f1_score_macro(perm_b, dev_labels)
        if perm_diff >= observed_diff:
            count += 1
    perm_p = count / n_perms
    results['permutation_ours_vs_tfidf'] = {
        'observed_diff': round(float(observed_diff), 4),
        'p': round(float(perm_p), 4),
        'significant': perm_p < 0.05
    }
    print(f"    Ours vs TF-IDF+LR: ΔF1={observed_diff:.4f}, p={perm_p:.4f} "
          f"{'***' if perm_p < 0.001 else '**' if perm_p < 0.01 else '*' if perm_p < 0.05 else 'n.s.'}")

    return results


def f1_score_macro(preds, labels):
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return f1


# ============================================================================
# 2. CALIBRATION ANALYSIS
# ============================================================================

def run_calibration_analysis():
    print("\n" + "=" * 70)
    print("2. CALIBRATION ANALYSIS")
    print("=" * 70)

    with open(RESULTS_DIR / 'models.json') as f:
        models = json.load(f)

    probs = np.array(models['simple']['probs'])
    labels = np.array(models['simple']['labels'])

    # Brier score
    brier = brier_score_loss(labels, probs)
    print(f"\n  Brier Score: {brier:.4f} (lower is better, 0 = perfect)")

    # Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        bin_accs.append(float(bin_acc))
        bin_confs.append(float(bin_conf))
        bin_counts.append(int(mask.sum()))

    # ECE
    total = len(probs)
    ece = sum(abs(bin_accs[i] - bin_confs[i]) * bin_counts[i] / total
              for i in range(n_bins))
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")

    # Maximum Calibration Error
    mce = max(abs(bin_accs[i] - bin_confs[i]) for i in range(n_bins) if bin_counts[i] > 0)
    print(f"  MCE (Maximum Calibration Error): {mce:.4f}")

    print(f"\n  Reliability diagram data:")
    print(f"  {'Bin':>10} {'Count':>6} {'Conf':>8} {'Acc':>8} {'|Diff|':>8}")
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        diff = abs(bin_accs[i] - bin_confs[i])
        print(f"  [{lo:.1f}-{hi:.1f}] {bin_counts[i]:>6} {bin_confs[i]:>8.3f} "
              f"{bin_accs[i]:>8.3f} {diff:>8.3f}")

    results = {
        'brier_score': round(float(brier), 4),
        'ece': round(float(ece), 4),
        'mce': round(float(mce), 4),
        'reliability_diagram': {
            'bin_accs': bin_accs,
            'bin_confs': bin_confs,
            'bin_counts': bin_counts,
            'bin_edges': bin_boundaries.tolist(),
        }
    }
    return results


# ============================================================================
# 3. CONFIDENCE-STRATIFIED ACCURACY
# ============================================================================

def run_confidence_analysis():
    print("\n" + "=" * 70)
    print("3. CONFIDENCE-STRATIFIED ANALYSIS")
    print("=" * 70)

    with open(RESULTS_DIR / 'models.json') as f:
        models = json.load(f)

    probs = np.array(models['simple']['probs'])
    labels = np.array(models['simple']['labels'])
    threshold = models['simple']['threshold']
    preds = (probs >= threshold).astype(int)

    # Confidence = distance from threshold
    confidence = np.abs(probs - threshold)

    # Stratify by confidence quartiles
    quartiles = np.percentile(confidence, [25, 50, 75])
    strata = ['Low (Q1)', 'Med-Low (Q2)', 'Med-High (Q3)', 'High (Q4)']
    boundaries = [0, quartiles[0], quartiles[1], quartiles[2], confidence.max() + 0.01]

    print(f"\n  {'Stratum':<16} {'N':>4} {'Acc':>8} {'F1':>8} {'Conf Range':>15}")
    results = {}
    for i in range(4):
        mask = (confidence >= boundaries[i]) & (confidence < boundaries[i + 1])
        n = mask.sum()
        if n > 0:
            acc = accuracy_score(labels[mask], preds[mask])
            _, _, f1, _ = precision_recall_fscore_support(labels[mask], preds[mask],
                                                          average='macro', zero_division=0)
        else:
            acc, f1 = 0, 0
        rng = f"[{boundaries[i]:.3f}-{boundaries[i+1]:.3f}]"
        print(f"  {strata[i]:<16} {n:>4} {acc:>8.3f} {f1:>8.3f} {rng:>15}")
        results[strata[i]] = {'n': int(n), 'accuracy': round(float(acc), 4),
                               'f1': round(float(f1), 4)}

    # Spearman correlation between confidence and correctness
    correct = (preds == labels).astype(int)
    rho, p = stats.spearmanr(confidence, correct)
    print(f"\n  Confidence-correctness correlation: ρ={rho:.3f}, p={p:.4f}")
    results['confidence_correctness_corr'] = {'rho': round(float(rho), 4),
                                               'p': round(float(p), 4)}

    return results


# ============================================================================
# 4. DISCRIMINATIVE FEATURES (TF-IDF)
# ============================================================================

def run_feature_analysis():
    print("\n" + "=" * 70)
    print("4. TOP DISCRIMINATIVE FEATURES")
    print("=" * 70)

    train_data = load_data(DATA_DIR / "train_rehydrated.jsonl")
    texts = [d['text'] for d in train_data]
    labels = [1 if d['conspiracy'].lower() == 'yes' else 0 for d in train_data]

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                             sublinear_tf=True, min_df=3)
    X = tfidf.fit_transform(texts)
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=SEED)
    lr.fit(X, labels)

    feature_names = tfidf.get_feature_names_out()
    coefs = lr.coef_[0]

    # Top conspiracy indicators
    top_yes_idx = np.argsort(coefs)[-20:][::-1]
    top_no_idx = np.argsort(coefs)[:20]

    results = {'conspiracy_indicators': [], 'non_conspiracy_indicators': []}

    print("\n  Top 20 features indicating CONSPIRACY:")
    for i, idx in enumerate(top_yes_idx):
        feat = feature_names[idx]
        w = coefs[idx]
        results['conspiracy_indicators'].append({'feature': feat, 'weight': round(float(w), 4)})
        print(f"    {i+1:2d}. {feat:<25} w={w:.4f}")

    print("\n  Top 20 features indicating NON-CONSPIRACY:")
    for i, idx in enumerate(top_no_idx):
        feat = feature_names[idx]
        w = coefs[idx]
        results['non_conspiracy_indicators'].append({'feature': feat, 'weight': round(float(w), 4)})
        print(f"    {i+1:2d}. {feat:<25} w={w:.4f}")

    return results


# ============================================================================
# 5. PER-SUBREDDIT PERFORMANCE
# ============================================================================

def run_subreddit_analysis():
    print("\n" + "=" * 70)
    print("5. PER-SUBREDDIT MODEL PERFORMANCE")
    print("=" * 70)

    dev_data = load_data_all(DATA_DIR / "dev_public.jsonl")

    with open(RESULTS_DIR / 'models.json') as f:
        models = json.load(f)
    probs = np.array(models['simple']['probs'])
    labels = np.array(models['simple']['labels'])
    threshold = models['simple']['threshold']
    preds = (probs >= threshold).astype(int)

    # Map dev data (filtered) to predictions
    # The model was evaluated on filtered data (no Can't tell)
    filtered_dev = [d for d in dev_data if d.get('conspiracy', '').lower() != "can't tell"]

    subreddit_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'labels': [], 'preds': []})
    for i, item in enumerate(filtered_dev):
        sub = item.get('subreddit', 'unknown')
        is_correct = int(preds[i] == labels[i])
        subreddit_results[sub]['correct'] += is_correct
        subreddit_results[sub]['total'] += 1
        subreddit_results[sub]['labels'].append(int(labels[i]))
        subreddit_results[sub]['preds'].append(int(preds[i]))

    # Sort by sample count
    sorted_subs = sorted(subreddit_results.items(), key=lambda x: x[1]['total'], reverse=True)

    print(f"\n  {'Subreddit':<25} {'N':>4} {'Acc':>8} {'Errors':>8}")
    print("  " + "-" * 50)
    results = {}
    for sub, data in sorted_subs[:20]:
        acc = data['correct'] / data['total']
        errors = data['total'] - data['correct']
        print(f"  {sub:<25} {data['total']:>4} {acc:>8.3f} {errors:>8}")
        results[sub] = {'n': data['total'], 'accuracy': round(float(acc), 4),
                         'errors': errors}

    # Conspiracy-themed vs other subreddits
    conspiracy_subs = {'conspiracy', 'conspiracy_commons', 'conspiracytheories',
                       'Bibleconspiracy', 'TopMindsOfReddit'}
    consp_correct, consp_total = 0, 0
    other_correct, other_total = 0, 0
    for sub, data in subreddit_results.items():
        if sub in conspiracy_subs:
            consp_correct += data['correct']
            consp_total += data['total']
        else:
            other_correct += data['correct']
            other_total += data['total']

    if consp_total > 0 and other_total > 0:
        consp_acc = consp_correct / consp_total
        other_acc = other_correct / other_total
        print(f"\n  Conspiracy-themed subs: {consp_acc:.3f} ({consp_total} samples)")
        print(f"  Other subreddits:      {other_acc:.3f} ({other_total} samples)")
        results['conspiracy_themed_acc'] = round(float(consp_acc), 4)
        results['other_acc'] = round(float(other_acc), 4)

    return results


# ============================================================================
# 6. CAN'T-TELL CLASS DEEP ANALYSIS
# ============================================================================

def run_cant_tell_analysis():
    print("\n" + "=" * 70)
    print("6. CAN'T-TELL CLASS ANALYSIS")
    print("=" * 70)

    dev_data = load_data_all(DATA_DIR / "dev_public.jsonl")
    cant_tell = [d for d in dev_data if d.get('conspiracy', '').lower() == "can't tell"]

    with open(BASE_DIR / "model" / "analysis" / "full_analysis.json") as f:
        full = json.load(f)

    ct = full['model_results']['cant_tell']
    print(f"\n  Can't-tell samples: {ct['n']}")
    print(f"  Predicted Yes: {ct['pred_yes']} ({ct['pred_yes']/ct['n']*100:.1f}%)")
    print(f"  Predicted No: {ct['pred_no']} ({ct['pred_no']/ct['n']*100:.1f}%)")
    print(f"  Mean probability: {ct['mean_prob']:.3f} (ideal: ~0.5)")
    print(f"  Std probability: {ct['std_prob']:.3f}")
    print(f"  In uncertain zone: {ct['in_uncertain_zone']}/{ct['n']} ({ct['in_uncertain_zone']/ct['n']*100:.1f}%)")
    print(f"\n  Interpretation: {ct['in_uncertain_zone']/ct['n']*100:.0f}% of Can't-tell posts")
    print(f"  fall in the model's uncertainty zone, suggesting the model")
    print(f"  is appropriately uncertain about genuinely ambiguous cases.")

    return ct


# ============================================================================
# GENERATE SUPPLEMENTARY FIGURES
# ============================================================================

def generate_sup_figures(calibration, confidence, features):
    print("\n" + "=" * 70)
    print("GENERATING SUPPLEMENTARY FIGURES")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'serif',
    })

    # Figure S1: Reliability Diagram
    print("  Fig S1: Reliability diagram")
    rd = calibration['reliability_diagram']
    bin_accs = np.array(rd['bin_accs'])
    bin_confs = np.array(rd['bin_confs'])
    bin_counts = np.array(rd['bin_counts'])
    bin_edges = np.array(rd['bin_edges'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 5.5),
                                     gridspec_kw={'height_ratios': [3, 1]})

    # Top: reliability curve
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = bin_counts > 0
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax1.bar(bin_centers[mask], bin_accs[np.where(mask)], width=0.08, alpha=0.6,
            color='#2196F3', edgecolor='white', label='Model')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'Reliability Diagram (ECE={calibration["ece"]:.3f})')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Bottom: histogram
    ax2.bar(bin_centers, bin_counts, width=0.08, color='#9E9E9E', edgecolor='white')
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'reliability_diagram.pdf')
    fig.savefig(FIGURES_DIR / 'reliability_diagram.png')
    plt.close(fig)

    # Figure S2: Top Discriminative Features
    print("  Fig S2: Discriminative features")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Conspiracy features
    consp_feats = features['conspiracy_indicators'][:15]
    names = [f['feature'] for f in consp_feats]
    weights = [f['weight'] for f in consp_feats]
    y_pos = range(len(names))
    ax1.barh(y_pos, weights, color='#F44336', edgecolor='white', height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('LR Coefficient')
    ax1.set_title('Top Conspiracy Indicators')
    ax1.invert_yaxis()
    ax1.grid(True, axis='x', alpha=0.3)

    # Non-conspiracy features
    nonc_feats = features['non_conspiracy_indicators'][:15]
    names = [f['feature'] for f in nonc_feats]
    weights = [abs(f['weight']) for f in nonc_feats]
    y_pos = range(len(names))
    ax2.barh(y_pos, weights, color='#2196F3', edgecolor='white', height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('|LR Coefficient|')
    ax2.set_title('Top Non-Conspiracy Indicators')
    ax2.invert_yaxis()
    ax2.grid(True, axis='x', alpha=0.3)

    fig.suptitle('Most Discriminative Features (TF-IDF + LR)', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / 'discriminative_features.pdf')
    fig.savefig(FIGURES_DIR / 'discriminative_features.png')
    plt.close(fig)

    # Figure S3: Confidence vs Accuracy
    print("  Fig S3: Confidence vs accuracy")
    with open(RESULTS_DIR / 'models.json') as f:
        models = json.load(f)
    probs = np.array(models['simple']['probs'])
    labels = np.array(models['simple']['labels'])
    threshold = models['simple']['threshold']
    preds = (probs >= threshold).astype(int)
    confidence_vals = np.abs(probs - threshold)
    correct = (preds == labels).astype(int)

    # Bin by confidence
    n_bins = 8
    sorted_idx = np.argsort(confidence_vals)
    bin_size = len(sorted_idx) // n_bins
    bin_accs_conf = []
    bin_conf_means = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(sorted_idx)
        idx = sorted_idx[start:end]
        bin_accs_conf.append(correct[idx].mean())
        bin_conf_means.append(confidence_vals[idx].mean())

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(bin_conf_means, bin_accs_conf, 'bo-', markersize=8, linewidth=2)
    ax.fill_between(bin_conf_means, bin_accs_conf, alpha=0.1, color='blue')
    ax.set_xlabel('Model Confidence (|P - threshold|)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Confidence-Stratified Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.savefig(FIGURES_DIR / 'confidence_accuracy.pdf')
    fig.savefig(FIGURES_DIR / 'confidence_accuracy.png')
    plt.close(fig)

    print(f"  All supplementary figures saved to {FIGURES_DIR}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("SemEval 2026 — Supplementary Analyses")
    print("=" * 70)

    all_results = {}

    # 1. Statistical significance
    all_results['significance'] = run_significance_tests()

    # 2. Calibration
    all_results['calibration'] = run_calibration_analysis()

    # 3. Confidence
    all_results['confidence'] = run_confidence_analysis()

    # 4. Features
    all_results['features'] = run_feature_analysis()

    # 5. Per-subreddit
    all_results['subreddit_perf'] = run_subreddit_analysis()

    # 6. Can't-tell
    all_results['cant_tell'] = run_cant_tell_analysis()

    # Generate figures
    generate_sup_figures(all_results['calibration'], all_results['confidence'],
                         all_results['features'])

    # Save all
    with open(RESULTS_DIR / 'supplementary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ALL SUPPLEMENTARY ANALYSES COMPLETE")
    print(f"Results: {RESULTS_DIR / 'supplementary.json'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
