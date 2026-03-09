#!/usr/bin/env python3
"""
Publication-Quality Figure Generator for SemEval 2026 Task 10 — Subtask 2
==========================================================================
Generates all figures for the paper with consistent, publication-ready styling:

  1. CV Ablation bar chart with error bars (KEY figure)
  2. Layer probing analysis
  3. System comparison overview
  4. Confusion matrix heatmap
  5. Probability distribution
  6. Threshold sensitivity
  7. Training curves (two-phase dynamics)
  8. K-fold CV box plot
  9. Multi-seed stability
  10. Noise robustness
  11. Marker density violin plot
"""

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import seaborn as sns

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).parent.parent  # subtask2/
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- Publication style ---
COLORS = {
    'primary': '#2563EB',      # Blue
    'secondary': '#DC2626',    # Red
    'success': '#16A34A',      # Green
    'warning': '#D97706',      # Amber
    'neutral': '#6B7280',      # Gray
    'light': '#93C5FD',        # Light blue
    'accent': '#7C3AED',       # Purple
    'dark': '#1F2937',         # Dark gray
}

SYSTEM_COLORS = {
    'ours': '#16A34A',
    'ablation_good': '#2563EB',
    'ablation_bad': '#DC2626',
    'baseline': '#9CA3AF',
    'model': '#60A5FA',
}

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def save_fig(fig, name):
    """Save figure in both PDF and PNG formats."""
    fig.savefig(FIGURES_DIR / f'{name}.pdf')
    fig.savefig(FIGURES_DIR / f'{name}.png')
    plt.close(fig)
    print(f"  Saved: {name}.pdf/.png")


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ============================================================================
# LOAD ALL DATA
# ============================================================================

def load_all_results():
    """Load all available result files."""
    data = {}
    data['baselines'] = load_json(RESULTS_DIR / 'baselines.json')
    data['models'] = load_json(RESULTS_DIR / 'models.json')
    data['ablations'] = load_json(RESULTS_DIR / 'ablations.json')
    data['kfold'] = load_json(RESULTS_DIR / 'kfold.json')
    data['advanced'] = load_json(RESULTS_DIR / 'advanced.json')
    data['supplementary'] = load_json(RESULTS_DIR / 'supplementary.json')
    data['cv_ablations'] = load_json(RESULTS_DIR / 'cv_ablations.json')
    data['multi_seed'] = load_json(RESULTS_DIR / 'multi_seed.json')
    data['bootstrap'] = load_json(RESULTS_DIR / 'analysis' / 'bootstrap_ci.json')
    return data


# ============================================================================
# FIGURE 1: CV ABLATION BAR CHART (THE KEY FIGURE)
# ============================================================================

def fig_cv_ablation(data):
    """Bar chart showing CV ablation results with error bars."""
    cv = data['cv_ablations']
    if not cv:
        print("  Skipping cv_ablation: no data")
        return

    # Order configs by importance
    config_order = ['full_system', 'no_focal', 'no_two_phase',
                    'no_layerwise_lr', 'frozen_only', 'minimal', 'cls_pooling']
    labels = []
    means = []
    stds = []
    fold_vals = []

    for key in config_order:
        if key in cv and 'summary' in cv[key]:
            labels.append(cv[key]['desc'])
            means.append(cv[key]['summary']['mean_f1'])
            stds.append(cv[key]['summary']['std_f1'])
            fold_vals.append(cv[key]['summary']['fold_f1s'])

    if not labels:
        print("  Skipping cv_ablation: no summaries")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    y_pos = np.arange(len(labels))
    colors = []
    for i, key in enumerate([k for k in config_order if k in cv and 'summary' in cv[k]]):
        if key == 'full_system':
            colors.append(SYSTEM_COLORS['ours'])
        elif key == 'cls_pooling':
            colors.append(SYSTEM_COLORS['ablation_bad'])
        else:
            colors.append(SYSTEM_COLORS['ablation_good'])

    bars = ax.barh(y_pos, means, xerr=stds, color=colors,
                    edgecolor='white', height=0.55, capsize=3,
                    error_kw={'linewidth': 1.2, 'capthick': 1.2})

    # Value labels
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_width() + s + 0.005,
                bar.get_y() + bar.get_height()/2,
                f'{m:.3f}±{s:.3f}', va='center', fontsize=8, color=COLORS['dark'])

    # Delta annotations
    base_f1 = means[0] if means else 0
    for i, (bar, m) in enumerate(zip(bars, means)):
        if i > 0:
            delta = m - base_f1
            color = COLORS['secondary'] if abs(delta) > 0.02 else COLORS['neutral']
            ax.text(0.005, bar.get_y() + bar.get_height()/2,
                    f'Δ={delta:+.3f}', va='center', fontsize=7.5,
                    color='white', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Macro F1 (5-fold CV)')
    ax.set_title('Ablation Study (5-Fold Cross-Validation, n=3,531)')
    ax.invert_yaxis()

    # Set x-axis to show relevant range
    x_min = min(means) - max(stds) - 0.03
    x_max = max(means) + max(stds) + 0.06
    ax.set_xlim(max(0, x_min), min(1, x_max))

    save_fig(fig, 'cv_ablation_study')


# ============================================================================
# FIGURE 2: LAYER PROBING
# ============================================================================

def fig_layer_probing(data):
    """Layer-wise probing analysis showing where conspiracy knowledge resides."""
    adv = data['advanced']
    if 'probing' not in adv:
        print("  Skipping layer_probing: no data")
        return

    probing = adv['probing']
    layers = sorted(int(k) for k in probing.keys())
    f1s = [probing[str(l)]['f1'] for l in layers]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.plot(layers, f1s, 'o-', color=COLORS['primary'], linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=2, markeredgecolor=COLORS['primary'])

    # Highlight peak
    peak_idx = np.argmax(f1s)
    ax.plot(layers[peak_idx], f1s[peak_idx], '*', color=COLORS['secondary'],
            markersize=14, zorder=5)
    ax.annotate(f'Peak: {f1s[peak_idx]:.3f}\n(Layer {layers[peak_idx]})',
                xy=(layers[peak_idx], f1s[peak_idx]),
                xytext=(layers[peak_idx] + 2, f1s[peak_idx] + 0.01),
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=1))

    # Add horizontal reference for chance
    ax.axhline(0.5, color=COLORS['neutral'], linestyle=':', alpha=0.5, linewidth=0.8)
    ax.text(0.5, 0.505, 'chance', fontsize=7, color=COLORS['neutral'])

    ax.set_xlabel('DeBERTa Layer')
    ax.set_ylabel('Macro F1 (Frozen Probe)')
    ax.set_title('Layer-wise Probing Analysis')
    ax.set_xticks(layers)

    save_fig(fig, 'layer_probing')


# ============================================================================
# FIGURE 3: SYSTEM COMPARISON
# ============================================================================

def fig_system_comparison(data):
    """Main results overview comparing all systems."""
    baselines = data['baselines']
    models = data['models']

    systems = []
    for key in ['majority', 'tfidf_lr', 'tfidf_svm']:
        if key in baselines:
            systems.append((baselines[key]['name'], baselines[key]['macro_f1'], 'baseline'))

    # Add CLS baseline
    ablations = data['ablations']
    if 'cls_pooling' in ablations:
        systems.append(('DeBERTa ([CLS])', ablations['cls_pooling']['macro_f1'], 'model'))

    if 'dual' in models:
        systems.append(('DeBERTa (CLS+Mean)', models['dual']['macro_f1'], 'model'))

    if 'simple' in models:
        systems.append(('DeBERTa (Mean Pool)', models['simple']['macro_f1'], 'ours'))

    if not systems:
        print("  Skipping system_comparison: no data")
        return

    fig, ax = plt.subplots(figsize=(5, 3.2))

    names = [s[0] for s in systems]
    f1s = [s[1] for s in systems]
    colors = [SYSTEM_COLORS.get(s[2], COLORS['neutral']) for s in systems]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, f1s, color=colors, edgecolor='white', height=0.55)

    for bar, f1val in zip(bars, f1s):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
                f'{f1val:.3f}', va='center', fontsize=9, color=COLORS['dark'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Macro F1')
    ax.set_title('System Comparison (Dev Set)')
    ax.invert_yaxis()
    ax.set_xlim(0, max(f1s) * 1.12)

    legend_elements = [
        mpatches.Patch(facecolor=SYSTEM_COLORS['baseline'], label='Baselines'),
        mpatches.Patch(facecolor=SYSTEM_COLORS['model'], label='Neural variants'),
        mpatches.Patch(facecolor=SYSTEM_COLORS['ours'], label='Our system'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

    save_fig(fig, 'results_overview')


# ============================================================================
# FIGURE 4: CONFUSION MATRIX
# ============================================================================

def fig_confusion_matrix(data):
    """Confusion matrix heatmap."""
    models = data['models']
    if 'simple' not in models or 'confusion_matrix' not in models['simple']:
        print("  Skipping confusion_matrix: no data")
        return

    cm = np.array(models['simple']['confusion_matrix'])

    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                cbar=False, annot_kws={'size': 14, 'fontweight': 'bold'},
                linewidths=0.5, linecolor='white')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (n={cm.sum()}, F1={models["simple"]["macro_f1"]:.3f})')

    save_fig(fig, 'confusion_matrix')


# ============================================================================
# FIGURE 5: PROBABILITY DISTRIBUTION
# ============================================================================

def fig_prob_distribution(data):
    """Predicted probability distribution by class."""
    models = data['models']
    if 'simple' not in models or 'probs' not in models['simple']:
        print("  Skipping prob_distribution: no data")
        return

    probs = np.array(models['simple']['probs'])
    labels = np.array(models['simple']['labels'])
    threshold = models['simple']['threshold']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    bins = np.linspace(0, 1, 25)
    ax.hist(probs[labels == 0], bins=bins, alpha=0.65, label='Non-conspiracy',
            color=COLORS['primary'], edgecolor='white', linewidth=0.5)
    ax.hist(probs[labels == 1], bins=bins, alpha=0.65, label='Conspiracy',
            color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
    ax.axvline(threshold, color=COLORS['dark'], linestyle='--', linewidth=1.5,
               label=f'τ = {threshold:.3f}')

    ax.set_xlabel('P(Conspiracy)')
    ax.set_ylabel('Count')
    ax.set_title('Predicted Probability Distribution')
    ax.legend(framealpha=0.9)

    save_fig(fig, 'prob_distribution')


# ============================================================================
# FIGURE 6: THRESHOLD SENSITIVITY
# ============================================================================

def fig_threshold_sensitivity(data):
    """Threshold sweep showing F1/P/R trade-off."""
    models = data['models']
    if 'simple' not in models or 'probs' not in models['simple']:
        print("  Skipping threshold_sensitivity: no data")
        return

    from sklearn.metrics import precision_recall_fscore_support

    probs = np.array(models['simple']['probs'])
    labels = np.array(models['simple']['labels'])

    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        f1s.append(f)
        precs.append(p)
        recs.append(r)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(thresholds, f1s, '-', color=COLORS['primary'], linewidth=2, label='Macro F1')
    ax.plot(thresholds, precs, '--', color=COLORS['success'], linewidth=1.2, label='Precision')
    ax.plot(thresholds, recs, '--', color=COLORS['secondary'], linewidth=1.2, label='Recall')

    best_idx = np.argmax(f1s)
    ax.plot(thresholds[best_idx], f1s[best_idx], '*', color=COLORS['warning'],
            markersize=12, zorder=5,
            label=f'Best: {f1s[best_idx]:.3f} (τ={thresholds[best_idx]:.2f})')

    ax.set_xlabel('Threshold (τ)')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Sensitivity')
    ax.legend(framealpha=0.9)

    # Shade ±0.03 around best
    ax.axvspan(thresholds[best_idx] - 0.03, thresholds[best_idx] + 0.03,
               alpha=0.1, color=COLORS['primary'])

    save_fig(fig, 'threshold_sensitivity')


# ============================================================================
# FIGURE 7: TRAINING CURVES
# ============================================================================

def fig_training_curves(data):
    """Two-phase training dynamics."""
    ablations = data['ablations']
    if 'full_system' not in ablations or 'epoch_log' not in ablations['full_system']:
        print("  Skipping training_curves: no data")
        return

    log = ablations['full_system']['epoch_log']
    epochs = list(range(1, len(log) + 1))
    losses = [e['loss'] for e in log]
    f1s_val = [e['val_f1'] for e in log]
    phases = [e['phase'] for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    p1_mask = [i for i, p in enumerate(phases) if p == 1]
    p2_mask = [i for i, p in enumerate(phases) if p == 2]
    p0_mask = [i for i, p in enumerate(phases) if p == 0]

    phase_styles = {
        1: {'color': COLORS['primary'], 'label': 'Phase 1 (frozen)'},
        2: {'color': COLORS['secondary'], 'label': 'Phase 2 (fine-tune)'},
        0: {'color': COLORS['success'], 'label': 'Single phase'},
    }

    for ax, values, ylabel, title in [
        (ax1, losses, 'Training Loss', 'Training Loss'),
        (ax2, f1s_val, 'Macro F1', 'Validation F1')
    ]:
        for phase, mask in [(1, p1_mask), (2, p2_mask), (0, p0_mask)]:
            if mask:
                style = phase_styles[phase]
                ax.plot([epochs[i] for i in mask], [values[i] for i in mask],
                        'o-', markersize=4, linewidth=1.5, **style)
        if p1_mask and p2_mask:
            boundary = epochs[max(p1_mask)] + 0.5
            ax.axvline(boundary, color=COLORS['neutral'], linestyle=':', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, framealpha=0.9)

    # Mark best F1 on right panel
    best_idx = np.argmax(f1s_val)
    ax2.plot(epochs[best_idx], f1s_val[best_idx], '*', color=COLORS['warning'],
             markersize=12, zorder=5)
    ax2.annotate(f'{f1s_val[best_idx]:.3f}',
                 xy=(epochs[best_idx], f1s_val[best_idx]),
                 xytext=(epochs[best_idx] + 0.3, f1s_val[best_idx] - 0.02),
                 fontsize=8)

    fig.suptitle('Two-Phase Training Dynamics', fontsize=11, y=1.02)
    plt.tight_layout()

    save_fig(fig, 'training_curves')


# ============================================================================
# FIGURE 8: K-FOLD CV BOX PLOT
# ============================================================================

def fig_kfold_cv(data):
    """Box plot of k-fold CV results."""
    kfold = data['kfold']
    if not kfold or 'summary' not in kfold:
        print("  Skipping kfold_cv: no data")
        return

    fold_f1s = kfold['summary']['fold_f1s']
    mean_f1 = kfold['summary']['mean_f1']
    std_f1 = kfold['summary']['std_f1']

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    bp = ax.boxplot(fold_f1s, patch_artist=True, widths=0.35,
                    boxprops=dict(facecolor=COLORS['light'], edgecolor=COLORS['primary']),
                    medianprops=dict(color=COLORS['dark'], linewidth=1.5),
                    whiskerprops=dict(color=COLORS['primary']),
                    capprops=dict(color=COLORS['primary']),
                    flierprops=dict(markerfacecolor=COLORS['secondary']))

    # Overlay individual points with jitter
    jitter = np.random.normal(0, 0.02, len(fold_f1s))
    ax.scatter([1 + j for j in jitter], fold_f1s, c=COLORS['secondary'],
               s=40, zorder=5, edgecolors='white', linewidth=0.5)

    ax.axhline(mean_f1, color=COLORS['success'], linestyle='--', linewidth=1,
               label=f'Mean: {mean_f1:.3f} ± {std_f1:.3f}')

    ax.set_ylabel('Macro F1')
    ax.set_title('5-Fold Cross-Validation')
    ax.set_xticklabels(['Full System'])
    ax.legend(fontsize=8, framealpha=0.9)

    save_fig(fig, 'kfold_cv')


# ============================================================================
# FIGURE 9: MULTI-SEED STABILITY
# ============================================================================

def fig_multi_seed(data):
    """Multi-seed stability visualization."""
    ms = data['multi_seed']
    if not ms or 'summary' not in ms:
        print("  Skipping multi_seed: no data")
        return

    seed_results = ms['seeds']
    seeds = sorted(seed_results.keys(), key=lambda x: int(x))
    f1s = [seed_results[s]['macro_f1'] for s in seeds]
    mean_f1 = ms['summary']['mean_f1']
    std_f1 = ms['summary']['std_f1']

    fig, ax = plt.subplots(figsize=(4.5, 3))

    x = np.arange(len(seeds))
    bars = ax.bar(x, f1s, color=COLORS['primary'], edgecolor='white', width=0.5)

    # Color the best seed differently
    best_idx = np.argmax(f1s)
    bars[best_idx].set_color(COLORS['success'])

    ax.axhline(mean_f1, color=COLORS['secondary'], linestyle='--', linewidth=1.2,
               label=f'Mean: {mean_f1:.3f} ± {std_f1:.3f}')
    ax.fill_between([-0.5, len(seeds) - 0.5],
                     mean_f1 - std_f1, mean_f1 + std_f1,
                     alpha=0.15, color=COLORS['secondary'])

    for bar, f in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{f:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Seed {s}' for s in seeds], rotation=30, ha='right')
    ax.set_ylabel('Macro F1 (Dev)')
    ax.set_title(f'Multi-Seed Stability ({len(seeds)} seeds)')
    ax.legend(framealpha=0.9)

    y_min = min(f1s) - 0.03
    y_max = max(f1s) + 0.02
    ax.set_ylim(y_min, y_max)

    save_fig(fig, 'multi_seed')


# ============================================================================
# FIGURE 10: NOISE ROBUSTNESS
# ============================================================================

def fig_noise_robustness(data):
    """Noise robustness line chart."""
    adv = data['advanced']
    if 'noise' not in adv:
        print("  Skipping noise_robustness: no data")
        return

    noise = adv['noise']
    pcts = sorted(noise.keys(), key=lambda x: float(x.replace('%', '')))
    x_vals = [float(p.replace('%', '')) for p in pcts]
    f1s = [noise[p]['f1'] for p in pcts]

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(x_vals, f1s, 'o-', color=COLORS['primary'], linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=2, markeredgecolor=COLORS['primary'])

    # Shade degradation zone
    ax.fill_between(x_vals, f1s, f1s[0], alpha=0.1, color=COLORS['secondary'])

    ax.set_xlabel('Label Noise (%)')
    ax.set_ylabel('Macro F1')
    ax.set_title('Robustness to Label Noise')
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f'{x:.0f}%' for x in x_vals])

    # Annotate degradation
    drop = f1s[0] - f1s[-1]
    ax.annotate(f'Δ = {-drop:+.3f}', xy=(x_vals[-1], f1s[-1]),
                xytext=(x_vals[-1] - 5, f1s[-1] - 0.015),
                fontsize=8, color=COLORS['secondary'],
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1))

    save_fig(fig, 'noise_robustness')


# ============================================================================
# FIGURE 11: ABLATION COMPARISON (DEV + CV SIDE BY SIDE)
# ============================================================================

def fig_ablation_comparison(data):
    """Compare dev ablation vs CV ablation side by side."""
    ablations = data['ablations']
    cv_ablations = data['cv_ablations']

    if not ablations or not cv_ablations:
        print("  Skipping ablation_comparison: no data")
        return

    config_order = ['full_system', 'no_focal', 'no_two_phase',
                    'no_layerwise_lr', 'frozen_only', 'cls_pooling']

    labels = []
    dev_f1s = []
    cv_means = []
    cv_stds = []

    for key in config_order:
        if key in ablations and key in cv_ablations and 'summary' in cv_ablations[key]:
            if key == 'full_system':
                labels.append('Full System')
            else:
                labels.append(ablations[key]['desc'])
            dev_f1s.append(ablations[key]['macro_f1'])
            cv_means.append(cv_ablations[key]['summary']['mean_f1'])
            cv_stds.append(cv_ablations[key]['summary']['std_f1'])

    if not labels:
        print("  Skipping ablation_comparison: no matching configs")
        return

    fig, ax = plt.subplots(figsize=(6, 3.8))

    y = np.arange(len(labels))
    height = 0.3

    bars1 = ax.barh(y - height/2, dev_f1s, height, label='Dev (n=77)',
                     color=COLORS['primary'], edgecolor='white', alpha=0.8)
    bars2 = ax.barh(y + height/2, cv_means, height, xerr=cv_stds,
                     label='5-fold CV (n=3,531)', color=COLORS['secondary'],
                     edgecolor='white', alpha=0.8, capsize=2,
                     error_kw={'linewidth': 1})

    # Value labels
    for bar, val in zip(bars1, dev_f1s):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7.5)
    for bar, m, s in zip(bars2, cv_means, cv_stds):
        ax.text(bar.get_width() + s + 0.003, bar.get_y() + bar.get_height()/2,
                f'{m:.3f}', va='center', fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Macro F1')
    ax.set_title('Ablation: Dev Set vs. Cross-Validation')
    ax.invert_yaxis()
    ax.legend(loc='lower right', framealpha=0.9)

    save_fig(fig, 'ablation_comparison')


# ============================================================================
# FIGURE 12: COMPREHENSIVE SUMMARY (2×2 panel)
# ============================================================================

def fig_summary_panel(data):
    """2×2 summary panel combining key results."""
    models = data['models']
    ablations = data['ablations']
    adv = data['advanced']
    cv = data['cv_ablations']

    fig = plt.figure(figsize=(8, 6.5))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: System comparison
    ax1 = fig.add_subplot(gs[0, 0])
    baselines = data['baselines']
    systems = []
    for key in ['majority', 'tfidf_lr', 'tfidf_svm']:
        if key in baselines:
            systems.append((baselines[key]['name'], baselines[key]['macro_f1'], 'baseline'))
    if 'cls_pooling' in ablations:
        systems.append(('DeBERTa ([CLS])', ablations['cls_pooling']['macro_f1'], 'model'))
    if 'simple' in models:
        systems.append(('Mean Pool (Ours)', models['simple']['macro_f1'], 'ours'))

    if systems:
        names = [s[0] for s in systems]
        f1s = [s[1] for s in systems]
        colors = [SYSTEM_COLORS.get(s[2], COLORS['neutral']) for s in systems]
        y = np.arange(len(names))
        bars = ax1.barh(y, f1s, color=colors, edgecolor='white', height=0.55)
        for bar, f in zip(bars, f1s):
            ax1.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
                     f'{f:.3f}', va='center', fontsize=8)
        ax1.set_yticks(y)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel('Macro F1', fontsize=9)
        ax1.set_title('(a) System Comparison', fontsize=10)
        ax1.invert_yaxis()
        ax1.set_xlim(0, max(f1s) * 1.12)

    # Panel B: Layer probing
    ax2 = fig.add_subplot(gs[0, 1])
    if 'probing' in adv:
        probing = adv['probing']
        layers = sorted(int(k) for k in probing.keys())
        f1s = [probing[str(l)]['f1'] for l in layers]
        ax2.plot(layers, f1s, 'o-', color=COLORS['primary'], linewidth=2, markersize=6,
                 markerfacecolor='white', markeredgewidth=1.5)
        peak_idx = np.argmax(f1s)
        ax2.plot(layers[peak_idx], f1s[peak_idx], '*', color=COLORS['secondary'],
                 markersize=12, zorder=5)
        ax2.set_xlabel('Layer', fontsize=9)
        ax2.set_ylabel('Macro F1', fontsize=9)
        ax2.set_title('(b) Layer Probing', fontsize=10)
        ax2.set_xticks(layers)

    # Panel C: CV Ablation (if available)
    ax3 = fig.add_subplot(gs[1, 0])
    if cv:
        config_order = ['full_system', 'no_focal', 'no_two_phase',
                        'no_layerwise_lr', 'frozen_only', 'cls_pooling']
        abl_labels = []
        abl_means = []
        abl_stds = []
        for key in config_order:
            if key in cv and 'summary' in cv[key]:
                abl_labels.append(cv[key]['desc'])
                abl_means.append(cv[key]['summary']['mean_f1'])
                abl_stds.append(cv[key]['summary']['std_f1'])

        if abl_labels:
            y = np.arange(len(abl_labels))
            colors = [SYSTEM_COLORS['ours']] + [SYSTEM_COLORS['ablation_good']] * (len(abl_labels) - 2) + [SYSTEM_COLORS['ablation_bad']]
            if len(colors) != len(abl_labels):
                colors = [SYSTEM_COLORS['ours']] + [SYSTEM_COLORS['ablation_good']] * (len(abl_labels) - 1)
                # Mark cls_pooling as bad
                for i, l in enumerate(abl_labels):
                    if 'CLS' in l or 'Mean Pool' in l:
                        colors[i] = SYSTEM_COLORS['ablation_bad']

            ax3.barh(y, abl_means, xerr=abl_stds, color=colors,
                     edgecolor='white', height=0.5, capsize=2)
            ax3.set_yticks(y)
            ax3.set_yticklabels(abl_labels, fontsize=8)
            ax3.set_xlabel('Macro F1', fontsize=9)
            ax3.set_title('(c) CV Ablation Study', fontsize=10)
            ax3.invert_yaxis()
    else:
        # Fall back to dev ablation
        config_order = ['full_system', 'no_focal', 'no_two_phase',
                        'no_layerwise_lr', 'frozen_only', 'cls_pooling']
        abl_labels = []
        abl_f1s = []
        for key in config_order:
            if key in ablations:
                abl_labels.append(ablations[key]['desc'])
                abl_f1s.append(ablations[key]['macro_f1'])
        if abl_labels:
            y = np.arange(len(abl_labels))
            colors = [SYSTEM_COLORS['ours']] + [SYSTEM_COLORS['ablation_good']] * (len(abl_labels) - 1)
            for i, l in enumerate(abl_labels):
                if 'CLS' in l or 'Mean Pool' in l:
                    colors[i] = SYSTEM_COLORS['ablation_bad']
            ax3.barh(y, abl_f1s, color=colors, edgecolor='white', height=0.5)
            ax3.set_yticks(y)
            ax3.set_yticklabels(abl_labels, fontsize=8)
            ax3.set_xlabel('Macro F1', fontsize=9)
            ax3.set_title('(c) Ablation Study (Dev)', fontsize=10)
            ax3.invert_yaxis()

    # Panel D: Confusion matrix
    ax4 = fig.add_subplot(gs[1, 1])
    if 'simple' in models and 'confusion_matrix' in models['simple']:
        cm = np.array(models['simple']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    cbar=False, annot_kws={'size': 13, 'fontweight': 'bold'},
                    linewidths=0.5, linecolor='white')
        ax4.set_xlabel('Predicted', fontsize=9)
        ax4.set_ylabel('Actual', fontsize=9)
        ax4.set_title('(d) Confusion Matrix', fontsize=10)

    save_fig(fig, 'summary_panel')


# ============================================================================
# FIGURE 13: ERROR ANALYSIS
# ============================================================================

def fig_error_analysis(data):
    """Error analysis visualization."""
    models = data['models']
    if 'simple' not in models or 'probs' not in models['simple']:
        print("  Skipping error_analysis: no data")
        return

    probs = np.array(models['simple']['probs'])
    labels = np.array(models['simple']['labels'])
    threshold = models['simple']['threshold']
    preds = (probs >= threshold).astype(int)

    # Categorize predictions
    tp = (preds == 1) & (labels == 1)
    fp = (preds == 1) & (labels == 0)
    fn = (preds == 0) & (labels == 1)
    tn = (preds == 0) & (labels == 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Left: probability by outcome
    categories = ['TP', 'FP', 'FN', 'TN']
    masks = [tp, fp, fn, tn]
    cat_colors = [COLORS['success'], COLORS['warning'], COLORS['secondary'], COLORS['primary']]

    for cat, mask, color in zip(categories, masks, cat_colors):
        if mask.sum() > 0:
            ax1.scatter(np.where(mask)[0], probs[mask], c=color, label=f'{cat} (n={mask.sum()})',
                        s=30, alpha=0.7, edgecolors='white', linewidth=0.3)

    ax1.axhline(threshold, color=COLORS['dark'], linestyle='--', linewidth=1,
                label=f'τ = {threshold:.3f}')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('P(Conspiracy)')
    ax1.set_title('Predictions by Outcome')
    ax1.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

    # Right: confidence distribution of errors
    error_probs = np.concatenate([probs[fp], probs[fn]]) if (fp.sum() + fn.sum()) > 0 else np.array([])
    correct_probs = np.concatenate([probs[tp], probs[tn]]) if (tp.sum() + tn.sum()) > 0 else np.array([])

    if len(error_probs) > 0:
        # Distance from threshold (uncertainty measure)
        error_dist = np.abs(error_probs - threshold)
        correct_dist = np.abs(correct_probs - threshold)

        ax2.hist(correct_dist, bins=15, alpha=0.6, label='Correct',
                 color=COLORS['success'], edgecolor='white')
        ax2.hist(error_dist, bins=15, alpha=0.6, label='Errors',
                 color=COLORS['secondary'], edgecolor='white')
        ax2.set_xlabel('|P - τ| (Decision Confidence)')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Confidence Analysis')
        ax2.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout()
    save_fig(fig, 'error_analysis')


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Generating publication-quality figures")
    print("=" * 60)

    data = load_all_results()
    print(f"Loaded results: {[k for k, v in data.items() if v]}")

    # Generate all figures
    print("\nGenerating figures...")
    fig_cv_ablation(data)        # Key figure
    fig_layer_probing(data)
    fig_system_comparison(data)
    fig_confusion_matrix(data)
    fig_prob_distribution(data)
    fig_threshold_sensitivity(data)
    fig_training_curves(data)
    fig_kfold_cv(data)
    fig_multi_seed(data)
    fig_noise_robustness(data)
    fig_ablation_comparison(data)
    fig_summary_panel(data)
    fig_error_analysis(data)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("DONE")


if __name__ == '__main__':
    main()
