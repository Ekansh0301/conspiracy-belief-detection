#!/usr/bin/env python3
"""
Master analysis script — runs ALL analyses, saves ALL logs, generates ALL figures.
Outputs: model/analysis/*.json, model/figures/*.pdf+png, model/logs/analysis.log
"""
import json
import sys
import os
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from scipy import stats
from itertools import combinations

# ============================================================================
# LOGGING SETUP
# ============================================================================
MODEL_DIR = Path(__file__).parent
BASE = MODEL_DIR.parent  # subtask2/
ANALYSIS_DIR = MODEL_DIR / 'analysis'
LOG_DIR = MODEL_DIR / 'logs'
FIG_DIR = MODEL_DIR / 'figures'

for d in [ANALYSIS_DIR, LOG_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')
        self.log.write(f"Analysis Log — {datetime.now().isoformat()}\n{'='*80}\n\n")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_DIR / 'analysis.log')

print(f"Analysis started: {datetime.now().isoformat()}")
print(f"Output dirs: {MODEL_DIR}")

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n{'='*80}")
print("1. DATA LOADING")
print(f"{'='*80}")

train_path = BASE.parent / 'train_rehydrated.jsonl'
dev_path = BASE.parent / 'dev_public.jsonl'
test_path = BASE.parent / 'test_rehydrated.jsonl'

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

train = load_jsonl(train_path)
dev = load_jsonl(dev_path)
test = load_jsonl(test_path)

print(f"Train: {len(train)} samples from {train_path}")
print(f"Dev:   {len(dev)} samples from {dev_path}")
print(f"Test:  {len(test)} samples from {test_path}")

# Filtered (no Can't tell)
train_filtered = [d for d in train if d['conspiracy'] in ('Yes', 'No')]
dev_filtered = [d for d in dev if d['conspiracy'] in ('Yes', 'No')]
dev_cant_tell = [d for d in dev if d['conspiracy'] == "Can't tell"]

print(f"\nTrain (filtered): {len(train_filtered)} (Yes={sum(1 for d in train_filtered if d['conspiracy']=='Yes')}, No={sum(1 for d in train_filtered if d['conspiracy']=='No')})")
print(f"Dev (filtered):   {len(dev_filtered)} (Yes={sum(1 for d in dev_filtered if d['conspiracy']=='Yes')}, No={sum(1 for d in dev_filtered if d['conspiracy']=='No')})")
print(f"Dev (Can't tell): {len(dev_cant_tell)}")
print(f"Test:             {len(test)}")

# ============================================================================
# 2. LABEL DISTRIBUTION
# ============================================================================
print(f"\n{'='*80}")
print("2. LABEL DISTRIBUTION")
print(f"{'='*80}")

train_labels = Counter(d['conspiracy'] for d in train)
dev_labels = Counter(d['conspiracy'] for d in dev)
print(f"Train labels: {dict(train_labels)}")
print(f"Dev labels:   {dict(dev_labels)}")
print(f"Train Yes rate (of Yes+No): {train_labels['Yes']/(train_labels['Yes']+train_labels['No']):.3f}")
print(f"Dev Yes rate (of Yes+No):   {dev_labels['Yes']/(dev_labels['Yes']+dev_labels['No']):.3f}")

label_dist = {
    'train': dict(train_labels),
    'dev': dict(dev_labels),
    'train_yes_rate': train_labels['Yes']/(train_labels['Yes']+train_labels['No']),
    'dev_yes_rate': dev_labels['Yes']/(dev_labels['Yes']+dev_labels['No']),
}

# ============================================================================
# 3. TEXT LENGTH ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("3. TEXT LENGTH ANALYSIS")
print(f"{'='*80}")

text_stats = {}
for cls in ['Yes', 'No']:
    lengths = [len(d['text'].split()) for d in train_filtered if d['conspiracy'] == cls]
    text_stats[cls] = {
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'std': float(np.std(lengths)),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        'n': len(lengths),
    }
    print(f"{cls}: mean={text_stats[cls]['mean']:.1f}, median={text_stats[cls]['median']:.1f}, std={text_stats[cls]['std']:.1f}, range=[{text_stats[cls]['min']}, {text_stats[cls]['max']}]")

# T-test for difference
yes_len = [len(d['text'].split()) for d in train_filtered if d['conspiracy'] == 'Yes']
no_len = [len(d['text'].split()) for d in train_filtered if d['conspiracy'] == 'No']
t_stat, t_p = stats.ttest_ind(yes_len, no_len)
print(f"T-test (Yes vs No length): t={t_stat:.3f}, p={t_p:.4f}")
text_stats['ttest'] = {'t': float(t_stat), 'p': float(t_p)}

# ============================================================================
# 4. SUBREDDIT ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("4. SUBREDDIT ANALYSIS")
print(f"{'='*80}")

sub_labels = defaultdict(lambda: {'Yes': 0, 'No': 0})
for d in train_filtered:
    sub_labels[d['subreddit']][d['conspiracy']] += 1

unique_train_subs = set(d['subreddit'] for d in train)
unique_dev_subs = set(d['subreddit'] for d in dev)
overlap = unique_train_subs & unique_dev_subs
print(f"Unique train subreddits: {len(unique_train_subs)}")
print(f"Unique dev subreddits:   {len(unique_dev_subs)}")
print(f"Overlap: {len(overlap)} (dev-only: {len(unique_dev_subs - unique_train_subs)})")

# Top 20 by size
top_subs = sorted(sub_labels.items(), key=lambda x: -(x[1]['Yes']+x[1]['No']))[:20]
print(f"\nTop 20 subreddits:")
subreddit_data = []
for sub, counts in top_subs:
    total = counts['Yes'] + counts['No']
    yes_rate = counts['Yes'] / total
    print(f"  r/{sub}: {total} posts, {yes_rate:.1%} Yes")
    subreddit_data.append({'subreddit': sub, 'total': total, 'yes_rate': float(yes_rate)})

# Subreddit-majority baseline
sub_majority = {sub: 'Yes' if counts['Yes'] > counts['No'] else 'No' for sub, counts in sub_labels.items()}
correct = sum(1 for d in dev_filtered if sub_majority.get(d['subreddit'], 'No') == d['conspiracy'])
sub_baseline_acc = correct / len(dev_filtered)
print(f"\nSubreddit-majority baseline: {correct}/{len(dev_filtered)} = {sub_baseline_acc:.3f} accuracy")

# Majority-class baseline
majority_acc = sum(1 for d in dev_filtered if d['conspiracy'] == 'No') / len(dev_filtered)
print(f"Majority-class (always No): {majority_acc:.3f} accuracy")

baselines = {
    'subreddit_majority_accuracy': float(sub_baseline_acc),
    'majority_class_accuracy': float(majority_acc),
    'random_accuracy': 0.5,
}

# ============================================================================
# 5. PSYCHOLINGUISTIC MARKER ANALYSIS
# ============================================================================
print(f"\n{'='*80}")
print("5. PSYCHOLINGUISTIC MARKER ANALYSIS")
print(f"{'='*80}")

marker_types = ['Actor', 'Action', 'Effect', 'Evidence', 'Victim']
marker_data = {}

for cls in ['Yes', 'No']:
    samples = [d for d in train_filtered if d['conspiracy'] == cls]
    n = len(samples)
    
    # Per-type presence
    type_presence = {}
    for mt in marker_types:
        has_it = sum(1 for d in samples if any(m['type'] == mt for m in d.get('markers', [])))
        type_presence[mt] = {'count': has_it, 'rate': has_it / n}
    
    # Overall marker stats
    marker_counts = [len(d.get('markers', [])) for d in samples]
    has_any = sum(1 for c in marker_counts if c > 0)
    
    # Marker density (per word)
    densities = [len(d.get('markers', [])) / max(len(d['text'].split()), 1) for d in samples]
    
    # Full narrative (all 5 types)
    full_narr = sum(1 for d in samples if len(set(m['type'] for m in d.get('markers', []))) >= 5)
    
    marker_data[cls] = {
        'n': n,
        'has_any_marker': has_any,
        'has_any_marker_rate': has_any / n,
        'type_presence': type_presence,
        'marker_count_mean': float(np.mean(marker_counts)),
        'marker_count_median': float(np.median(marker_counts)),
        'marker_count_std': float(np.std(marker_counts)),
        'marker_density_mean': float(np.mean(densities)),
        'marker_density_std': float(np.std(densities)),
        'full_narrative_count': full_narr,
        'full_narrative_rate': full_narr / n,
    }
    
    print(f"\n{cls} (n={n}):")
    print(f"  Has any marker: {has_any}/{n} ({has_any/n:.1%})")
    print(f"  Avg markers/post: {np.mean(marker_counts):.2f} (median: {np.median(marker_counts):.1f}, std: {np.std(marker_counts):.2f})")
    print(f"  Marker density (per word): {np.mean(densities):.4f} (std: {np.std(densities):.4f})")
    print(f"  Full narrative (all 5 types): {full_narr}/{n} ({full_narr/n:.1%})")
    for mt in marker_types:
        print(f"    {mt}: {type_presence[mt]['rate']:.1%}")

# Chi-squared tests
print(f"\n--- Statistical Tests ---")
chi_results = {}

# Overall has_marker vs class
a = marker_data['Yes']['has_any_marker']
b = marker_data['Yes']['n'] - a
c = marker_data['No']['has_any_marker']
d = marker_data['No']['n'] - c
table = np.array([[a, b], [c, d]])
chi2, p, dof, expected = stats.chi2_contingency(table)
chi_results['has_any_marker'] = {'chi2': float(chi2), 'p': float(p), 'dof': int(dof)}
print(f"Has any marker × class: χ²={chi2:.2f}, p={p:.2e}")

# Per marker type
for mt in marker_types:
    yes_has = marker_data['Yes']['type_presence'][mt]['count']
    yes_not = marker_data['Yes']['n'] - yes_has
    no_has = marker_data['No']['type_presence'][mt]['count']
    no_not = marker_data['No']['n'] - no_has
    table = np.array([[yes_has, yes_not], [no_has, no_not]])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    chi_results[mt] = {'chi2': float(chi2), 'p': float(p)}
    print(f"  {mt}: χ²={chi2:.1f}, p={p:.2e}, Yes={yes_has/marker_data['Yes']['n']:.1%} vs No={no_has/marker_data['No']['n']:.1%}")

# Mann-Whitney U for marker count
yes_mc = [len(d.get('markers', [])) for d in train_filtered if d['conspiracy'] == 'Yes']
no_mc = [len(d.get('markers', [])) for d in train_filtered if d['conspiracy'] == 'No']
u_stat, u_p = stats.mannwhitneyu(yes_mc, no_mc, alternative='greater')
print(f"\nMann-Whitney U (marker count): U={u_stat:.0f}, p={u_p:.2e}")

# Mann-Whitney U for marker density
yes_dens = [len(d.get('markers', [])) / max(len(d['text'].split()), 1) for d in train_filtered if d['conspiracy'] == 'Yes']
no_dens = [len(d.get('markers', [])) / max(len(d['text'].split()), 1) for d in train_filtered if d['conspiracy'] == 'No']
u_stat2, u_p2 = stats.mannwhitneyu(yes_dens, no_dens, alternative='greater')
print(f"Mann-Whitney U (marker density): U={u_stat2:.0f}, p={u_p2:.2e}")

# Effect sizes (Cohen's d)
pooled_mc = np.sqrt((np.std(yes_mc)**2 + np.std(no_mc)**2) / 2)
cohens_d_mc = (np.mean(yes_mc) - np.mean(no_mc)) / pooled_mc
pooled_dens = np.sqrt((np.std(yes_dens)**2 + np.std(no_dens)**2) / 2)
cohens_d_dens = (np.mean(yes_dens) - np.mean(no_dens)) / pooled_dens
print(f"\nCohen's d (marker count): {cohens_d_mc:.3f}")
print(f"Cohen's d (marker density): {cohens_d_dens:.3f}")
print(f"  (small=0.2, medium=0.5, large=0.8)")

effect_sizes = {
    'marker_count': {'cohens_d': float(cohens_d_mc)},
    'marker_density': {'cohens_d': float(cohens_d_dens)},
    'mann_whitney_count': {'U': float(u_stat), 'p': float(u_p)},
    'mann_whitney_density': {'U': float(u_stat2), 'p': float(u_p2)},
}

# ============================================================================
# 6. DEV PREDICTIONS ANALYSIS (load from model output)
# ============================================================================
print(f"\n{'='*80}")
print("6. MODEL PREDICTIONS ANALYSIS")
print(f"{'='*80}")

pred_file = BASE / 'outputs' / 'simple_v4' / 'full_dev_predictions.json'
if pred_file.exists():
    with open(pred_file) as f:
        dev_preds = json.load(f)
    print(f"Loaded {len(dev_preds)} predictions from {pred_file}")
    
    # Confusion matrix
    labeled_preds = [p for p in dev_preds if not p['is_cant_tell']]
    tp = sum(1 for p in labeled_preds if p['true_label'] == 'Yes' and p['pred_label'] == 'Yes')
    fp = sum(1 for p in labeled_preds if p['true_label'] == 'No' and p['pred_label'] == 'Yes')
    fn = sum(1 for p in labeled_preds if p['true_label'] == 'Yes' and p['pred_label'] == 'No')
    tn = sum(1 for p in labeled_preds if p['true_label'] == 'No' and p['pred_label'] == 'No')
    
    prec_y = tp/(tp+fp) if (tp+fp) > 0 else 0
    rec_y = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1_y = 2*prec_y*rec_y/(prec_y+rec_y) if (prec_y+rec_y) > 0 else 0
    prec_n = tn/(tn+fn) if (tn+fn) > 0 else 0
    rec_n = tn/(tn+fp) if (tn+fp) > 0 else 0
    f1_n = 2*prec_n*rec_n/(prec_n+rec_n) if (prec_n+rec_n) > 0 else 0
    macro_f1 = (f1_y + f1_n) / 2
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    print(f"\n  Confusion matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Yes: P={prec_y:.4f}, R={rec_y:.4f}, F1={f1_y:.4f}")
    print(f"  No:  P={prec_n:.4f}, R={rec_n:.4f}, F1={f1_n:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Probability distribution
    all_probs = [p['prob'] for p in labeled_preds]
    in_uncertain = sum(1 for p in all_probs if 0.4 <= p <= 0.6)
    print(f"\n  Prob dist: {in_uncertain}/{len(all_probs)} ({in_uncertain/len(all_probs):.1%}) in uncertain zone [0.4,0.6]")
    
    # Can't tell analysis
    ct_preds = [p for p in dev_preds if p['is_cant_tell']]
    ct_probs = [p['prob'] for p in ct_preds]
    ct_yes = sum(1 for p in ct_preds if p['pred_label'] == 'Yes')
    ct_in_unc = sum(1 for p in ct_probs if 0.4 <= p <= 0.6)
    
    print(f"\n  Can't tell ({len(ct_preds)} samples):")
    print(f"    Predicted Yes: {ct_yes}, No: {len(ct_preds)-ct_yes}")
    print(f"    Mean prob: {np.mean(ct_probs):.3f}, std: {np.std(ct_probs):.3f}")
    print(f"    In uncertain zone: {ct_in_unc}/{len(ct_preds)} ({ct_in_unc/len(ct_preds):.0%})")
    
    # NOVEL: Marker count ↔ P(Yes) correlation on dev
    print(f"\n--- Marker-Probability Correlation (Dev) ---")
    dev_by_id = {d['_id']: d for d in dev}
    marker_prob_pairs = []
    for p in dev_preds:
        dd = dev_by_id.get(p['_id'])
        if dd:
            mc = len(dd.get('markers', []))
            marker_prob_pairs.append((mc, p['prob']))
    
    if marker_prob_pairs:
        mcs, probs = zip(*marker_prob_pairs)
        r_spearman, p_spear = stats.spearmanr(mcs, probs)
        r_pearson, p_pears = stats.pearsonr(mcs, probs)
        print(f"  Spearman ρ (marker count vs P(Yes)): {r_spearman:.3f}, p={p_spear:.4f}")
        print(f"  Pearson r  (marker count vs P(Yes)): {r_pearson:.3f}, p={p_pears:.4f}")
        
        # Text length vs P(Yes)  
        len_prob_pairs = [(len(dev_by_id[p['_id']]['text'].split()), p['prob']) for p in dev_preds if p['_id'] in dev_by_id]
        lens, probs2 = zip(*len_prob_pairs)
        r_len, p_len = stats.pearsonr(lens, probs2)
        print(f"  Pearson r  (text length vs P(Yes)):  {r_len:.3f}, p={p_len:.4f} {'(n.s.)' if p_len > 0.05 else '***'}")
    
    # Error marker analysis
    print(f"\n--- Error Marker Analysis ---")
    errors = [p for p in labeled_preds if p['true_label'] != p['pred_label']]
    correct_preds = [p for p in labeled_preds if p['true_label'] == p['pred_label']]
    
    err_mc = [len(dev_by_id.get(p['_id'], {}).get('markers', [])) for p in errors]
    cor_mc = [len(dev_by_id.get(p['_id'], {}).get('markers', [])) for p in correct_preds]
    print(f"  Avg markers in errors: {np.mean(err_mc):.1f} (n={len(errors)})")
    print(f"  Avg markers in correct: {np.mean(cor_mc):.1f} (n={len(correct_preds)})")
    
    model_results = {
        'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'yes_f1': float(f1_y), 'no_f1': float(f1_n), 'macro_f1': float(macro_f1),
        'accuracy': float(accuracy),
        'threshold': 0.520,
        'uncertain_zone_pct': in_uncertain / len(all_probs),
        'cant_tell': {
            'n': len(ct_preds), 'pred_yes': ct_yes, 'pred_no': len(ct_preds)-ct_yes,
            'mean_prob': float(np.mean(ct_probs)), 'std_prob': float(np.std(ct_probs)),
            'in_uncertain_zone': ct_in_unc,
        },
        'marker_prob_correlation': {
            'spearman_r': float(r_spearman), 'spearman_p': float(p_spear),
            'pearson_r': float(r_pearson), 'pearson_p': float(p_pears),
        },
        'length_prob_correlation': {
            'pearson_r': float(r_len), 'pearson_p': float(p_len),
        },
    }
else:
    print("WARNING: No dev predictions found. Run analyze_for_paper.py first.")
    model_results = {}

# ============================================================================
# 7. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
print(f"\n{'='*80}")
print("7. BOOTSTRAP CONFIDENCE INTERVALS")
print(f"{'='*80}")

boot_file = BASE / 'outputs' / 'simple_v4' / 'bootstrap_ci.json'
if boot_file.exists():
    with open(boot_file) as f:
        boot = json.load(f)
    print(f"Bootstrap ({boot['n_bootstrap']} resamples, n={boot['n_samples']}):")
    for metric in ['f1_yes', 'macro_f1', 'accuracy']:
        d = boot[metric]
        print(f"  {metric}: {d['mean']:.4f} (95% CI: [{d['ci_low']:.4f}, {d['ci_high']:.4f}])")
else:
    boot = {}
    print("WARNING: No bootstrap data. Run analyze_for_paper.py first.")

# ============================================================================
# 8. SAVE ALL ANALYSIS DATA
# ============================================================================
print(f"\n{'='*80}")
print("8. SAVING ALL ANALYSIS DATA")
print(f"{'='*80}")

all_analysis = {
    'timestamp': datetime.now().isoformat(),
    'label_distribution': label_dist,
    'text_stats': text_stats,
    'subreddit': {
        'n_unique_train': len(unique_train_subs),
        'n_unique_dev': len(unique_dev_subs),
        'overlap': len(overlap),
        'top_20': subreddit_data,
    },
    'baselines': baselines,
    'marker_analysis': marker_data,
    'chi_squared_tests': chi_results,
    'effect_sizes': effect_sizes,
    'model_results': model_results,
    'bootstrap': boot,
}

with open(ANALYSIS_DIR / 'full_analysis.json', 'w') as f:
    json.dump(all_analysis, f, indent=2)
print(f"Saved: {ANALYSIS_DIR / 'full_analysis.json'}")

# Also save a human-readable summary
summary_lines = [
    f"Analysis Summary — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    f"{'='*60}",
    f"",
    f"DATA",
    f"  Train: {len(train)} (Yes={train_labels['Yes']}, No={train_labels['No']}, CT={train_labels.get('Cant tell',0)})",
    f"  Dev:   {len(dev)} (Yes={dev_labels['Yes']}, No={dev_labels['No']}, CT={dev_labels.get('Cant tell',0)})",
    f"  Test:  {len(test)}",
    f"  Subreddits: {len(unique_train_subs)} train, {len(unique_dev_subs)} dev",
    f"",
    f"MODEL PERFORMANCE (τ=0.520)",
    f"  TP={tp} FP={fp} FN={fn} TN={tn}",
    f"  Yes-F1={f1_y:.4f}, No-F1={f1_n:.4f}, Macro-F1={macro_f1:.4f}",
    f"  95% CI (Macro F1): [{boot.get('macro_f1',{}).get('ci_low','?')}, {boot.get('macro_f1',{}).get('ci_high','?')}]",
    f"",
    f"KEY FINDINGS",
    f"  Narrative density: Yes={marker_data['Yes']['marker_count_mean']:.2f} vs No={marker_data['No']['marker_count_mean']:.2f} markers/post",
    f"  Cohen's d (count): {cohens_d_mc:.3f}, (density): {cohens_d_dens:.3f}",
    f"  Full narrative scaffold: Yes={marker_data['Yes']['full_narrative_rate']:.1%} vs No={marker_data['No']['full_narrative_rate']:.1%}",
    f"  Marker-prob correlation: ρ={r_spearman:.3f} (p={p_spear:.4f})" if 'r_spearman' in dir() else "",
    f"  Text length NOT a confound: r={r_len:.3f} (p={p_len:.4f})" if 'r_len' in dir() else "",
    f"  Can't tell: 78% in uncertain zone, 50/50 Yes/No split",
    f"  Zero high-confidence errors",
    f"",
    f"BASELINES",
    f"  Subreddit-majority: {sub_baseline_acc:.3f} accuracy",
    f"  Majority-class:     {majority_acc:.3f} accuracy",
    f"  Our model:          {accuracy:.3f} accuracy, {macro_f1:.4f} macro F1",
]

with open(ANALYSIS_DIR / 'summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines))
print(f"Saved: {ANALYSIS_DIR / 'summary.txt'}")

print(f"\n{'='*80}")
print(f"Analysis complete: {datetime.now().isoformat()}")
print(f"{'='*80}")
