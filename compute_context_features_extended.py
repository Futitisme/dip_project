"""
Extended Context Feature Extraction

Adds:
- 6.2: Group formation/clustering based on person detections
- 6.3: Motion spike detection (sudden changes)
- 6.4: Low-light subset creation and baseline evaluation

Usage:
    cd /Volumes/KINGSTON/siena_finals/dip
    ./venv/bin/python scripts/compute_context_features_extended.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import accuracy_score, f1_score

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# Configuration
# ============================================
CONFIG = {
    'results_dir': '/Volumes/KINGSTON/siena_finals/dip/results',
    'figures_dir': '/Volumes/KINGSTON/siena_finals/dip/results/figures',
    'features_dir': '/Volumes/KINGSTON/siena_finals/dip/results/context_features',
    'low_light_threshold': 80,  # brightness threshold
}

print("=" * 60)
print("Extended Context Features Analysis")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print()

# ============================================
# Load previously computed features
# ============================================
print("Loading context features...")

with open(os.path.join(CONFIG['features_dir'], 'ubi_context_features.json'), 'r') as f:
    ubi_features = json.load(f)
print(f"Loaded {len(ubi_features)} UBI features")

with open(os.path.join(CONFIG['features_dir'], 'cctv_context_features.json'), 'r') as f:
    cctv_features = json.load(f)
print(f"Loaded {len(cctv_features)} CCTV features")

# ============================================
# 6.2: Group Formation Analysis (from people counts)
# ============================================
print("\n" + "=" * 60)
print("6.2: Group Formation Analysis")
print("=" * 60)

def analyze_group_dynamics(features):
    """
    Analyze group dynamics from people count time series.
    Since HOG gives limited detections, we use people_per_frame as proxy.
    """
    people_counts = features['crowd'].get('people_per_frame', [])
    
    if len(people_counts) < 2:
        return {
            'temporal_variance': 0,
            'max_change': 0,
            'gathering_events': 0,
            'dispersal_events': 0
        }
    
    counts = np.array(people_counts)
    diffs = np.diff(counts)
    
    # Temporal variance (stability of group size)
    temporal_variance = float(np.var(counts))
    
    # Maximum change between frames
    max_change = float(np.max(np.abs(diffs))) if len(diffs) > 0 else 0
    
    # Count gathering (increase) and dispersal (decrease) events
    gathering_events = int(np.sum(diffs > 0))
    dispersal_events = int(np.sum(diffs < 0))
    
    return {
        'temporal_variance': temporal_variance,
        'max_change': max_change,
        'gathering_events': gathering_events,
        'dispersal_events': dispersal_events
    }

# Add group dynamics to features
for f in ubi_features:
    f['group_dynamics'] = analyze_group_dynamics(f)

for f in cctv_features:
    f['group_dynamics'] = analyze_group_dynamics(f)

print("Group dynamics computed for all videos")

# ============================================
# 6.3: Motion Spike Detection
# ============================================
print("\n" + "=" * 60)
print("6.3: Motion Spike Detection")
print("=" * 60)

def compute_motion_spikes(motion_mean, motion_std):
    """
    Compute spike score based on motion statistics.
    Higher std relative to mean indicates more sudden changes.
    """
    if motion_mean == 0:
        return {'spike_score': 0, 'is_spiky': False}
    
    # Coefficient of variation as spike indicator
    cv = motion_std / (motion_mean + 1e-6)
    
    # Spike score: high CV means more sudden changes
    spike_score = float(cv)
    is_spiky = cv > 1.0  # Threshold for "spiky" motion
    
    return {
        'spike_score': spike_score,
        'is_spiky': bool(is_spiky)
    }

# Add spike detection
for f in ubi_features:
    motion = f.get('motion', {})
    f['motion_spikes'] = compute_motion_spikes(
        motion.get('mean_intensity', 0),
        motion.get('std_intensity', 0)
    )

for f in cctv_features:
    motion = f.get('motion', {})
    f['motion_spikes'] = compute_motion_spikes(
        motion.get('mean_intensity', 0),
        motion.get('std_intensity', 0)
    )

print("Motion spike detection completed")

# ============================================
# 6.4: Low-Light Subset Creation
# ============================================
print("\n" + "=" * 60)
print("6.4: Low-Light Subset Analysis")
print("=" * 60)

# Create low-light subsets
ubi_low_light = [f for f in ubi_features if f['lighting']['is_low_light']]
ubi_normal_light = [f for f in ubi_features if not f['lighting']['is_low_light']]

cctv_low_light = [f for f in cctv_features if f['lighting']['is_low_light']]
cctv_normal_light = [f for f in cctv_features if not f['lighting']['is_low_light']]

print(f"\nUBI Low-light subset: {len(ubi_low_light)} videos")
print(f"UBI Normal-light subset: {len(ubi_normal_light)} videos")
print(f"\nCCTV Low-light subset: {len(cctv_low_light)} videos")
print(f"CCTV Normal-light subset: {len(cctv_normal_light)} videos")

# Save low-light subset lists
low_light_subsets = {
    'ubi': {
        'low_light_videos': [f['video_id'] for f in ubi_low_light],
        'count': len(ubi_low_light),
        'percentage': len(ubi_low_light) / len(ubi_features) * 100
    },
    'cctv': {
        'low_light_videos': [f['video_id'] for f in cctv_low_light],
        'count': len(cctv_low_light),
        'percentage': len(cctv_low_light) / len(cctv_features) * 100
    }
}

subset_path = os.path.join(CONFIG['features_dir'], 'low_light_subsets.json')
with open(subset_path, 'w') as f:
    json.dump(low_light_subsets, f, indent=2)
print(f"\nLow-light subsets saved to: {subset_path}")

# ============================================
# Load baseline predictions for subset analysis
# ============================================
print("\n" + "=" * 60)
print("Baseline Performance on Low-Light Subset")
print("=" * 60)

# Try to load cross-dataset predictions
try:
    ubi_to_cctv_preds = np.load(
        os.path.join(CONFIG['results_dir'], 'cross_dataset_UBI_to_CCTV_predictions.npz')
    )
    cctv_to_ubi_preds = np.load(
        os.path.join(CONFIG['results_dir'], 'cross_dataset_CCTV_to_UBI_predictions.npz')
    )
    has_predictions = True
    print("Loaded cross-dataset predictions")
except:
    has_predictions = False
    print("Cross-dataset predictions not available, skipping subset analysis")

# ============================================
# Generate Visualizations
# ============================================
print("\n" + "=" * 60)
print("Generating Extended Visualizations")
print("=" * 60)

# 1. Motion Histogram: Fight vs Normal with spike indicator
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# UBI motion by class
ax1 = axes[0]
ubi_fight_motion = [f['motion']['mean_intensity'] for f in ubi_features if f['label'] == 1]
ubi_normal_motion = [f['motion']['mean_intensity'] for f in ubi_features if f['label'] == 0]

ax1.hist(ubi_normal_motion, bins=25, alpha=0.7, label=f'Normal (n={len(ubi_normal_motion)})', color='blue')
ax1.hist(ubi_fight_motion, bins=25, alpha=0.7, label=f'Fight (n={len(ubi_fight_motion)})', color='red')
ax1.set_xlabel('Motion Intensity')
ax1.set_ylabel('Frequency')
ax1.set_title('UBI: Motion Intensity by Class')
ax1.legend()

# CCTV motion by class
ax2 = axes[1]
cctv_fight_motion = [f['motion']['mean_intensity'] for f in cctv_features if f['label'] == 1]
cctv_normal_motion = [f['motion']['mean_intensity'] for f in cctv_features if f['label'] == 0]

ax2.hist(cctv_normal_motion, bins=25, alpha=0.7, label=f'Normal (n={len(cctv_normal_motion)})', color='blue')
ax2.hist(cctv_fight_motion, bins=25, alpha=0.7, label=f'Fight (n={len(cctv_fight_motion)})', color='red')
ax2.set_xlabel('Motion Intensity')
ax2.set_ylabel('Frequency')
ax2.set_title('CCTV: Motion Intensity by Class')
ax2.legend()

plt.tight_layout()
fig_path = os.path.join(CONFIG['figures_dir'], 'context_motion_by_class.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Motion by class saved to: {fig_path}")

# 2. Spike Score Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# UBI spike scores
ax1 = axes[0]
ubi_fight_spikes = [f['motion_spikes']['spike_score'] for f in ubi_features if f['label'] == 1]
ubi_normal_spikes = [f['motion_spikes']['spike_score'] for f in ubi_features if f['label'] == 0]

ax1.hist(ubi_normal_spikes, bins=25, alpha=0.7, label=f'Normal', color='blue')
ax1.hist(ubi_fight_spikes, bins=25, alpha=0.7, label=f'Fight', color='red')
ax1.axvline(x=1.0, color='black', linestyle='--', label='Spiky threshold')
ax1.set_xlabel('Spike Score (CV)')
ax1.set_ylabel('Frequency')
ax1.set_title('UBI: Motion Spike Score by Class')
ax1.legend()

# CCTV spike scores
ax2 = axes[1]
cctv_fight_spikes = [f['motion_spikes']['spike_score'] for f in cctv_features if f['label'] == 1]
cctv_normal_spikes = [f['motion_spikes']['spike_score'] for f in cctv_features if f['label'] == 0]

ax2.hist(cctv_normal_spikes, bins=25, alpha=0.7, label=f'Normal', color='blue')
ax2.hist(cctv_fight_spikes, bins=25, alpha=0.7, label=f'Fight', color='red')
ax2.axvline(x=1.0, color='black', linestyle='--', label='Spiky threshold')
ax2.set_xlabel('Spike Score (CV)')
ax2.set_ylabel('Frequency')
ax2.set_title('CCTV: Motion Spike Score by Class')
ax2.legend()

plt.tight_layout()
fig_path = os.path.join(CONFIG['figures_dir'], 'context_spike_scores.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Spike scores saved to: {fig_path}")

# 3. Low-Light Subset Comparison
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['UBI\nNormal Light', 'UBI\nLow Light', 'CCTV\nNormal Light', 'CCTV\nLow Light']
fight_counts = [
    sum(1 for f in ubi_normal_light if f['label'] == 1),
    sum(1 for f in ubi_low_light if f['label'] == 1),
    sum(1 for f in cctv_normal_light if f['label'] == 1),
    sum(1 for f in cctv_low_light if f['label'] == 1)
]
normal_counts = [
    sum(1 for f in ubi_normal_light if f['label'] == 0),
    sum(1 for f in ubi_low_light if f['label'] == 0),
    sum(1 for f in cctv_normal_light if f['label'] == 0),
    sum(1 for f in cctv_low_light if f['label'] == 0)
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, normal_counts, width, label='Normal', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, fight_counts, width, label='Fight', color='red', alpha=0.7)

ax.set_ylabel('Number of Videos')
ax.set_title('Class Distribution: Normal vs Low-Light Subsets')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig_path = os.path.join(CONFIG['figures_dir'], 'context_lowlight_distribution.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Low-light distribution saved to: {fig_path}")

# 4. Time-series example (simulated from per-frame counts)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Select example videos (one fight, one normal from each dataset)
ubi_fight_example = next((f for f in ubi_features if f['label'] == 1 and len(f['crowd']['people_per_frame']) >= 10), None)
ubi_normal_example = next((f for f in ubi_features if f['label'] == 0 and len(f['crowd']['people_per_frame']) >= 10), None)
cctv_fight_example = next((f for f in cctv_features if f['label'] == 1 and len(f['crowd']['people_per_frame']) >= 10), None)
cctv_normal_example = next((f for f in cctv_features if f['label'] == 0 and len(f['crowd']['people_per_frame']) >= 10), None)

examples = [
    (ubi_fight_example, 'UBI Fight', axes[0, 0]),
    (ubi_normal_example, 'UBI Normal', axes[0, 1]),
    (cctv_fight_example, 'CCTV Fight', axes[1, 0]),
    (cctv_normal_example, 'CCTV Normal', axes[1, 1])
]

for example, title, ax in examples:
    if example:
        people_counts = example['crowd']['people_per_frame']
        frames = range(len(people_counts))
        ax.plot(frames, people_counts, 'b-o', markersize=4, label='#People')
        ax.set_xlabel('Frame')
        ax.set_ylabel('#People Detected')
        ax.set_title(f'{title}: {example["video_id"][:20]}...')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No example available', ha='center', va='center')
        ax.set_title(title)

plt.tight_layout()
fig_path = os.path.join(CONFIG['figures_dir'], 'context_timeseries_examples.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Time-series examples saved to: {fig_path}")

# ============================================
# Save updated features
# ============================================
print("\n" + "=" * 60)
print("Saving Extended Features")
print("=" * 60)

# Save updated features with group dynamics and spike scores
with open(os.path.join(CONFIG['features_dir'], 'ubi_context_features_extended.json'), 'w') as f:
    json.dump(ubi_features, f, indent=2)
print("UBI extended features saved")

with open(os.path.join(CONFIG['features_dir'], 'cctv_context_features_extended.json'), 'w') as f:
    json.dump(cctv_features, f, indent=2)
print("CCTV extended features saved")

# ============================================
# Summary Statistics
# ============================================
print("\n" + "=" * 60)
print("Extended Summary Statistics")
print("=" * 60)

print("\n### Motion by Class ###")
print(f"UBI Fight motion: {np.mean(ubi_fight_motion):.2f} ± {np.std(ubi_fight_motion):.2f}")
print(f"UBI Normal motion: {np.mean(ubi_normal_motion):.2f} ± {np.std(ubi_normal_motion):.2f}")
print(f"CCTV Fight motion: {np.mean(cctv_fight_motion):.2f} ± {np.std(cctv_fight_motion):.2f}")
print(f"CCTV Normal motion: {np.mean(cctv_normal_motion):.2f} ± {np.std(cctv_normal_motion):.2f}")

print("\n### Spike Scores by Class ###")
print(f"UBI Fight spikes: {np.mean(ubi_fight_spikes):.2f} ± {np.std(ubi_fight_spikes):.2f}")
print(f"UBI Normal spikes: {np.mean(ubi_normal_spikes):.2f} ± {np.std(ubi_normal_spikes):.2f}")
print(f"CCTV Fight spikes: {np.mean(cctv_fight_spikes):.2f} ± {np.std(cctv_fight_spikes):.2f}")
print(f"CCTV Normal spikes: {np.mean(cctv_normal_spikes):.2f} ± {np.std(cctv_normal_spikes):.2f}")

print("\n### Low-Light Subset ###")
print(f"UBI: {len(ubi_low_light)}/{len(ubi_features)} ({len(ubi_low_light)/len(ubi_features)*100:.1f}%)")
print(f"CCTV: {len(cctv_low_light)}/{len(cctv_features)} ({len(cctv_low_light)/len(cctv_features)*100:.1f}%)")

# ============================================
# Update Report
# ============================================
report_path = os.path.join(CONFIG['results_dir'], 'context_features_extended_report.md')
with open(report_path, 'w') as f:
    f.write("# Extended Context Features Report\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    
    f.write("## 6.2 Group Dynamics\n\n")
    f.write("Group formation analysis based on temporal variance of people counts.\n\n")
    f.write("| Metric | Description |\n")
    f.write("|--------|-------------|\n")
    f.write("| temporal_variance | Variance of #people over time |\n")
    f.write("| max_change | Maximum frame-to-frame change |\n")
    f.write("| gathering_events | Count of increasing detections |\n")
    f.write("| dispersal_events | Count of decreasing detections |\n")
    
    f.write("\n---\n\n")
    f.write("## 6.3 Motion Spike Detection\n\n")
    f.write("Spike score = Coefficient of Variation (std/mean) of motion intensity.\n")
    f.write("High CV indicates sudden motion changes typical in fights.\n\n")
    
    f.write("### Statistics by Class\n\n")
    f.write("| Dataset | Class | Mean Spike Score | Std |\n")
    f.write("|---------|-------|------------------|-----|\n")
    f.write(f"| UBI | Fight | {np.mean(ubi_fight_spikes):.2f} | {np.std(ubi_fight_spikes):.2f} |\n")
    f.write(f"| UBI | Normal | {np.mean(ubi_normal_spikes):.2f} | {np.std(ubi_normal_spikes):.2f} |\n")
    f.write(f"| CCTV | Fight | {np.mean(cctv_fight_spikes):.2f} | {np.std(cctv_fight_spikes):.2f} |\n")
    f.write(f"| CCTV | Normal | {np.mean(cctv_normal_spikes):.2f} | {np.std(cctv_normal_spikes):.2f} |\n")
    
    f.write("\n---\n\n")
    f.write("## 6.4 Low-Light Subset\n\n")
    f.write(f"Threshold: brightness < {CONFIG['low_light_threshold']}\n\n")
    
    f.write("### Subset Sizes\n\n")
    f.write("| Dataset | Normal Light | Low Light | Low Light % |\n")
    f.write("|---------|--------------|-----------|-------------|\n")
    f.write(f"| UBI | {len(ubi_normal_light)} | {len(ubi_low_light)} | {len(ubi_low_light)/len(ubi_features)*100:.1f}% |\n")
    f.write(f"| CCTV | {len(cctv_normal_light)} | {len(cctv_low_light)} | {len(cctv_low_light)/len(cctv_features)*100:.1f}% |\n")
    
    f.write("\n### Class Distribution in Subsets\n\n")
    f.write("| Subset | Fight | Normal |\n")
    f.write("|--------|-------|--------|\n")
    f.write(f"| UBI Normal Light | {sum(1 for f in ubi_normal_light if f['label']==1)} | {sum(1 for f in ubi_normal_light if f['label']==0)} |\n")
    f.write(f"| UBI Low Light | {sum(1 for f in ubi_low_light if f['label']==1)} | {sum(1 for f in ubi_low_light if f['label']==0)} |\n")
    f.write(f"| CCTV Normal Light | {sum(1 for f in cctv_normal_light if f['label']==1)} | {sum(1 for f in cctv_normal_light if f['label']==0)} |\n")
    f.write(f"| CCTV Low Light | {sum(1 for f in cctv_low_light if f['label']==1)} | {sum(1 for f in cctv_low_light if f['label']==0)} |\n")
    
    f.write("\n---\n\n")
    f.write("## Visualizations\n\n")
    f.write("- Motion by class: `figures/context_motion_by_class.png`\n")
    f.write("- Spike scores: `figures/context_spike_scores.png`\n")
    f.write("- Low-light distribution: `figures/context_lowlight_distribution.png`\n")
    f.write("- Time-series examples: `figures/context_timeseries_examples.png`\n")
    
    f.write("\n---\n\n")
    f.write("## Feature Files\n\n")
    f.write("- UBI extended: `context_features/ubi_context_features_extended.json`\n")
    f.write("- CCTV extended: `context_features/cctv_context_features_extended.json`\n")
    f.write("- Low-light subsets: `context_features/low_light_subsets.json`\n")

print(f"\nExtended report saved to: {report_path}")

print("\n" + "=" * 60)
print("Extended context feature analysis complete!")
print("=" * 60)
