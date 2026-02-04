"""
Context Feature Extraction Module

Computes contextual features for violence detection:
1. Crowd density / number of people (using HOG person detector)
2. Motion intensity (optical flow statistics)
3. Low-light proxy (brightness/contrast)

Usage:
    cd /Volumes/KINGSTON/siena_finals/dip
    ./venv/bin/python scripts/compute_context_features.py
"""

import os
import sys
import cv2
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# Configuration
# ============================================
CONFIG = {
    'results_dir': '/Volumes/KINGSTON/siena_finals/dip/results',
    'figures_dir': '/Volumes/KINGSTON/siena_finals/dip/results/figures',
    'features_dir': '/Volumes/KINGSTON/siena_finals/dip/results/context_features',
    'sample_frames': 16,  # Number of frames to sample per video
    'resize_width': 640,  # Resize for faster processing
    'resize_height': 480,
}

# Create directories
os.makedirs(CONFIG['features_dir'], exist_ok=True)
os.makedirs(CONFIG['figures_dir'], exist_ok=True)

print("=" * 60)
print("Context Feature Extraction")
print("=" * 60)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print()

# ============================================
# Initialize Person Detector (HOG + SVM)
# ============================================
print("Initializing HOG person detector...")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
print("✅ HOG detector ready")

# ============================================
# Feature extraction functions
# ============================================

def detect_people(frame, hog_detector):
    """
    Detect people in a frame using HOG + SVM.
    Returns number of people detected and bounding boxes.
    """
    # Resize for faster processing
    h, w = frame.shape[:2]
    scale = min(CONFIG['resize_width'] / w, CONFIG['resize_height'] / h)
    if scale < 1:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    
    # Detect people
    boxes, weights = hog_detector.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05,
        hitThreshold=0.3
    )
    
    return len(boxes), boxes, weights


def compute_motion_intensity(prev_gray, curr_gray):
    """
    Compute motion intensity using optical flow.
    Returns mean and max flow magnitude.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    return {
        'mean': float(np.mean(mag)),
        'max': float(np.max(mag)),
        'std': float(np.std(mag))
    }


def compute_lighting_features(frame):
    """
    Compute lighting/brightness features.
    Returns brightness and contrast metrics.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Histogram-based features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Low-light indicator (percentage of dark pixels)
    dark_ratio = np.sum(hist[:50])  # pixels with value < 50
    
    return {
        'brightness': float(brightness),
        'contrast': float(contrast),
        'dark_ratio': float(dark_ratio),
        'is_low_light': bool(brightness < 80 or dark_ratio > 0.5)
    }


def extract_video_features(video_path, video_id, hog_detector, num_frames=16):
    """
    Extract all context features from a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames < 2:
        cap.release()
        return None
    
    # Sample frame indices
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames // num_frames
        frame_indices = [i * step for i in range(num_frames)]
    
    # Storage for frame-level features
    people_counts = []
    motion_intensities = []
    lighting_features = []
    
    prev_gray = None
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize for consistent processing
        frame = cv2.resize(frame, (CONFIG['resize_width'], CONFIG['resize_height']))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. People detection
        num_people, _, _ = detect_people(frame, hog_detector)
        people_counts.append(num_people)
        
        # 2. Motion intensity (requires previous frame)
        if prev_gray is not None:
            motion = compute_motion_intensity(prev_gray, gray)
            motion_intensities.append(motion)
        
        # 3. Lighting features
        lighting = compute_lighting_features(frame)
        lighting_features.append(lighting)
        
        prev_gray = gray
    
    cap.release()
    
    if len(people_counts) == 0:
        return None
    
    # Aggregate features
    features = {
        'video_id': video_id,
        'video_path': video_path,
        'total_frames': total_frames,
        'fps': fps,
        'sampled_frames': len(frame_indices),
        
        # Crowd density features
        'crowd': {
            'mean_people': float(np.mean(people_counts)),
            'median_people': float(np.median(people_counts)),
            'max_people': int(np.max(people_counts)),
            'min_people': int(np.min(people_counts)),
            'std_people': float(np.std(people_counts)),
            'people_per_frame': people_counts
        },
        
        # Motion features
        'motion': {
            'mean_intensity': float(np.mean([m['mean'] for m in motion_intensities])) if motion_intensities else 0,
            'max_intensity': float(np.max([m['max'] for m in motion_intensities])) if motion_intensities else 0,
            'std_intensity': float(np.std([m['mean'] for m in motion_intensities])) if motion_intensities else 0,
        },
        
        # Lighting features
        'lighting': {
            'mean_brightness': float(np.mean([l['brightness'] for l in lighting_features])),
            'mean_contrast': float(np.mean([l['contrast'] for l in lighting_features])),
            'mean_dark_ratio': float(np.mean([l['dark_ratio'] for l in lighting_features])),
            'is_low_light': bool(np.mean([l['is_low_light'] for l in lighting_features]) > 0.5)
        }
    }
    
    return features


# ============================================
# Process UBI-Fights dataset
# ============================================
print("\n" + "=" * 60)
print("Processing UBI-Fights dataset...")
print("=" * 60)

# Import UBI data paths
from Data_UBI_optimized import train_videos as ubi_train, test_videos as ubi_test, get_video_path

ubi_all_videos = ubi_train + ubi_test
print(f"Total UBI videos: {len(ubi_all_videos)}")

ubi_features = []
for i, video_name in enumerate(tqdm(ubi_all_videos, desc="UBI videos")):
    video_path = get_video_path(video_name)
    
    # Determine label (fight starts with 'F')
    label = 1 if video_name.startswith('F') else 0
    
    features = extract_video_features(video_path, video_name, hog, CONFIG['sample_frames'])
    
    if features:
        features['label'] = label
        features['dataset'] = 'UBI'
        features['subset'] = 'test' if video_name in ubi_test else 'train'
        ubi_features.append(features)

print(f"Successfully processed: {len(ubi_features)}/{len(ubi_all_videos)} UBI videos")

# ============================================
# Process CCTV-Fights dataset
# ============================================
print("\n" + "=" * 60)
print("Processing CCTV-Fights dataset...")
print("=" * 60)

# Import CCTV data paths
from Data_CCTV_optimized import (
    train_videos as cctv_train, 
    validation_videos as cctv_val, 
    test_videos as cctv_test,
    data as cctv_data
)

cctv_all_videos = cctv_train + cctv_val + cctv_test
print(f"Total CCTV videos: {len(cctv_all_videos)}")

cctv_features = []
for i, (video_id, video_path) in enumerate(tqdm(cctv_all_videos, desc="CCTV videos")):
    # Get label from annotations
    if video_id in cctv_data:
        annotations = cctv_data[video_id].get('annotations', [])
        label = 1 if len(annotations) > 0 else 0
    else:
        label = 0
    
    features = extract_video_features(video_path, video_id, hog, CONFIG['sample_frames'])
    
    if features:
        features['label'] = label
        features['dataset'] = 'CCTV'
        # Determine subset
        if (video_id, video_path) in cctv_test:
            features['subset'] = 'test'
        elif (video_id, video_path) in cctv_val:
            features['subset'] = 'validation'
        else:
            features['subset'] = 'train'
        cctv_features.append(features)

print(f"Successfully processed: {len(cctv_features)}/{len(cctv_all_videos)} CCTV videos")

# ============================================
# Save features
# ============================================
print("\n" + "=" * 60)
print("Saving features...")
print("=" * 60)

# Save as JSON
ubi_features_path = os.path.join(CONFIG['features_dir'], 'ubi_context_features.json')
with open(ubi_features_path, 'w') as f:
    json.dump(ubi_features, f, indent=2)
print(f"UBI features saved to: {ubi_features_path}")

cctv_features_path = os.path.join(CONFIG['features_dir'], 'cctv_context_features.json')
with open(cctv_features_path, 'w') as f:
    json.dump(cctv_features, f, indent=2)
print(f"CCTV features saved to: {cctv_features_path}")

# ============================================
# Generate visualizations
# ============================================
print("\n" + "=" * 60)
print("Generating visualizations...")
print("=" * 60)

# Extract arrays for plotting
ubi_avg_people = [f['crowd']['mean_people'] for f in ubi_features]
cctv_avg_people = [f['crowd']['mean_people'] for f in cctv_features]

ubi_labels = [f['label'] for f in ubi_features]
cctv_labels = [f['label'] for f in cctv_features]

# 1. Histogram: Distribution of avg #people by dataset
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.hist(ubi_avg_people, bins=20, alpha=0.7, label=f'UBI (n={len(ubi_avg_people)})', color='blue')
ax1.hist(cctv_avg_people, bins=20, alpha=0.7, label=f'CCTV (n={len(cctv_avg_people)})', color='orange')
ax1.set_xlabel('Average #People per Video')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Crowd Density: UBI vs CCTV')
ax1.legend()

# 2. Histogram by class (fight vs normal)
ax2 = axes[1]
ubi_fight_people = [ubi_avg_people[i] for i in range(len(ubi_features)) if ubi_labels[i] == 1]
ubi_normal_people = [ubi_avg_people[i] for i in range(len(ubi_features)) if ubi_labels[i] == 0]
cctv_fight_people = [cctv_avg_people[i] for i in range(len(cctv_features)) if cctv_labels[i] == 1]
cctv_normal_people = [cctv_avg_people[i] for i in range(len(cctv_features)) if cctv_labels[i] == 0]

all_fight = ubi_fight_people + cctv_fight_people
all_normal = ubi_normal_people + cctv_normal_people

ax2.hist(all_normal, bins=20, alpha=0.7, label=f'Normal (n={len(all_normal)})', color='green')
ax2.hist(all_fight, bins=20, alpha=0.7, label=f'Fight (n={len(all_fight)})', color='red')
ax2.set_xlabel('Average #People per Video')
ax2.set_ylabel('Frequency')
ax2.set_title('Crowd Density: Fight vs Normal')
ax2.legend()

plt.tight_layout()
hist_path = os.path.join(CONFIG['figures_dir'], 'context_crowd_density_histogram.png')
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"Histogram saved to: {hist_path}")

# 3. Scatter: Motion intensity vs avg #people
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# UBI scatter
ax1 = axes[0]
ubi_motion = [f['motion']['mean_intensity'] for f in ubi_features]
colors_ubi = ['red' if l == 1 else 'blue' for l in ubi_labels]
ax1.scatter(ubi_avg_people, ubi_motion, c=colors_ubi, alpha=0.6, s=30)
ax1.set_xlabel('Average #People')
ax1.set_ylabel('Motion Intensity')
ax1.set_title('UBI: Motion vs Crowd Density')
# Custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Fight'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal')]
ax1.legend(handles=legend_elements)

# CCTV scatter
ax2 = axes[1]
cctv_motion = [f['motion']['mean_intensity'] for f in cctv_features]
colors_cctv = ['red' if l == 1 else 'blue' for l in cctv_labels]
ax2.scatter(cctv_avg_people, cctv_motion, c=colors_cctv, alpha=0.6, s=30)
ax2.set_xlabel('Average #People')
ax2.set_ylabel('Motion Intensity')
ax2.set_title('CCTV: Motion vs Crowd Density')
ax2.legend(handles=legend_elements)

plt.tight_layout()
scatter_path = os.path.join(CONFIG['figures_dir'], 'context_motion_vs_crowd.png')
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"Scatter plot saved to: {scatter_path}")

# 4. Lighting analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ubi_brightness = [f['lighting']['mean_brightness'] for f in ubi_features]
cctv_brightness = [f['lighting']['mean_brightness'] for f in cctv_features]

# Histogram of brightness
ax1 = axes[0]
ax1.hist(ubi_brightness, bins=20, alpha=0.7, label=f'UBI (n={len(ubi_brightness)})', color='blue')
ax1.hist(cctv_brightness, bins=20, alpha=0.7, label=f'CCTV (n={len(cctv_brightness)})', color='orange')
ax1.axvline(x=80, color='red', linestyle='--', label='Low-light threshold')
ax1.set_xlabel('Mean Brightness')
ax1.set_ylabel('Frequency')
ax1.set_title('Brightness Distribution: UBI vs CCTV')
ax1.legend()

# Low-light vs normal classification accuracy potential
ax2 = axes[1]
ubi_low_light = sum(1 for f in ubi_features if f['lighting']['is_low_light'])
ubi_normal_light = len(ubi_features) - ubi_low_light
cctv_low_light = sum(1 for f in cctv_features if f['lighting']['is_low_light'])
cctv_normal_light = len(cctv_features) - cctv_low_light

x = np.arange(2)
width = 0.35
bars1 = ax2.bar(x - width/2, [ubi_normal_light, ubi_low_light], width, label='UBI', color='blue')
bars2 = ax2.bar(x + width/2, [cctv_normal_light, cctv_low_light], width, label='CCTV', color='orange')
ax2.set_ylabel('Number of Videos')
ax2.set_title('Normal vs Low-Light Videos')
ax2.set_xticks(x)
ax2.set_xticklabels(['Normal Light', 'Low Light'])
ax2.legend()

# Add count labels
for bar in bars1:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
lighting_path = os.path.join(CONFIG['figures_dir'], 'context_lighting_analysis.png')
plt.savefig(lighting_path, dpi=150)
plt.close()
print(f"Lighting analysis saved to: {lighting_path}")

# ============================================
# Summary statistics
# ============================================
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)

print("\n### Crowd Density ###")
print(f"UBI - Mean people: {np.mean(ubi_avg_people):.2f} ± {np.std(ubi_avg_people):.2f}")
print(f"CCTV - Mean people: {np.mean(cctv_avg_people):.2f} ± {np.std(cctv_avg_people):.2f}")

print("\n### Motion Intensity ###")
print(f"UBI - Mean motion: {np.mean(ubi_motion):.2f} ± {np.std(ubi_motion):.2f}")
print(f"CCTV - Mean motion: {np.mean(cctv_motion):.2f} ± {np.std(cctv_motion):.2f}")

print("\n### Lighting ###")
print(f"UBI - Mean brightness: {np.mean(ubi_brightness):.2f}")
print(f"UBI - Low-light videos: {ubi_low_light}/{len(ubi_features)} ({ubi_low_light/len(ubi_features)*100:.1f}%)")
print(f"CCTV - Mean brightness: {np.mean(cctv_brightness):.2f}")
print(f"CCTV - Low-light videos: {cctv_low_light}/{len(cctv_features)} ({cctv_low_light/len(cctv_features)*100:.1f}%)")

# ============================================
# Save summary report
# ============================================
report_path = os.path.join(CONFIG['results_dir'], 'context_features_report.md')
with open(report_path, 'w') as f:
    f.write("# Context Features Extraction Report\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    
    f.write("## Overview\n\n")
    f.write("Extracted contextual features for violence detection:\n")
    f.write("1. **Crowd density** - number of people detected (HOG + SVM)\n")
    f.write("2. **Motion intensity** - optical flow statistics\n")
    f.write("3. **Lighting features** - brightness, contrast, low-light detection\n\n")
    
    f.write("---\n\n")
    f.write("## Dataset Coverage\n\n")
    f.write("| Dataset | Videos Processed | Fight | Normal |\n")
    f.write("|---------|------------------|-------|--------|\n")
    f.write(f"| UBI-Fights | {len(ubi_features)} | {sum(ubi_labels)} | {len(ubi_labels) - sum(ubi_labels)} |\n")
    f.write(f"| CCTV-Fights | {len(cctv_features)} | {sum(cctv_labels)} | {len(cctv_labels) - sum(cctv_labels)} |\n")
    
    f.write("\n---\n\n")
    f.write("## Crowd Density Statistics\n\n")
    f.write("| Dataset | Mean #People | Std | Max |\n")
    f.write("|---------|--------------|-----|-----|\n")
    f.write(f"| UBI | {np.mean(ubi_avg_people):.2f} | {np.std(ubi_avg_people):.2f} | {np.max(ubi_avg_people):.0f} |\n")
    f.write(f"| CCTV | {np.mean(cctv_avg_people):.2f} | {np.std(cctv_avg_people):.2f} | {np.max(cctv_avg_people):.0f} |\n")
    
    f.write("\n### By Class\n\n")
    f.write("| Class | Mean #People | Count |\n")
    f.write("|-------|--------------|-------|\n")
    if all_fight:
        f.write(f"| Fight | {np.mean(all_fight):.2f} | {len(all_fight)} |\n")
    if all_normal:
        f.write(f"| Normal | {np.mean(all_normal):.2f} | {len(all_normal)} |\n")
    
    f.write("\n---\n\n")
    f.write("## Motion Intensity Statistics\n\n")
    f.write("| Dataset | Mean Motion | Std |\n")
    f.write("|---------|-------------|-----|\n")
    f.write(f"| UBI | {np.mean(ubi_motion):.2f} | {np.std(ubi_motion):.2f} |\n")
    f.write(f"| CCTV | {np.mean(cctv_motion):.2f} | {np.std(cctv_motion):.2f} |\n")
    
    f.write("\n---\n\n")
    f.write("## Lighting Analysis\n\n")
    f.write("| Dataset | Mean Brightness | Low-Light Videos | Percentage |\n")
    f.write("|---------|-----------------|------------------|------------|\n")
    f.write(f"| UBI | {np.mean(ubi_brightness):.2f} | {ubi_low_light} | {ubi_low_light/len(ubi_features)*100:.1f}% |\n")
    f.write(f"| CCTV | {np.mean(cctv_brightness):.2f} | {cctv_low_light} | {cctv_low_light/len(cctv_features)*100:.1f}% |\n")
    
    f.write("\n---\n\n")
    f.write("## Visualizations\n\n")
    f.write("- Crowd density histogram: `figures/context_crowd_density_histogram.png`\n")
    f.write("- Motion vs crowd scatter: `figures/context_motion_vs_crowd.png`\n")
    f.write("- Lighting analysis: `figures/context_lighting_analysis.png`\n")
    
    f.write("\n---\n\n")
    f.write("## Feature Files\n\n")
    f.write(f"- UBI features: `context_features/ubi_context_features.json`\n")
    f.write(f"- CCTV features: `context_features/cctv_context_features.json`\n")
    
    f.write("\n---\n\n")
    f.write("## Notes\n\n")
    f.write("- Person detection uses OpenCV HOG + SVM (built-in, no external model required)\n")
    f.write("- Low-light threshold: brightness < 80 or dark pixel ratio > 50%\n")
    f.write("- Features are computed on 16 sampled frames per video\n")
    f.write("- These features will be integrated with the baseline model in Block 7\n")

print(f"\nReport saved to: {report_path}")

print("\n" + "=" * 60)
print("Context feature extraction complete!")
print("=" * 60)
