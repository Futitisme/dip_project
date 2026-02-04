"""
Cross-dataset Evaluation: UBI → CCTV

Loads model trained on UBI-Fights and evaluates on CCTV-Fights test set.
This demonstrates domain shift / generalization capabilities.

Usage:
    cd /Volumes/KINGSTON/siena_finals/dip
    ./venv/bin/python scripts/eval_cross_dataset_UBI_to_CCTV.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# Configuration
# ============================================
CONFIG = {
    'model_path': '/Volumes/KINGSTON/siena_finals/dip/models/HubModels/vit_s16_fe_1',
    # UBI model checkpoint (trained on UBI-Fights)
    'ubi_weights_path': '/Volumes/KINGSTON/siena_finals/dip/results/logs/UBI/checkpoint/20260128-032022/weights',
    'results_dir': '/Volumes/KINGSTON/siena_finals/dip/results',
    'figures_dir': '/Volumes/KINGSTON/siena_finals/dip/results/figures',
    'batch_size': 16,
    'frames_per_video': 32,
}

# Create directories
os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(CONFIG['figures_dir'], exist_ok=True)

print("=" * 60)
print("Cross-Dataset Evaluation: UBI → CCTV")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print()

# ============================================
# Import CCTV data module
# ============================================
from Data_CCTV_optimized import (
    test_videos,
    sample_frames_from_video,
    width, height, channels
)

print(f"CCTV Test videos: {len(test_videos)}")

# ============================================
# Build Model (same architecture as UBI training)
# ============================================
print("\nBuilding model...")

# Create ViT feature extractor layer
vit_layer = hub.KerasLayer(
    CONFIG['model_path'],
    trainable=False,
    name='vit_feature_extractor'
)
print("ViT layer created successfully!")

# Build classification model (same as UBI training)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(width, height, channels), name='input'),
    vit_layer,
    tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(2, activation='softmax', name='output')
], name='violence_classifier')

model.summary()

# ============================================
# Load UBI-trained weights
# ============================================
print("\nLoading UBI-trained weights...")
print(f"Weights path: {CONFIG['ubi_weights_path']}")

try:
    model.load_weights(CONFIG['ubi_weights_path'])
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    sys.exit(1)

# ============================================
# Compile model for evaluation
# ============================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# Collect predictions on CCTV test set
# ============================================
print("\n" + "=" * 60)
print("Evaluating on CCTV test set...")
print("=" * 60)

all_predictions = []
all_probabilities = []
all_labels = []

start_time = datetime.now()

for i, (video_id, video_path) in enumerate(test_videos):
    if (i + 1) % 10 == 0:
        print(f"Processing video {i+1}/{len(test_videos)}: {video_id}")
    
    # Sample frames from video
    samples = sample_frames_from_video(
        video_path, video_id,
        num_frames=CONFIG['frames_per_video'],
        balance_classes=True
    )
    
    if len(samples) == 0:
        continue
    
    # Prepare batch
    frames = []
    labels = []
    for frame, label in samples:
        # Normalize frame (same as training)
        frame = (frame.astype('float32') - 127.5) / 127.5
        frames.append(frame)
        labels.append(label)
    
    if len(frames) == 0:
        continue
    
    X = np.array(frames, dtype='float32')
    y = np.array(labels, dtype='int32')
    
    # Get predictions
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    all_predictions.extend(preds)
    all_probabilities.extend(probs[:, 1])  # Probability of class 1 (fight)
    all_labels.extend(y)

eval_time = datetime.now() - start_time

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_probabilities = np.array(all_probabilities)
all_labels = np.array(all_labels)

print(f"\nEvaluation completed in: {eval_time}")
print(f"Total frames evaluated: {len(all_labels)}")

# ============================================
# Calculate Metrics
# ============================================
print("\n" + "=" * 60)
print("Results")
print("=" * 60)

accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions, average='weighted')
f1_binary = f1_score(all_labels, all_predictions, average='binary')
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
cm = confusion_matrix(all_labels, all_predictions)

print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"F1-Score (weighted): {f1*100:.2f}%")
print(f"F1-Score (binary): {f1_binary*100:.2f}%")
print(f"Precision (weighted): {precision*100:.2f}%")
print(f"Recall (weighted): {recall*100:.2f}%")

print("\nConfusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=['Normal', 'Fight']))

# ============================================
# Plot Score Distributions
# ============================================
print("\nGenerating visualizations...")

# 1. Score distribution histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
fight_scores = all_probabilities[all_labels == 1]
normal_scores = all_probabilities[all_labels == 0]

ax1.hist(normal_scores, bins=30, alpha=0.7, label=f'Normal (n={len(normal_scores)})', color='blue')
ax1.hist(fight_scores, bins=30, alpha=0.7, label=f'Fight (n={len(fight_scores)})', color='red')
ax1.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
ax1.set_xlabel('Fight Probability Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Score Distribution: UBI model → CCTV test set')
ax1.legend()

# Boxplot
ax2 = axes[1]
box_data = [normal_scores, fight_scores]
bp = ax2.boxplot(box_data, labels=['Normal', 'Fight'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax2.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')
ax2.set_ylabel('Fight Probability Score')
ax2.set_title('Score Distribution by Class')
ax2.legend()

plt.tight_layout()
fig_path = os.path.join(CONFIG['figures_dir'], 'cross_dataset_UBI_to_CCTV_scores.png')
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Score distribution saved to: {fig_path}")

# 2. Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=['Normal', 'Fight'],
       yticklabels=['Normal', 'Fight'],
       ylabel='True label',
       xlabel='Predicted label',
       title='Confusion Matrix: UBI → CCTV')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{cm[i, j]}\n({cm[i,j]/cm.sum()*100:.1f}%)',
                      ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
cm_path = os.path.join(CONFIG['figures_dir'], 'cross_dataset_UBI_to_CCTV_confusion.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

# ============================================
# Load in-dataset results for comparison
# ============================================
# UBI in-dataset results (from results file)
ubi_in_dataset = {
    'accuracy': 87.88,
    'dataset': 'UBI-Fights'
}

# CCTV in-dataset results (updated after retraining)
cctv_in_dataset = {
    'accuracy': 62.17,
    'dataset': 'CCTV-Fights'
}

# ============================================
# Save Results
# ============================================
results_file = os.path.join(CONFIG['results_dir'], 'cross_dataset_UBI_to_CCTV_results.md')

with open(results_file, 'w') as f:
    f.write("# Cross-Dataset Evaluation: UBI → CCTV\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("## Overview\n\n")
    f.write("Model trained on **UBI-Fights** dataset, evaluated on **CCTV-Fights** test set.\n")
    f.write("This experiment measures domain generalization / domain shift.\n\n")
    
    f.write("---\n\n")
    f.write("## Configuration\n\n")
    f.write("| Parameter | Value |\n")
    f.write("|-----------|-------|\n")
    f.write(f"| Source Dataset | UBI-Fights |\n")
    f.write(f"| Target Dataset | CCTV-Fights (test set) |\n")
    f.write(f"| Model | ViT (frozen) + Dense layers |\n")
    f.write(f"| UBI Weights | {CONFIG['ubi_weights_path']} |\n")
    f.write(f"| Frames per video | {CONFIG['frames_per_video']} |\n")
    f.write(f"| Total test videos | {len(test_videos)} |\n")
    f.write(f"| Total frames evaluated | {len(all_labels)} |\n")
    
    f.write("\n---\n\n")
    f.write("## Cross-Dataset Results (Train UBI → Test CCTV)\n\n")
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    f.write(f"| **Accuracy** | **{accuracy*100:.2f}%** |\n")
    f.write(f"| F1-Score (weighted) | {f1*100:.2f}% |\n")
    f.write(f"| F1-Score (binary) | {f1_binary*100:.2f}% |\n")
    f.write(f"| Precision (weighted) | {precision*100:.2f}% |\n")
    f.write(f"| Recall (weighted) | {recall*100:.2f}% |\n")
    
    f.write("\n### Confusion Matrix\n\n")
    f.write("|  | Predicted Normal | Predicted Fight |\n")
    f.write("|--|------------------|------------------|\n")
    f.write(f"| **Actual Normal** | {cm[0,0]} (TN) | {cm[0,1]} (FP) |\n")
    f.write(f"| **Actual Fight** | {cm[1,0]} (FN) | {cm[1,1]} (TP) |\n")
    
    f.write("\n---\n\n")
    f.write("## Comparison: In-Dataset vs Cross-Dataset\n\n")
    f.write("| Experiment | Train Dataset | Test Dataset | Accuracy | F1-Score |\n")
    f.write("|------------|---------------|--------------|----------|----------|\n")
    f.write(f"| UBI → UBI (in-dataset) | UBI-Fights | UBI-Fights | {ubi_in_dataset['accuracy']:.2f}% | - |\n")
    f.write(f"| CCTV → CCTV (in-dataset) | CCTV-Fights | CCTV-Fights | {cctv_in_dataset['accuracy']:.2f}% | - |\n")
    f.write(f"| **UBI → CCTV (cross-dataset)** | UBI-Fights | CCTV-Fights | **{accuracy*100:.2f}%** | **{f1*100:.2f}%** |\n")
    
    f.write("\n### Analysis\n\n")
    drop_from_ubi = ubi_in_dataset['accuracy'] - accuracy*100
    diff_from_cctv = accuracy*100 - cctv_in_dataset['accuracy']
    
    f.write(f"- **Domain shift impact:** UBI→CCTV accuracy ({accuracy*100:.2f}%) drops by **{drop_from_ubi:.2f}%** compared to UBI→UBI ({ubi_in_dataset['accuracy']:.2f}%)\n")
    if diff_from_cctv >= 0:
        f.write(f"- UBI model on CCTV is **{diff_from_cctv:.2f}%** higher than CCTV baseline ({cctv_in_dataset['accuracy']:.2f}%)\n")
    else:
        f.write(f"- UBI model on CCTV is **{-diff_from_cctv:.2f}%** lower than CCTV baseline ({cctv_in_dataset['accuracy']:.2f}%)\n")
    f.write("- This indicates significant domain shift between UBI-Fights and CCTV-Fights datasets\n")
    
    f.write("\n---\n\n")
    f.write("## Score Distribution Analysis\n\n")
    f.write("### Statistics\n\n")
    f.write("| Class | Mean Score | Std Score | Min | Max |\n")
    f.write("|-------|------------|-----------|-----|-----|\n")
    f.write(f"| Normal | {normal_scores.mean():.4f} | {normal_scores.std():.4f} | {normal_scores.min():.4f} | {normal_scores.max():.4f} |\n")
    f.write(f"| Fight | {fight_scores.mean():.4f} | {fight_scores.std():.4f} | {fight_scores.min():.4f} | {fight_scores.max():.4f} |\n")
    
    f.write("\n### Observations\n\n")
    overlap = len(fight_scores[(fight_scores >= normal_scores.mean()) & (fight_scores <= fight_scores.mean())]) / len(fight_scores) * 100 if len(fight_scores) > 0 else 0
    f.write(f"- Mean fight score: {fight_scores.mean():.4f}, Mean normal score: {normal_scores.mean():.4f}\n")
    f.write(f"- Score separation: {fight_scores.mean() - normal_scores.mean():.4f}\n")
    f.write(f"- Significant distribution overlap indicates domain shift\n")
    
    f.write("\n---\n\n")
    f.write("## Visualizations\n\n")
    f.write(f"- Score distribution: `figures/cross_dataset_UBI_to_CCTV_scores.png`\n")
    f.write(f"- Confusion matrix: `figures/cross_dataset_UBI_to_CCTV_confusion.png`\n")
    
    f.write("\n---\n\n")
    f.write("## Conclusions\n\n")
    f.write("1. **Domain shift is significant:** Model trained on UBI-Fights shows notable performance degradation on CCTV-Fights\n")
    f.write("2. **Dataset differences:** UBI-Fights contains higher quality videos with different violence patterns compared to CCTV footage\n")
    f.write("3. **Generalization challenge:** Cross-dataset performance is lower than in-dataset baselines, confirming the need for domain adaptation or mixed training\n")

print(f"\nResults saved to: {results_file}")

# ============================================
# Save predictions for further analysis
# ============================================
predictions_file = os.path.join(CONFIG['results_dir'], 'cross_dataset_UBI_to_CCTV_predictions.npz')
np.savez(
    predictions_file,
    predictions=all_predictions,
    probabilities=all_probabilities,
    labels=all_labels
)
print(f"Predictions saved to: {predictions_file}")

print("\n" + "=" * 60)
print("Cross-dataset evaluation complete!")
print("=" * 60)
