"""
Cross-dataset Evaluation: CCTV → UBI

Loads model trained on CCTV-Fights and evaluates on UBI-Fights test set.
This demonstrates domain shift / generalization capabilities (symmetric to UBI→CCTV).

Usage:
    cd /Volumes/KINGSTON/siena_finals/dip
    ./venv/bin/python scripts/eval_cross_dataset_CCTV_to_UBI.py
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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# Configuration
# ============================================
CONFIG = {
    'model_path': '/Volumes/KINGSTON/siena_finals/dip/models/HubModels/vit_s16_fe_1',
    # CCTV model checkpoint (trained on CCTV-Fights)
    'cctv_weights_path': '/Volumes/KINGSTON/siena_finals/dip/checkpoints/cctv_simple/best_weights.weights.h5',
    'results_dir': '/Volumes/KINGSTON/siena_finals/dip/results',
    'figures_dir': '/Volumes/KINGSTON/siena_finals/dip/results/figures',
    'batch_size': 16,
    'num_frames_per_video': 8,  # Same as UBI training
}

# Create directories
os.makedirs(CONFIG['results_dir'], exist_ok=True)
os.makedirs(CONFIG['figures_dir'], exist_ok=True)

print("=" * 60)
print("Cross-Dataset Evaluation: CCTV → UBI")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print()

# ============================================
# Import UBI data module
# ============================================
from Data_UBI_optimized import (
    test_videos, get_video_path,
    sample_frames_from_video,
    width, height, channels
)

print(f"UBI Test videos: {len(test_videos)}")

# ============================================
# Build Model (same architecture as CCTV training)
# ============================================
print("\nBuilding model...")

# Create ViT feature extractor layer
vit_layer = hub.KerasLayer(
    CONFIG['model_path'],
    trainable=False,
    name='vit_feature_extractor'
)
print("ViT layer created successfully!")

# Build classification model (same as CCTV training)
# Note: CCTV model uses 'cctv_violence_classifier' name but architecture is same
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
# Load CCTV-trained weights
# ============================================
print("\nLoading CCTV-trained weights...")
print(f"Weights path: {CONFIG['cctv_weights_path']}")

try:
    model.load_weights(CONFIG['cctv_weights_path'])
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
# Collect predictions on UBI test set
# ============================================
print("\n" + "=" * 60)
print("Evaluating on UBI test set...")
print("=" * 60)

all_predictions = []
all_probabilities = []
all_labels = []

start_time = datetime.now()

for i, video_name in enumerate(test_videos):
    if (i + 1) % 10 == 0:
        print(f"Processing video {i+1}/{len(test_videos)}: {video_name}")
    
    # Sample frames from video (using UBI's sample function)
    frames, labels = sample_frames_from_video(
        video_name, 
        num_frames=CONFIG['num_frames_per_video'],
        random_start=False  # Deterministic for evaluation
    )
    
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
f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
f1_binary = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
cm = confusion_matrix(all_labels, all_predictions)

print(f"\nAccuracy: {accuracy*100:.2f}%")
print(f"F1-Score (weighted): {f1*100:.2f}%")
print(f"F1-Score (binary): {f1_binary*100:.2f}%")
print(f"Precision (weighted): {precision*100:.2f}%")
print(f"Recall (weighted): {recall*100:.2f}%")

print("\nConfusion Matrix:")
if cm.shape == (2, 2):
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
else:
    print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=['Normal', 'Fight'], zero_division=0))

# ============================================
# Plot Score Distributions
# ============================================
print("\nGenerating visualizations...")

# Separate scores by class
fight_scores = all_probabilities[all_labels == 1]
normal_scores = all_probabilities[all_labels == 0]

# 1. Score distribution histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1 = axes[0]
if len(normal_scores) > 0:
    ax1.hist(normal_scores, bins=30, alpha=0.7, label=f'Normal (n={len(normal_scores)})', color='blue')
if len(fight_scores) > 0:
    ax1.hist(fight_scores, bins=30, alpha=0.7, label=f'Fight (n={len(fight_scores)})', color='red')
ax1.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
ax1.set_xlabel('Fight Probability Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Score Distribution: CCTV model → UBI test set')
ax1.legend()

# Boxplot
ax2 = axes[1]
box_data = []
box_labels = []
if len(normal_scores) > 0:
    box_data.append(normal_scores)
    box_labels.append('Normal')
if len(fight_scores) > 0:
    box_data.append(fight_scores)
    box_labels.append('Fight')

if len(box_data) > 0:
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral'][:len(box_data)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
ax2.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')
ax2.set_ylabel('Fight Probability Score')
ax2.set_title('Score Distribution by Class')
ax2.legend()

plt.tight_layout()
fig_path = os.path.join(CONFIG['figures_dir'], 'cross_dataset_CCTV_to_UBI_scores.png')
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
       title='Confusion Matrix: CCTV → UBI')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, f'{cm[i, j]}\n({cm[i,j]/cm.sum()*100:.1f}%)',
                      ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
cm_path = os.path.join(CONFIG['figures_dir'], 'cross_dataset_CCTV_to_UBI_confusion.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

# ============================================
# Reference results for comparison
# ============================================
# UBI in-dataset results
ubi_in_dataset = {
    'accuracy': 87.88,
    'dataset': 'UBI-Fights'
}

# CCTV in-dataset results (updated)
cctv_in_dataset = {
    'accuracy': 62.17,
    'dataset': 'CCTV-Fights'
}

# UBI → CCTV cross-dataset results (updated)
ubi_to_cctv = {
    'accuracy': 49.49,
    'f1_weighted': 32.77
}

# ============================================
# Save Results
# ============================================
results_file = os.path.join(CONFIG['results_dir'], 'cross_dataset_CCTV_to_UBI_results.md')

with open(results_file, 'w') as f:
    f.write("# Cross-Dataset Evaluation: CCTV → UBI\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("## Overview\n\n")
    f.write("Model trained on **CCTV-Fights** dataset, evaluated on **UBI-Fights** test set.\n")
    f.write("This experiment measures domain generalization / domain shift (symmetric to UBI→CCTV).\n\n")
    
    f.write("---\n\n")
    f.write("## Configuration\n\n")
    f.write("| Parameter | Value |\n")
    f.write("|-----------|-------|\n")
    f.write(f"| Source Dataset | CCTV-Fights |\n")
    f.write(f"| Target Dataset | UBI-Fights (test set) |\n")
    f.write(f"| Model | ViT (frozen) + Dense layers |\n")
    f.write(f"| CCTV Weights | {CONFIG['cctv_weights_path']} |\n")
    f.write(f"| Frames per video | {CONFIG['num_frames_per_video']} |\n")
    f.write(f"| Total test videos | {len(test_videos)} |\n")
    f.write(f"| Total frames evaluated | {len(all_labels)} |\n")
    
    f.write("\n---\n\n")
    f.write("## Cross-Dataset Results (Train CCTV → Test UBI)\n\n")
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
    if cm.shape == (2, 2):
        f.write(f"| **Actual Normal** | {cm[0,0]} (TN) | {cm[0,1]} (FP) |\n")
        f.write(f"| **Actual Fight** | {cm[1,0]} (FN) | {cm[1,1]} (TP) |\n")
    
    f.write("\n---\n\n")
    f.write("## Comparison: In-Dataset vs Cross-Dataset\n\n")
    f.write("| Experiment | Train Dataset | Test Dataset | Accuracy | F1-Score |\n")
    f.write("|------------|---------------|--------------|----------|----------|\n")
    f.write(f"| UBI → UBI (in-dataset) | UBI-Fights | UBI-Fights | {ubi_in_dataset['accuracy']:.2f}% | - |\n")
    f.write(f"| CCTV → CCTV (in-dataset) | CCTV-Fights | CCTV-Fights | {cctv_in_dataset['accuracy']:.2f}% | - |\n")
    f.write(f"| UBI → CCTV (cross-dataset) | UBI-Fights | CCTV-Fights | {ubi_to_cctv['accuracy']:.2f}% | {ubi_to_cctv['f1_weighted']:.2f}% |\n")
    f.write(f"| **CCTV → UBI (cross-dataset)** | CCTV-Fights | UBI-Fights | **{accuracy*100:.2f}%** | **{f1*100:.2f}%** |\n")
    
    f.write("\n### Analysis\n\n")
    drop_from_cctv = cctv_in_dataset['accuracy'] - accuracy*100
    diff_from_ubi = accuracy*100 - ubi_in_dataset['accuracy']
    
    f.write(f"- **Domain shift impact:** CCTV→UBI accuracy ({accuracy*100:.2f}%) ")
    if drop_from_cctv > 0:
        f.write(f"drops by **{drop_from_cctv:.2f}%** compared to CCTV→CCTV ({cctv_in_dataset['accuracy']:.2f}%)\n")
    else:
        f.write(f"improves by **{-drop_from_cctv:.2f}%** compared to CCTV→CCTV ({cctv_in_dataset['accuracy']:.2f}%)\n")
    
    if diff_from_ubi >= 0:
        f.write(f"- CCTV model on UBI is **{diff_from_ubi:.2f}%** higher than UBI baseline ({ubi_in_dataset['accuracy']:.2f}%)\n")
    else:
        f.write(f"- CCTV model on UBI is **{-diff_from_ubi:.2f}%** lower than UBI baseline ({ubi_in_dataset['accuracy']:.2f}%)\n")
    
    f.write("\n**Cross-dataset comparison:**\n")
    f.write(f"- UBI → CCTV: {ubi_to_cctv['accuracy']:.2f}%\n")
    f.write(f"- CCTV → UBI: {accuracy*100:.2f}%\n")
    asymmetry = abs(ubi_to_cctv['accuracy'] - accuracy*100)
    f.write(f"- Asymmetry: {asymmetry:.2f}% difference between transfer directions\n")
    
    f.write("\n---\n\n")
    f.write("## Score Distribution Analysis\n\n")
    f.write("### Statistics\n\n")
    f.write("| Class | Mean Score | Std Score | Min | Max |\n")
    f.write("|-------|------------|-----------|-----|-----|\n")
    if len(normal_scores) > 0:
        f.write(f"| Normal | {normal_scores.mean():.4f} | {normal_scores.std():.4f} | {normal_scores.min():.4f} | {normal_scores.max():.4f} |\n")
    if len(fight_scores) > 0:
        f.write(f"| Fight | {fight_scores.mean():.4f} | {fight_scores.std():.4f} | {fight_scores.min():.4f} | {fight_scores.max():.4f} |\n")
    
    f.write("\n### Observations\n\n")
    if len(fight_scores) > 0 and len(normal_scores) > 0:
        f.write(f"- Mean fight score: {fight_scores.mean():.4f}, Mean normal score: {normal_scores.mean():.4f}\n")
        f.write(f"- Score separation: {fight_scores.mean() - normal_scores.mean():.4f}\n")
    
    f.write("\n---\n\n")
    f.write("## Visualizations\n\n")
    f.write(f"- Score distribution: `figures/cross_dataset_CCTV_to_UBI_scores.png`\n")
    f.write(f"- Confusion matrix: `figures/cross_dataset_CCTV_to_UBI_confusion.png`\n")
    
    f.write("\n---\n\n")
    f.write("## Conclusions\n\n")
    f.write("1. **Domain shift is bidirectional:** Both UBI→CCTV and CCTV→UBI show performance degradation\n")
    f.write("2. **Asymmetric transfer:** Transfer performance differs depending on direction\n")
    f.write("3. **Dataset characteristics matter:** Different video quality, violence patterns, and recording conditions affect generalization\n")

print(f"\nResults saved to: {results_file}")

# ============================================
# Save predictions for further analysis
# ============================================
predictions_file = os.path.join(CONFIG['results_dir'], 'cross_dataset_CCTV_to_UBI_predictions.npz')
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
