"""
Training script for CCTV-Fights Dataset

Uses ViT as frozen feature extractor (no NSL due to JAX2TF gradient limitations).
Memory-efficient with lazy loading and frame sampling.

Usage:
    cd /Volumes/KINGSTON/siena_finals/dip
    ./venv/bin/python scripts/train_CCTV_simple.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================
# Configuration
# ============================================
CONFIG = {
    'model_path': '/Volumes/KINGSTON/siena_finals/dip/models/HubModels/vit_s16_fe_1',
    'checkpoint_dir': '/Volumes/KINGSTON/siena_finals/dip/checkpoints/cctv_simple',
    'results_dir': '/Volumes/KINGSTON/siena_finals/dip/results',
    'batch_size': 16,
    'epochs': 15,
    'learning_rate': 0.001,
    'frames_per_video': 32,
    'vit_trainable': False,  # Freeze ViT, only train Dense layers
}

# Create directories
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['results_dir'], exist_ok=True)

print("=" * 60)
print("CCTV-Fights Training (Simple - No NSL)")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Epochs: {CONFIG['epochs']}")
print(f"ViT trainable: {CONFIG['vit_trainable']}")
print()

# ============================================
# Import data module
# ============================================
from Data_CCTV_optimized import (
    train_videos, validation_videos, test_videos,
    generatorTrainData, generatorValidationData, generatorTestData,
    width, height, channels,
    train_frames_per_epoch, validation_frames_per_epoch, test_frames_per_epoch,
    FRAMES_PER_VIDEO
)

# ============================================
# Calculate steps
# ============================================
steps_per_epoch = train_frames_per_epoch // CONFIG['batch_size']
validation_steps = validation_frames_per_epoch // CONFIG['batch_size']

print(f"\nDataset info:")
print(f"  Training videos: {len(train_videos)}")
print(f"  Validation videos: {len(validation_videos)}")
print(f"  Test videos: {len(test_videos)}")
print(f"  Frames per video: {FRAMES_PER_VIDEO}")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Validation steps: {validation_steps}")
print()

# ============================================
# Load ViT Model
# ============================================
print("Loading ViT model...")
print(f"Model path: {CONFIG['model_path']}")

# ============================================
# Build Model
# ============================================
print("\nBuilding model...")

# Create ViT feature extractor layer - use string path for serialization support
vit_layer = hub.KerasLayer(
    CONFIG['model_path'],  # Use string path, not loaded object
    trainable=CONFIG['vit_trainable'],
    name='vit_feature_extractor'
)
print("ViT layer created successfully!")

# Build classification model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(width, height, channels), name='input'),
    vit_layer,
    tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(2, activation='softmax', name='output')
], name='cctv_violence_classifier')

model.summary()

# ============================================
# Compile Model
# ============================================
print("\nCompiling model...")

# Use legacy Adam optimizer for M1 Mac compatibility
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=CONFIG['learning_rate'])

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# Callbacks
# ============================================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CONFIG['checkpoint_dir'], 'best_weights.weights.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,  # Save only weights to avoid serialization issues
        mode='max',
        verbose=1,
        save_format='h5'  # Explicit format
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================
# Training
# ============================================
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

start_time = datetime.now()

history = model.fit(
    generatorTrainData(batch_size_train=CONFIG['batch_size'], frames_per_video=CONFIG['frames_per_video']),
    epochs=CONFIG['epochs'],
    steps_per_epoch=steps_per_epoch,
    validation_data=generatorValidationData(batch_size=CONFIG['batch_size'], frames_per_video=CONFIG['frames_per_video']),
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

training_time = datetime.now() - start_time
print(f"\nTraining completed in: {training_time}")

# ============================================
# Evaluation
# ============================================
print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)

test_steps = test_frames_per_epoch // CONFIG['batch_size']
test_results = model.evaluate(
    generatorTestData(batch_size_test=CONFIG['batch_size'], frames_per_video=CONFIG['frames_per_video']),
    steps=test_steps,
    verbose=1
)

print(f"\nTest Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")

# ============================================
# Detailed Evaluation (F1, Confusion Matrix)
# ============================================
print("\n" + "=" * 60)
print("Collecting predictions for detailed metrics...")
print("=" * 60)

# Collect predictions
all_y_true = []
all_y_pred = []

test_gen = generatorTestData(batch_size_test=CONFIG['batch_size'], frames_per_video=CONFIG['frames_per_video'])
for _ in range(test_steps):
    x_batch, y_batch = next(test_gen)
    preds = model.predict(x_batch, verbose=0)
    all_y_true.extend(np.argmax(y_batch, axis=1))
    all_y_pred.extend(np.argmax(preds, axis=1))

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Compute metrics
accuracy = np.mean(all_y_true == all_y_pred)
f1_fight = f1_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)
f1_macro = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
f1_weighted = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)

print(f"\n--- Detailed Metrics ---")
print(f"Accuracy:           {accuracy*100:.2f}%")
print(f"F1-score (Fight):   {f1_fight*100:.2f}%")
print(f"F1-score (Macro):   {f1_macro*100:.2f}%")
print(f"F1-score (Weighted):{f1_weighted*100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(all_y_true, all_y_pred, 
                          target_names=['Non-Fight', 'Fight'],
                          digits=4))

# Confusion Matrix
cm = confusion_matrix(all_y_true, all_y_pred)
print("\n--- Confusion Matrix ---")
print(f"                 Predicted")
print(f"                 Non-Fight  Fight")
print(f"Actual Non-Fight    {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"Actual Fight        {cm[1,0]:5d}   {cm[1,1]:5d}")

# Plot Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Fight', 'Fight'],
            yticklabels=['Non-Fight', 'Fight'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix (Counts)')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=['Non-Fight', 'Fight'],
            yticklabels=['Non-Fight', 'Fight'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix (Normalized)')

plt.suptitle('CCTV-Fights In-Dataset Evaluation', fontsize=14, fontweight='bold')
plt.tight_layout()

cm_path = os.path.join(CONFIG['results_dir'], 'figures', 'CCTV_confusion_matrix.png')
os.makedirs(os.path.dirname(cm_path), exist_ok=True)
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix saved to: {cm_path}")
plt.close()

# ============================================
# Save Results
# ============================================
results_file = os.path.join(CONFIG['results_dir'], 'CCTV_baseline_results.md')
with open(results_file, 'w') as f:
    f.write("# CCTV-Fights Baseline Results (In-Dataset)\n\n")
    f.write(f"**Дата эксперимента:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"**Модель:** ViT (frozen) + Dense layers\n")
    f.write(f"**Метод:** Supervised Learning (без NSL из-за ограничений градиентов)\n")
    f.write(f"**Random Seed:** {RANDOM_SEED}\n\n")
    f.write("---\n\n")
    
    f.write("## Конфигурация\n\n")
    f.write("| Параметр | Значение |\n")
    f.write("|----------|----------|\n")
    f.write(f"| Backbone | ViT-S/16 (frozen) |\n")
    f.write(f"| Input | Optical Flow (224×224×3) |\n")
    f.write(f"| Batch size | {CONFIG['batch_size']} |\n")
    f.write(f"| Epochs | {CONFIG['epochs']} |\n")
    f.write(f"| Learning rate | {CONFIG['learning_rate']} |\n")
    f.write(f"| Frames per video | {CONFIG['frames_per_video']} |\n")
    f.write("| Optimizer | Adam (legacy) |\n")
    f.write("| Early stopping | patience=5 |\n\n")
    f.write("---\n\n")
    
    f.write("## Данные\n\n")
    f.write("| Метрика | Значение |\n")
    f.write("|---------|----------|\n")
    f.write(f"| Training videos | {len(train_videos)} |\n")
    f.write(f"| Validation videos | {len(validation_videos)} |\n")
    f.write(f"| Test videos | {len(test_videos)} |\n")
    f.write(f"| Frames per video | {FRAMES_PER_VIDEO} |\n")
    f.write(f"| Total available | 500 (из 1000 в groundtruth) |\n\n")
    f.write("---\n\n")
    
    f.write("## Результаты обучения\n\n")
    f.write("| Эпоха | Train Loss | Train Acc | Val Loss | Val Acc |\n")
    f.write("|-------|------------|-----------|----------|----------|\n")
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    for i in range(len(history.history['loss'])):
        marker = " ⭐" if (i + 1) == best_epoch else ""
        f.write(f"| {i+1} | {history.history['loss'][i]:.4f} | {history.history['accuracy'][i]*100:.2f}% | {history.history['val_loss'][i]:.4f} | {history.history['val_accuracy'][i]*100:.2f}%{marker} |\n")
    f.write(f"\n**Best epoch:** {best_epoch}\n\n")
    f.write("---\n\n")
    
    f.write("## Финальные метрики (Frame-level)\n\n")
    f.write("| Метрика | Значение |\n")
    f.write("|---------|----------|\n")
    f.write(f"| **Accuracy** | **{accuracy*100:.2f}%** |\n")
    f.write(f"| **F1-score (Fight)** | **{f1_fight*100:.2f}%** |\n")
    f.write(f"| F1-score (Macro) | {f1_macro*100:.2f}% |\n")
    f.write(f"| F1-score (Weighted) | {f1_weighted*100:.2f}% |\n")
    f.write(f"| Test Loss | {test_results[0]:.4f} |\n")
    f.write(f"| Training Time | {training_time} |\n\n")
    f.write("---\n\n")
    
    f.write("## Confusion Matrix\n\n")
    f.write("|  | Predicted Non-Fight | Predicted Fight |\n")
    f.write("|--|---------------------|------------------|\n")
    f.write(f"| **Actual Non-Fight** | {cm[0,0]} ({cm_norm[0,0]*100:.1f}%) | {cm[0,1]} ({cm_norm[0,1]*100:.1f}%) |\n")
    f.write(f"| **Actual Fight** | {cm[1,0]} ({cm_norm[1,0]*100:.1f}%) | {cm[1,1]} ({cm_norm[1,1]*100:.1f}%) |\n\n")
    f.write("![Confusion Matrix](figures/CCTV_confusion_matrix.png)\n\n")
    f.write("---\n\n")
    
    f.write("## Артефакты\n\n")
    f.write(f"- **Checkpoint:** `{CONFIG['checkpoint_dir']}/best_weights.weights.h5`\n")
    f.write(f"- **Confusion Matrix:** `results/figures/CCTV_confusion_matrix.png`\n\n")
    f.write("---\n\n")
    
    f.write("## Примечания\n\n")
    f.write("- ViT заморожен (не fine-tuned)\n")
    f.write("- Без Neural Structured Learning (NSL) из-за ограничений JAX2TF\n")
    f.write("- Frame-level классификация с использованием временных аннотаций\n")
    f.write("- Сбалансированное семплирование fight/normal кадров из каждого видео\n")

print(f"\nResults saved to: {results_file}")

# ============================================
# Save final model weights
# ============================================
final_weights_path = os.path.join(CONFIG['checkpoint_dir'], 'final_weights.weights.h5')
# Remove existing file to avoid h5py "name already exists" error
if os.path.exists(final_weights_path):
    os.remove(final_weights_path)
# Also remove old .h5 version if it exists
old_weights_path = os.path.join(CONFIG['checkpoint_dir'], 'final_weights.h5')
if os.path.exists(old_weights_path):
    os.remove(old_weights_path)
model.save_weights(final_weights_path)
print(f"Final weights saved to: {final_weights_path}")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
