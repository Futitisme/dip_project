"""
SIMPLE Training script for UBI-Fights dataset (WITHOUT NSL)
Uses ViT as frozen feature extractor + trainable Dense layers

The original ViT model from Kaggle doesn't support gradients with NSL,
so we use standard supervised learning here.

Usage:
    cd /Volumes/KINGSTON/siena_finals/dip
    ./venv/bin/python scripts/train_UBI_simple.py
"""
import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import tensorflow_hub as hub
import tensorflow as tf
import datetime
import numpy as np

# Import OPTIMIZED data loader
from Data_UBI_optimized import (
    train_videos, test_videos,
    generatorTrainData, generatorTestData,
    width, height, channels,
    train_frames_per_epoch, test_frames_per_epoch
)

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Model
    'vit_model_path': 'models/HubModels/vit_s16_fe_1',
    'vit_trainable': False,  # Freeze ViT, only train Dense layers
    
    # Training
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.001,
    
    # Paths
    'log_dir': 'results/logs/UBI/fit/',
    'checkpoint_dir': 'results/logs/UBI/checkpoint/',
}

# ============================================
# SETUP
# ============================================
print("\n" + "="*60)
print("UBI-Fights Training (Simple - No NSL)")
print("="*60)

# Check available memory
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"\nSystem memory: {mem.total/1e9:.1f} GB total, {mem.available/1e9:.1f} GB available")
except:
    pass

print(f"\nConfiguration:")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  ViT trainable: {CONFIG['vit_trainable']}")

# Create output directories
os.makedirs(CONFIG['log_dir'], exist_ok=True)
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# ============================================
# BUILD MODEL
# ============================================
print("\nLoading ViT model...")
loaded_model = hub.load(CONFIG['vit_model_path'])
print("✅ ViT model loaded")

print("\nBuilding classification model...")

# Use ViT as feature extractor (frozen)
vit_layer = hub.KerasLayer(
    loaded_model, 
    trainable=CONFIG['vit_trainable'],
    name='vit_feature_extractor'
)

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
# COMPILE MODEL
# ============================================
print("\nCompiling model...")

# Use legacy optimizer for M1 Mac
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=CONFIG['learning_rate'])

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# CALLBACKS
# ============================================
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard
log_dir = CONFIG['log_dir'] + timestamp
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=0,
    write_graph=False
)

# Model checkpoint
checkpoint_path = CONFIG['checkpoint_dir'] + timestamp + '/weights'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy',
    verbose=1
)

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [tensorboard_callback, cp_callback, early_stop, lr_scheduler]

# ============================================
# DATA WRAPPER (convert dict to tuple)
# ============================================
def data_generator_wrapper(generator_func, batch_size):
    """Wrap generator to output (x, y) tuples instead of dict"""
    gen = generator_func(batch_size)
    while True:
        batch = next(gen)
        yield batch['feature'], batch['label']

# ============================================
# CALCULATE STEPS
# ============================================
steps_per_epoch = train_frames_per_epoch // CONFIG['batch_size']
validation_steps = max(1, test_frames_per_epoch // CONFIG['batch_size'])

print(f"\nTraining setup:")
print(f"  Train videos: {len(train_videos)}")
print(f"  Test videos: {len(test_videos)}")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Validation steps: {validation_steps}")

# ============================================
# TRAINING
# ============================================
print("\n" + "="*60)
print("Starting training...")
print("="*60 + "\n")

start_time_train = time.time()

try:
    history = model.fit(
        data_generator_wrapper(generatorTrainData, CONFIG['batch_size']),
        epochs=CONFIG['epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_data=data_generator_wrapper(generatorTestData, CONFIG['batch_size']),
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time_train
    print(f"\n✅ Training completed!")
    print(f"Total training time: {training_time/60:.2f} minutes")
    
    # Plot training history
    print("\nTraining History:")
    print(f"  Best val_accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"  Final train_accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    
except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted by user")
    training_time = time.time() - start_time_train
    print(f"Training time before interrupt: {training_time/60:.2f} minutes")

except Exception as e:
    print(f"\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# EVALUATION
# ============================================
print("\n" + "="*60)
print("Final evaluation on test set...")
print("="*60)

try:
    start_time_test = time.time()
    results = model.evaluate(
        data_generator_wrapper(generatorTestData, CONFIG['batch_size']),
        steps=validation_steps
    )
    
    inference_time = time.time() - start_time_test
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Test loss: {results[0]:.4f}")
    print(f"Test accuracy: {results[1]*100:.2f}%")
    print(f"\nModel saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={CONFIG['log_dir']}")
    
except Exception as e:
    print(f"Evaluation error: {e}")
