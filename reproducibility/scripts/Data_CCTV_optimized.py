"""
Optimized CCTV-Fights Data Loader

Memory-efficient data loading with lazy loading and frame sampling.
Uses temporal annotations to label frames as fight/normal.
"""

import cv2
import json
import numpy as np
import os
import random
import tensorflow as tf

# ============================================
# Configuration
# ============================================
path_base = '/Volumes/KINGSTON/siena_finals/dip/datasets/CCTV_Fights/'
path_videos = path_base  # Videos are in subfolders like mpeg-401-500/

# Load ground truth
with open(path_base + 'groundtruth.json', 'r') as f:
    ground_truth_data = json.load(f)
data = ground_truth_data['database']

width = 224
height = 224
channels = 3

# ============================================
# Find available videos
# ============================================
def find_available_videos():
    """Find all available video files in the dataset folders."""
    available = {}
    for folder in os.listdir(path_base):
        folder_path = os.path.join(path_base, folder)
        if os.path.isdir(folder_path) and folder.startswith('mpeg-'):
            for f in os.listdir(folder_path):
                if f.endswith('.mpeg') and not f.startswith('._'):
                    video_id = f.replace('.mpeg', '')
                    available[video_id] = os.path.join(folder_path, f)
    return available

available_videos = find_available_videos()
print(f"Found {len(available_videos)} available videos")

# Split videos by subset (only using available videos)
train_videos = []
validation_videos = []
test_videos = []

for video_id, video_path in available_videos.items():
    if video_id in data:
        subset = data[video_id]['subset']
        if subset == 'training':
            train_videos.append((video_id, video_path))
        elif subset == 'validation':
            validation_videos.append((video_id, video_path))
        elif subset == 'testing':
            test_videos.append((video_id, video_path))

print(f"Training videos: {len(train_videos)}")
print(f"Validation videos: {len(validation_videos)}")
print(f"Testing videos: {len(test_videos)}")

# ============================================
# Frame-level annotation functions
# ============================================
def get_fight_frame_ranges(video_id):
    """Get frame ranges where fight occurs in a video."""
    if video_id not in data:
        return []
    
    frame_rate = data[video_id]['frame_rate']
    annotations = data[video_id].get('annotations', [])
    
    ranges = []
    for ann in annotations:
        segment = ann['segment']
        start_frame = int(frame_rate * segment[0])
        end_frame = int(frame_rate * segment[1])
        ranges.append((start_frame, end_frame))
    
    return ranges

def is_frame_fight(frame_idx, fight_ranges):
    """Check if a frame index falls within any fight range."""
    for start, end in fight_ranges:
        if start <= frame_idx < end:
            return True
    return False

# ============================================
# Optical flow computation
# ============================================
def compute_optical_flow(prev_gray, curr_gray, hsv_template):
    """Compute optical flow between two frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    hsv = hsv_template.copy()
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

# ============================================
# Lazy frame sampling
# ============================================
def sample_frames_from_video(video_path, video_id, num_frames=32, balance_classes=True):
    """
    Sample frames from a video with their fight/normal labels.
    
    Args:
        video_path: Path to video file
        video_id: Video ID for looking up annotations
        num_frames: Number of frames to sample
        balance_classes: If True, try to balance fight/normal frames
    
    Returns:
        List of (optical_flow_frame, label) tuples
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return []
    
    fight_ranges = get_fight_frame_ranges(video_id)
    
    # Determine which frames are fight vs normal
    fight_frames = []
    normal_frames = []
    for i in range(total_frames):
        if is_frame_fight(i, fight_ranges):
            fight_frames.append(i)
        else:
            normal_frames.append(i)
    
    # Sample frames
    if balance_classes and len(fight_frames) > 0 and len(normal_frames) > 0:
        # Balance: sample equal from fight and normal
        n_per_class = num_frames // 2
        sampled_fight = random.sample(fight_frames, min(n_per_class, len(fight_frames)))
        sampled_normal = random.sample(normal_frames, min(n_per_class, len(normal_frames)))
        frame_indices = sorted(sampled_fight + sampled_normal)
    else:
        # Random sampling
        frame_indices = sorted(random.sample(range(total_frames - 1), min(num_frames, total_frames - 1)))
    
    if len(frame_indices) == 0:
        cap.release()
        return []
    
    # Read and process frames
    samples = []
    hsv_template = np.zeros((height, width, 3), dtype=np.uint8)
    hsv_template[..., 1] = 255
    
    prev_frame = None
    prev_gray = None
    prev_idx = -2
    
    for target_idx in frame_indices:
        # Read frames up to target
        while cap.get(cv2.CAP_PROP_POS_FRAMES) <= target_idx:
            ret, frame = cap.read()
            if not ret:
                break
            
            curr_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            
            if curr_idx == target_idx or curr_idx == target_idx - 1:
                resized = cv2.resize(frame, (width, height))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                
                if curr_idx == target_idx - 1:
                    prev_frame = resized
                    prev_gray = gray
                    prev_idx = curr_idx
                elif curr_idx == target_idx:
                    if prev_idx == target_idx - 1 and prev_gray is not None:
                        # Compute optical flow
                        flow_frame = compute_optical_flow(prev_gray, gray, hsv_template)
                        label = 1 if is_frame_fight(target_idx, fight_ranges) else 0
                        samples.append((flow_frame, label))
                    
                    # Update prev for next iteration
                    prev_frame = resized
                    prev_gray = gray
                    prev_idx = curr_idx
    
    cap.release()
    return samples

# ============================================
# Data generators
# ============================================
def generatorTrainData(batch_size_train=16, frames_per_video=32):
    """Generator for training data with lazy loading."""
    while True:
        # Shuffle videos each epoch
        videos = train_videos.copy()
        random.shuffle(videos)
        
        batch_x = []
        batch_y = []
        
        for video_id, video_path in videos:
            # Sample frames from this video
            samples = sample_frames_from_video(
                video_path, video_id, 
                num_frames=frames_per_video,
                balance_classes=True
            )
            
            for frame, label in samples:
                # Normalize frame
                frame = (frame.astype('float32') - 127.5) / 127.5
                batch_x.append(frame)
                batch_y.append(label)
                
                if len(batch_x) >= batch_size_train:
                    x = np.array(batch_x, dtype='float32')
                    y = np.array(batch_y, dtype='float32')
                    
                    # One-hot encode labels
                    y_onehot = np.zeros((len(y), 2), dtype='float32')
                    y_onehot[np.arange(len(y)), y.astype(int)] = 1
                    
                    yield (x, y_onehot)
                    
                    batch_x = []
                    batch_y = []

def generatorValidationData(batch_size=16, frames_per_video=32):
    """Generator for validation data with lazy loading."""
    while True:
        videos = validation_videos.copy()
        random.shuffle(videos)
        
        batch_x = []
        batch_y = []
        
        for video_id, video_path in videos:
            samples = sample_frames_from_video(
                video_path, video_id,
                num_frames=frames_per_video,
                balance_classes=True
            )
            
            for frame, label in samples:
                frame = (frame.astype('float32') - 127.5) / 127.5
                batch_x.append(frame)
                batch_y.append(label)
                
                if len(batch_x) >= batch_size:
                    x = np.array(batch_x, dtype='float32')
                    y = np.array(batch_y, dtype='float32')
                    
                    y_onehot = np.zeros((len(y), 2), dtype='float32')
                    y_onehot[np.arange(len(y)), y.astype(int)] = 1
                    
                    yield (x, y_onehot)
                    
                    batch_x = []
                    batch_y = []

def generatorTestData(batch_size_test=16, frames_per_video=32):
    """Generator for test data with lazy loading."""
    while True:
        videos = test_videos.copy()
        random.shuffle(videos)
        
        batch_x = []
        batch_y = []
        
        for video_id, video_path in videos:
            samples = sample_frames_from_video(
                video_path, video_id,
                num_frames=frames_per_video,
                balance_classes=True
            )
            
            for frame, label in samples:
                frame = (frame.astype('float32') - 127.5) / 127.5
                batch_x.append(frame)
                batch_y.append(label)
                
                if len(batch_x) >= batch_size_test:
                    x = np.array(batch_x, dtype='float32')
                    y = np.array(batch_y, dtype='float32')
                    
                    y_onehot = np.zeros((len(y), 2), dtype='float32')
                    y_onehot[np.arange(len(y)), y.astype(int)] = 1
                    
                    yield (x, y_onehot)
                    
                    batch_x = []
                    batch_y = []

# ============================================
# Calculate steps per epoch
# ============================================
FRAMES_PER_VIDEO = 32

train_frames_per_epoch = len(train_videos) * FRAMES_PER_VIDEO
validation_frames_per_epoch = len(validation_videos) * FRAMES_PER_VIDEO
test_frames_per_epoch = len(test_videos) * FRAMES_PER_VIDEO

print(f"\nEstimated frames per epoch:")
print(f"  Training: {train_frames_per_epoch}")
print(f"  Validation: {validation_frames_per_epoch}")
print(f"  Testing: {test_frames_per_epoch}")
