"""
OPTIMIZED Data loader for UBI-Fights dataset
Uses lazy loading to avoid memory issues on 16GB RAM systems

Key differences from original:
- Does NOT load all videos into memory at once
- Generates optical flow on-the-fly during training
- Uses TensorFlow Dataset API for efficient data pipeline
"""
import cv2
import csv
import numpy as np
import random
import tensorflow as tf
import os
from tqdm import tqdm

# ============================================
# PATHS
# ============================================
path_base = 'datasets/UBI_Fights/annotation/'
path_videos = 'datasets/UBI_Fights/videos/'
path_test_csv = 'datasets/UBI_Fights/test_videos.csv'

# ============================================
# IMAGE PARAMETERS
# ============================================
width = 224
height = 224
channels = 3

# ============================================
# LOAD VIDEO LISTS (only metadata, not actual videos)
# ============================================
print("Loading video metadata...")

# Filter out macOS hidden files
fight_videos = [v for v in os.listdir(path_videos + 'fight/') if v.endswith('.mp4') and not v.startswith('.')]
normal_videos = [v for v in os.listdir(path_videos + 'normal/') if v.endswith('.mp4') and not v.startswith('.')]

print(f"Found {len(fight_videos)} fight videos, {len(normal_videos)} normal videos")

# Load test videos list
test_videos = [l[0] + '.mp4' for l in list(csv.reader(open(path_test_csv)))]
all_videos = fight_videos + normal_videos
train_videos = [p for p in all_videos if p not in test_videos]

print(f"Train videos: {len(train_videos)}, Test videos: {len(test_videos)}")


def get_video_path(video_name):
    """Get full path for a video"""
    if video_name.startswith('F'):
        return path_videos + 'fight/' + video_name
    else:
        return path_videos + 'normal/' + video_name


def get_video_info(video_name):
    """Get video frame count and annotation without loading frames"""
    video_path = get_video_path(video_name)
    annotation_path = path_base + video_name.split('.')[0] + '.csv'
    
    # Get frame count
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Load annotations
    with open(annotation_path, 'r') as f:
        labels = [int(row[0]) for row in csv.reader(f)]
    
    return frame_count, labels


def compute_optical_flow_frame(prev_gray, curr_frame, hsv_template):
    """Compute optical flow for a single frame pair"""
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    hsv = hsv_template.copy()
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr, curr_gray


def sample_frames_from_video(video_name, num_frames=16, random_start=True):
    """
    Sample a fixed number of frames from a video with optical flow.
    This is much more memory efficient than loading all frames.
    """
    video_path = get_video_path(video_name)
    annotation_path = path_base + video_name.split('.')[0] + '.csv'
    
    # Load annotations
    with open(annotation_path, 'r') as f:
        all_labels = [int(row[0]) for row in csv.reader(f)]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= num_frames + 1:
        # Video too short, use all frames
        start_idx = 0
        indices = list(range(1, total_frames))
    else:
        if random_start:
            max_start = total_frames - num_frames - 1
            start_idx = random.randint(0, max(0, max_start))
        else:
            start_idx = 0
        indices = list(range(start_idx + 1, min(start_idx + num_frames + 1, total_frames)))
    
    # Read first frame for optical flow initialization
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return [], []
    
    prev_frame = cv2.resize(prev_frame, (width, height))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Template for HSV
    hsv_template = np.zeros((height, width, 3), dtype=np.uint8)
    hsv_template[..., 1] = 255
    
    frames = []
    labels = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        curr_frame = cv2.resize(curr_frame, (width, height))
        optical_flow, prev_gray = compute_optical_flow_frame(prev_gray, curr_frame, hsv_template)
        
        # Normalize
        optical_flow = (optical_flow.astype('float32') - 127.5) / 127.5
        
        frames.append(optical_flow)
        if idx - 1 < len(all_labels):
            labels.append(all_labels[idx - 1])
        else:
            labels.append(0)
    
    cap.release()
    return frames, labels


def create_frame_index(video_list):
    """Create index of all (video, frame_idx, label) tuples"""
    print("Creating frame index...")
    index = []
    
    for video_name in tqdm(video_list, desc="Indexing videos"):
        try:
            frame_count, labels = get_video_info(video_name)
            # We lose 1 frame due to optical flow
            for i in range(min(frame_count - 1, len(labels))):
                index.append((video_name, i, labels[i]))
        except Exception as e:
            print(f"Error indexing {video_name}: {e}")
            continue
    
    random.shuffle(index)
    return index


# Create frame indices (fast, just reads metadata)
print("\nCreating train frame index...")
train_index = create_frame_index(train_videos)
print(f"Total train frames indexed: {len(train_index)}")

print("\nCreating test frame index...")
test_index = create_frame_index(test_videos)
print(f"Total test frames indexed: {len(test_index)}")


class VideoFrameGenerator:
    """
    Memory-efficient generator that loads frames on demand.
    Caches recently used videos to speed up sequential access.
    """
    def __init__(self, frame_index, batch_size=16, cache_size=5):
        self.frame_index = frame_index
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.video_cache = {}  # {video_name: (frames, labels)}
        self.cache_order = []
    
    def get_frame(self, video_name, frame_idx, label):
        """Get a single frame, using cache if available"""
        if video_name not in self.video_cache:
            # Load video into cache
            frames, labels = self._load_video_segment(video_name, frame_idx)
            
            # Manage cache size
            if len(self.cache_order) >= self.cache_size:
                oldest = self.cache_order.pop(0)
                if oldest in self.video_cache:
                    del self.video_cache[oldest]
            
            self.video_cache[video_name] = (frames, labels, frame_idx)
            self.cache_order.append(video_name)
        
        cached_frames, cached_labels, start_idx = self.video_cache[video_name]
        local_idx = frame_idx - start_idx
        
        if 0 <= local_idx < len(cached_frames):
            return cached_frames[local_idx], label
        else:
            # Frame not in cache, reload
            frames, labels = self._load_video_segment(video_name, frame_idx)
            self.video_cache[video_name] = (frames, labels, frame_idx)
            if len(frames) > 0:
                return frames[0], label
            return None, label
    
    def _load_video_segment(self, video_name, start_frame, segment_size=32):
        """Load a segment of frames around the requested frame"""
        video_path = get_video_path(video_name)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []
        
        # Start a bit before to get optical flow context
        actual_start = max(0, start_frame - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
        
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return [], []
        
        prev_frame = cv2.resize(prev_frame, (width, height))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        hsv_template = np.zeros((height, width, 3), dtype=np.uint8)
        hsv_template[..., 1] = 255
        
        frames = []
        for _ in range(segment_size):
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            curr_frame = cv2.resize(curr_frame, (width, height))
            optical_flow, prev_gray = compute_optical_flow_frame(prev_gray, curr_frame, hsv_template)
            optical_flow = (optical_flow.astype('float32') - 127.5) / 127.5
            frames.append(optical_flow)
        
        cap.release()
        return frames, []
    
    def __call__(self):
        """Generator function for tf.data.Dataset"""
        indices = list(range(len(self.frame_index)))
        random.shuffle(indices)
        
        for idx in indices:
            video_name, frame_idx, label = self.frame_index[idx]
            frame, _ = self.get_frame(video_name, frame_idx, label)
            if frame is not None:
                yield frame, np.float32(label)


def create_dataset(frame_index, batch_size=16, shuffle_buffer=1000):
    """Create tf.data.Dataset with optimized pipeline"""
    generator = VideoFrameGenerator(frame_index, batch_size)
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(height, width, channels), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: {'feature': x, 'label': y})
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# Legacy generators for compatibility (uses sampling approach)
def generatorTrainData(batch_size_train=16):
    """Memory-efficient training generator using video sampling"""
    while True:
        # Shuffle videos each epoch
        videos = train_videos.copy()
        random.shuffle(videos)
        
        batch_x = []
        batch_y = []
        
        for video_name in videos:
            # Sample frames from this video
            frames, labels = sample_frames_from_video(video_name, num_frames=8, random_start=True)
            
            for frame, label in zip(frames, labels):
                batch_x.append(frame)
                batch_y.append(label)
                
                if len(batch_x) >= batch_size_train:
                    x = np.array(batch_x[:batch_size_train]).astype('float32')
                    y = np.array(batch_y[:batch_size_train]).astype('float32')
                    
                    batch_x = batch_x[batch_size_train:]
                    batch_y = batch_y[batch_size_train:]
                    
                    yield {'feature': tf.convert_to_tensor(x), 'label': tf.convert_to_tensor(y)}


def generatorTestData(batch_size_test=16):
    """Memory-efficient test generator using video sampling"""
    while True:
        videos = test_videos.copy()
        random.shuffle(videos)
        
        batch_x = []
        batch_y = []
        
        for video_name in videos:
            frames, labels = sample_frames_from_video(video_name, num_frames=8, random_start=False)
            
            for frame, label in zip(frames, labels):
                batch_x.append(frame)
                batch_y.append(label)
                
                if len(batch_x) >= batch_size_test:
                    x = np.array(batch_x[:batch_size_test]).astype('float32')
                    y = np.array(batch_y[:batch_size_test]).astype('float32')
                    
                    batch_x = batch_x[batch_size_test:]
                    batch_y = batch_y[batch_size_test:]
                    
                    yield {'feature': tf.convert_to_tensor(x), 'label': tf.convert_to_tensor(y)}


# Calculate steps
train_frames_per_epoch = len(train_videos) * 8  # 8 frames sampled per video
test_frames_per_epoch = len(test_videos) * 8

print(f"\nOptimized data loader ready!")
print(f"Train: ~{train_frames_per_epoch} frames/epoch ({len(train_videos)} videos × 8 frames)")
print(f"Test: ~{test_frames_per_epoch} frames/epoch ({len(test_videos)} videos × 8 frames)")


if __name__ == "__main__":
    print("\nTesting optimized generator...")
    gen = generatorTrainData(batch_size_train=16)
    batch = next(gen)
    print(f"Batch shape: features={batch['feature'].shape}, labels={batch['label'].shape}")
    print("✅ Optimized data loader test passed!")
