"""
Data Utilities
Helper functions for data loading and preprocessing
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import os
from tqdm import tqdm

def load_image(image_path: str, grayscale: bool = False) -> Optional[np.ndarray]:
    """
    Load image from path
    
    Args:
        image_path: Path to image file
        grayscale: Convert to grayscale if True
    
    Returns:
        Image as numpy array or None if failed
    """
    if not os.path.exists(image_path):
        return None
    
    try:
        if grayscale:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(image_path)
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        size: Target size (width, height)
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image (0-255)
    
    Returns:
        Normalized image (0-1)
    """
    return image.astype(np.float32) / 255.0


def preprocess_eye_roi(
    eye_roi: np.ndarray, 
    target_size: Tuple[int, int] = (32, 32)
) -> np.ndarray:
    """
    Preprocess eye ROI for model input
    
    Args:
        eye_roi: Eye region image
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image ready for model (normalized, resized)
    """
    # Convert to grayscale if needed
    if len(eye_roi.shape) == 3:
        eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    
    # Resize
    eye_roi = resize_image(eye_roi, target_size)
    
    # Histogram equalization (improve contrast)
    eye_roi = cv2.equalizeHist(eye_roi)
    
    # Normalize
    eye_roi = normalize_image(eye_roi)
    
    # Add channel dimension: (32, 32) -> (32, 32, 1)
    eye_roi = np.expand_dims(eye_roi, axis=-1)
    
    return eye_roi


def preprocess_mouth_roi(
    mouth_roi: np.ndarray,
    target_size: Tuple[int, int] = (48, 48)
) -> np.ndarray:
    """
    Preprocess mouth ROI for model input
    
    Args:
        mouth_roi: Mouth region image
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed RGB image
    """
    # Ensure RGB
    if len(mouth_roi.shape) == 2:
        mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_GRAY2RGB)
    elif mouth_roi.shape[2] == 4:
        mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGRA2RGB)
    else:
        mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2RGB)
    
    # Resize
    mouth_roi = resize_image(mouth_roi, target_size)
    
    # Normalize
    mouth_roi = normalize_image(mouth_roi)
    
    return mouth_roi


def count_images_in_directory(directory: str) -> dict:
    """
    Count images in directory by class
    
    Args:
        directory: Root directory containing class folders
    
    Returns:
        Dictionary with counts per class
    """
    counts = {}
    
    if not os.path.exists(directory):
        return counts
    
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Count image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        count = sum(
            1 for f in os.listdir(class_path)
            if os.path.splitext(f.lower())[1] in image_extensions
        )
        
        counts[class_name] = count
    
    return counts


def get_image_paths(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image paths in directory recursively
    
    Args:
        directory: Root directory
        extensions: List of valid extensions (default: common image formats)
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    extensions = [ext.lower() for ext in extensions]
    image_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file.lower())[1] in extensions:
                image_paths.append(os.path.join(root, file))
    
    return image_paths


def verify_image_integrity(image_path: str) -> Tuple[bool, str]:
    """
    Verify if image can be loaded and is valid
    
    Args:
        image_path: Path to image
    
    Returns:
        (is_valid, error_message)
    """
    try:
        img = cv2.imread(image_path)
        
        if img is None:
            return False, "Failed to load (corrupted or invalid format)"
        
        h, w = img.shape[:2]
        
        if h < 10 or w < 10:
            return False, f"Image too small ({w}x{h})"
        
        if h > 10000 or w > 10000:
            return False, f"Image too large ({w}x{h})"
        
        return True, "OK"
    
    except Exception as e:
        return False, str(e)


def batch_verify_images(directory: str, show_progress: bool = True) -> dict:
    """
    Verify all images in directory
    
    Args:
        directory: Directory containing images
        show_progress: Show progress bar
    
    Returns:
        Dictionary with verification results
    """
    image_paths = get_image_paths(directory)
    
    results = {
        'total': len(image_paths),
        'valid': 0,
        'invalid': 0,
        'errors': []
    }
    
    iterator = tqdm(image_paths, desc="Verifying images") if show_progress else image_paths
    
    for img_path in iterator:
        is_valid, error = verify_image_integrity(img_path)
        
        if is_valid:
            results['valid'] += 1
        else:
            results['invalid'] += 1
            results['errors'].append({
                'path': img_path,
                'error': error
            })
    
    return results


if __name__ == "__main__":
    # Test utilities
    print("="*70)
    print("DATA UTILITIES TEST")
    print("="*70)
    print()
    
    # Test image counting
    print("Testing count_images_in_directory...")
    counts = count_images_in_directory("data/raw/eyes/train")
    print(f"Image counts: {counts}")
    print()
    
    # Test preprocessing
    print("Testing image preprocessing...")
    
    # Create dummy image
    dummy_eye = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    processed_eye = preprocess_eye_roi(dummy_eye)
    print(f"Eye ROI shape: {dummy_eye.shape} -> {processed_eye.shape}")
    print(f"Eye ROI range: [{processed_eye.min():.3f}, {processed_eye.max():.3f}]")
    
    dummy_mouth = np.random.randint(0, 255, (60, 60, 3), dtype=np.uint8)
    processed_mouth = preprocess_mouth_roi(dummy_mouth)
    print(f"Mouth ROI shape: {dummy_mouth.shape} -> {processed_mouth.shape}")
    print(f"Mouth ROI range: [{processed_mouth.min():.3f}, {processed_mouth.max():.3f}]")
    
    print()
    print("="*70)