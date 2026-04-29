"""
YOLO preprocessing: image preparation for inference.

Handles letterbox resizing, normalization, and coordinate scaling.
"""

import numpy as np
from typing import Tuple


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    fill_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterbox padding to maintain aspect ratio.
    
    Args:
        image: Input BGR image (H, W, 3)
        target_size: Target (width, height)
        fill_value: Padding fill value (default: 114 gray)
        
    Returns:
        Tuple of:
        - Padded image (target_size)
        - Scale factor applied
        - Padding offsets (pad_x, pad_y)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to fit while maintaining aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    import cv2
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded canvas
    padded = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)
    
    # Center the resized image
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return padded, scale, (pad_x, pad_y)


def preprocess_for_yolo(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Full preprocessing pipeline for YOLO inference.
    
    Args:
        image: Input BGR image (H, W, 3)
        target_size: Model input size (width, height)
        
    Returns:
        Tuple of:
        - Input tensor (1, 3, H, W) float32 normalized to [0, 1]
        - Scale factor for coordinate mapping
        - Padding offsets (pad_x, pad_y)
    """
    # Apply letterbox
    padded, scale, padding = letterbox(image, target_size)
    
    # Convert BGR -> RGB
    rgb = padded[:, :, ::-1]
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # Transpose HWC -> CHW
    transposed = normalized.transpose(2, 0, 1)
    
    # Add batch dimension
    batched = np.expand_dims(transposed, 0)
    
    return batched, scale, padding


def scale_boxes(
    boxes: np.ndarray,
    scale: float,
    padding: Tuple[int, int],
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Scale bounding boxes from model output back to original image coordinates.
    
    OPT-2: Optimized with vectorized operations and in-place clipping.
    
    Args:
        boxes: Boxes in xyxy format (N, 4) in model input coordinates
        scale: Scale factor used during preprocessing
        padding: Padding offsets (pad_x, pad_y)
        original_size: Original image size (width, height)
        
    Returns:
        Scaled boxes in original image coordinates
    """
    if len(boxes) == 0:
        return boxes
    
    pad_x, pad_y = padding
    orig_w, orig_h = original_size
    
    # OPT-2: Vectorized scaling with fewer intermediate arrays
    scaled = boxes.copy()
    scaled[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale  # x coords
    scaled[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale  # y coords
    
    # In-place clipping
    np.clip(scaled[:, [0, 2]], 0, orig_w, out=scaled[:, [0, 2]])
    np.clip(scaled[:, [1, 3]], 0, orig_h, out=scaled[:, [1, 3]])
    
    return scaled
