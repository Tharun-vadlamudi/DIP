"""
Historical Document Restoration & Binarization
Module: enhancement.py (Member 1)
Description: Histogram equalization, contrast stretching, and CLAHE.
"""

from typing import Tuple
import cv2
import numpy as np


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Standard global histogram equalization."""
    return cv2.equalizeHist(image)


def contrast_stretching(image: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """Contrast stretching based on percentile clipping."""
    p_low, p_high = np.percentile(image, (lower_percentile, upper_percentile))
    if p_high - p_low == 0:
        return image.copy()
    stretched = (image.astype(np.float32) - p_low) * (255.0 / (p_high - p_low))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def enhance_document(
    image: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Applies chosen contrast enhancement method.
    Methods: 'clahe', 'hist_eq', 'contrast_stretching'
    """
    if method == "clahe":
        return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    elif method == "hist_eq":
        return histogram_equalization(image)
    elif method == "contrast_stretching":
        return contrast_stretching(image)
    else:
        return image.copy()
