"""
Historical Document Restoration & Binarization
Module: preprocessing.py (Member 1)
Description: Grayscale conversion, intensity normalization, and noise filtering.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts input image to single-channel 8-bit grayscale if needed."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Normalizes image pixel intensities to full 8-bit range [0, 255]."""
    norm = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def filter_noise_median(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Applies median filtering to suppress salt-and-pepper noise."""
    return cv2.medianBlur(image, ksize)


def filter_noise_gaussian(
    image: np.ndarray, ksize: Tuple[int, int] = (3, 3), sigma: float = 1.0
) -> np.ndarray:
    """Applies Gaussian smoothing to suppress high-frequency Gaussian noise."""
    return cv2.GaussianBlur(image, ksize, sigmaX=sigma)


def resize_document(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resizes document image for consistent scale processing.
    target_size: (width, height) tuple
    scale_factor: uniform scaling factor (e.g. 0.5)
    """
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=interpolation)
    elif scale_factor is not None and scale_factor != 1.0:
        return cv2.resize(
            image, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation
        )
    return image.copy()


def preprocess_document(
    image: np.ndarray,
    noise_filter: str = "median",
    ksize: int = 3,
    target_size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
) -> np.ndarray:
    """
    Integrated preprocessing pipeline for Member 1:
    1. Grayscale conversion
    2. Intensity normalization
    3. Noise filtering
    4. Optional resizing
    """
    gray = to_grayscale(image)
    norm = normalize_intensity(gray)
    if noise_filter == "median":
        filtered = filter_noise_median(norm, ksize=ksize)
    elif noise_filter == "gaussian":
        filtered = filter_noise_gaussian(norm, ksize=(ksize, ksize))
    else:
        filtered = norm

    if target_size is not None or scale_factor is not None:
        filtered = resize_document(filtered, target_size=target_size, scale_factor=scale_factor)

    return filtered
