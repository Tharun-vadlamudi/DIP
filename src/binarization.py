"""
Historical Document Restoration & Binarization
Module: binarization.py (Member 3)
Description: Comprehensive classical document binarization algorithms including
global thresholding (Otsu), local adaptive thresholding (Mean, Gaussian),
and document-degradation specialized thresholding (Niblack, Sauvola, Wolf-Jolion).
"""

from typing import Tuple, Union
import cv2
import numpy as np


def global_fixed_threshold(image: np.ndarray, thresh_val: int = 128) -> np.ndarray:
    """
    Standard global fixed thresholding.
    Returns binary image: 0 = foreground (text), 255 = background (paper).
    """
    _, binary = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)
    return binary


def otsu_threshold(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Otsu's Global Thresholding (Otsu, 1979).
    Maximizes the between-class variance between foreground and background.
    Returns: (binary_image, optimal_threshold)
    """
    thresh_val, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, thresh_val


def adaptive_mean_threshold(image: np.ndarray, window_size: int = 31, c: int = 10) -> np.ndarray:
    """
    Adaptive Mean Thresholding.
    Threshold at (x,y) is the mean of window_size x window_size minus constant C.
    """
    if window_size % 2 == 0:
        window_size += 1
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_size, c
    )


def adaptive_gaussian_threshold(image: np.ndarray, window_size: int = 31, c: int = 10) -> np.ndarray:
    """
    Adaptive Gaussian Thresholding.
    Threshold at (x,y) is a Gaussian-weighted sum of window_size x window_size minus constant C.
    """
    if window_size % 2 == 0:
        window_size += 1
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window_size, c
    )


def niblack_threshold(
    image: np.ndarray, window_size: int = 31, k: float = -0.2
) -> np.ndarray:
    """
    Niblack's Local Thresholding (Niblack, 1986).
    Formula: T(x, y) = m(x, y) + k * s(x, y)
    where m(x, y) is local mean, s(x, y) is local standard deviation.
    """
    if window_size % 2 == 0:
        window_size += 1

    img_f = image.astype(np.float64)
    ksize = (window_size, window_size)

    mean = cv2.boxFilter(img_f, cv2.CV_64F, ksize, borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.boxFilter(img_f * img_f, cv2.CV_64F, ksize, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(0.0, sq_mean - (mean * mean))
    std = np.sqrt(var)

    thresh = mean + k * std
    binary = np.where(img_f > thresh, 255, 0).astype(np.uint8)
    return binary


def sauvola_threshold(
    image: np.ndarray, window_size: int = 25, k: float = 0.34, r: float = 128.0
) -> np.ndarray:
    """
    Sauvola and Pietikäinen's Adaptive Document Binarization (Sauvola & Pietikäinen, 2000).
    Formulated specifically for degraded documents with variable illumination and stains:
    Formula: T(x, y) = m(x, y) * [1 + k * (s(x, y) / R - 1)]
    where:
      m(x, y): local window mean
      s(x, y): local standard deviation
      R: dynamic range of standard deviation (128 for 8-bit images)
      k: control parameter in [0.2, 0.5]
    """
    if window_size % 2 == 0:
        window_size += 1

    img_f = image.astype(np.float64)
    ksize = (window_size, window_size)

    # Local moments via fast box filter O(1)
    mean = cv2.boxFilter(img_f, cv2.CV_64F, ksize, borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.boxFilter(img_f * img_f, cv2.CV_64F, ksize, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(0.0, sq_mean - (mean * mean))
    std = np.sqrt(var)

    thresh = mean * (1.0 + k * (std / r - 1.0))
    binary = np.where(img_f > thresh, 255, 0).astype(np.uint8)
    return binary


def wolf_threshold(
    image: np.ndarray, window_size: int = 25, k: float = 0.35
) -> np.ndarray:
    """
    Wolf and Jolion's Binarization Algorithm (Wolf & Jolion, 2002).
    Addresses Sauvola's degradation on low-contrast characters by normalizing with global min:
    Formula: T(x, y) = m(x, y) - k * (1 - s(x, y) / max_s) * (m(x, y) - min_I)
    """
    if window_size % 2 == 0:
        window_size += 1

    img_f = image.astype(np.float64)
    ksize = (window_size, window_size)

    mean = cv2.boxFilter(img_f, cv2.CV_64F, ksize, borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.boxFilter(img_f * img_f, cv2.CV_64F, ksize, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(0.0, sq_mean - (mean * mean))
    std = np.sqrt(var)

    min_val = np.min(img_f)
    max_std = np.max(std)
    if max_std == 0:
        max_std = 1.0

    thresh = mean - k * (1.0 - std / max_std) * (mean - min_val)
    binary = np.where(img_f > thresh, 255, 0).astype(np.uint8)
    return binary


def binarize_document(
    image: np.ndarray,
    method: str = "sauvola",
    window_size: int = 25,
    k: float = 0.34,
    c: int = 10,
) -> np.ndarray:
    """
    Unified binarization dispatcher.
    Supported methods: 'otsu', 'adaptive_mean', 'adaptive_gaussian', 'niblack', 'sauvola', 'wolf'
    """
    method = method.lower()
    if method == "otsu":
        bin_img, _ = otsu_threshold(image)
        return bin_img
    elif method in ["adaptive_mean", "mean"]:
        return adaptive_mean_threshold(image, window_size=window_size, c=c)
    elif method in ["adaptive_gaussian", "gaussian"]:
        return adaptive_gaussian_threshold(image, window_size=window_size, c=c)
    elif method == "niblack":
        return niblack_threshold(image, window_size=window_size, k=k if k < 0 else -0.2)
    elif method == "sauvola":
        return sauvola_threshold(image, window_size=window_size, k=k)
    elif method == "wolf":
        return wolf_threshold(image, window_size=window_size, k=k)
    else:
        raise ValueError(f"Unknown binarization method: {method}")
