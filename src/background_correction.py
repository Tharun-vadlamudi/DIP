"""
Historical Document Restoration & Binarization
Module: background_correction.py (Member 2)
Description: Classical background estimation and illumination correction
via Gaussian division and morphological rolling-ball approximations.
"""

from typing import Tuple
import cv2
import numpy as np


def estimate_background_gaussian(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Estimates slowly varying background illumination using large Gaussian smoothing."""
    image_f = image.astype(np.float32)
    bg = cv2.GaussianBlur(image_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return bg


def estimate_background_morphological(image: np.ndarray, ksize: int = 51) -> np.ndarray:
    """Estimates background using large structural opening (rolling ball approximation)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return bg.astype(np.float32)


def correct_illumination_division(
    image: np.ndarray,
    background: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Corrects uneven illumination by dividing the input image by the estimated background.
    Normalizes by the mean background intensity to preserve global tone:
      I_corr = (I * mean(BG)) / max(BG, epsilon)
    """
    image_f = image.astype(np.float32)
    bg_mean = float(np.mean(background))
    corrected = (image_f * bg_mean) / np.maximum(background, epsilon)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def restore_illumination(
    image: np.ndarray,
    sigma: float = 25.0,
    epsilon: float = 1.0,
) -> np.ndarray:
    """Integrated Member 2 illumination restoration pipeline using Gaussian division."""
    bg = estimate_background_gaussian(image, sigma=sigma)
    return correct_illumination_division(image, bg, epsilon=epsilon)
