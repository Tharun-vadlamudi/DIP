"""
Historical Document Restoration & Binarization
Module: restoration.py (Member 2)
Description: Document restoration, character edge sharpening, and unsharp masking.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Tuple
import cv2
import numpy as np
from src.background_correction import restore_illumination


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 2.0,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Sharpens character strokes using classical unsharp masking:
      I_sharp = I + strength * (I - GaussianBlur(I))
    """
    img_f = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = img_f - blurred
    sharpened = img_f + strength * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def homomorphic_filter(
    image: np.ndarray,
    d0: float = 30.0,
    gamma_h: float = 1.5,
    gamma_l: float = 0.5,
    c: float = 1.0,
) -> np.ndarray:
    """
    Classical frequency-domain homomorphic filtering (Member 2 Deliverable 6.2).
    Separates illumination from reflectance:
      ln(I) = ln(L) + ln(R)
    Applies Gaussian high-emphasis filter in frequency domain via 2D FFT,
    attenuating low-frequency non-uniform illumination and enhancing
    high-frequency text edges.
    """
    img_f = image.astype(np.float64) / 255.0
    img_log = np.log1p(img_f)

    # 2D FFT and center shift
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    # Spatial frequency grid
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    X, Y = np.meshgrid(x, y)
    D2 = X**2 + Y**2

    # High-emphasis filter transfer function H(u, v)
    H = (gamma_h - gamma_l) * (1.0 - np.exp(-c * D2 / (d0**2 + 1e-8))) + gamma_l

    filtered_shift = dft_shift * H
    filtered_dft = np.fft.ifftshift(filtered_shift)
    img_back = np.fft.ifft2(filtered_dft)
    img_exp = np.expm1(np.real(img_back))

    norm = cv2.normalize(img_exp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return np.clip(norm, 0, 255).astype(np.uint8)


def restore_document_stage2(
    image: np.ndarray,
    method: str = "gaussian_division",
    sigma_bg: float = 25.0,
    apply_sharpening: bool = False,
    sharp_strength: float = 0.5,
) -> np.ndarray:
    """
    Executes complete Member 2 restoration:
    1. Background correction (Gaussian division or Frequency-Domain Homomorphic filter)
    2. Optional edge sharpening
    """
    if method == "homomorphic":
        restored = homomorphic_filter(image)
    else:
        restored = restore_illumination(image, sigma=sigma_bg)

    if apply_sharpening:
        restored = unsharp_mask(restored, strength=sharp_strength)
    return restored
