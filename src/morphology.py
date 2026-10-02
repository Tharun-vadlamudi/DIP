"""
Historical Document Restoration & Binarization
Module: morphology.py (Member 3)
Description: Morphological cleanup and connected-component post-processing
for binary historical document images. Eliminates salt-and-pepper noise,
bridges hairline stroke fractures, and removes scanner border artifacts.
"""

from typing import Tuple
import cv2
import numpy as np


def morphological_opening(
    binary_image: np.ndarray,
    kernel_size: Tuple[int, int] = (2, 2),
    shape: int = cv2.MORPH_RECT,
) -> np.ndarray:
    """
    Applies morphological opening to the text foreground.
    Input/Output: 0 = foreground (text), 255 = background (paper).
    Opening (Erosion followed by Dilation) eliminates small isolated noise specks.
    """
    # Invert so text is 255 (OpenCV morphology functions operate on white foreground)
    fg = 255 - binary_image
    kernel = cv2.getStructuringElement(shape, kernel_size)
    opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    return 255 - opened


def morphological_closing(
    binary_image: np.ndarray,
    kernel_size: Tuple[int, int] = (2, 2),
    shape: int = cv2.MORPH_RECT,
) -> np.ndarray:
    """
    Applies morphological closing to the text foreground.
    Closing (Dilation followed by Erosion) bridges hairline character breaks
    and fills tiny pinholes inside character strokes.
    """
    fg = 255 - binary_image
    kernel = cv2.getStructuringElement(shape, kernel_size)
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    return 255 - closed


def remove_small_components(
    binary_image: np.ndarray,
    min_area: int = 10,
    connectivity: int = 8,
) -> np.ndarray:
    """
    Filter out isolated foreground components whose pixel area is smaller than min_area.
    Vectorized implementation using NumPy array indexing for instantaneous execution.
    """
    if min_area <= 1:
        return binary_image.copy()

    fg = 255 - binary_image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=connectivity)

    if num_labels <= 1:
        return binary_image.copy()

    # Create boolean lookup table for labels: keep if area >= min_area (background label 0 is always False)
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0
    keep_mask = (areas >= min_area)

    # Vectorized direct lookup
    clean_fg = np.where(keep_mask[labels], 255, 0).astype(np.uint8)
    return 255 - clean_fg


def remove_border_artifacts(
    binary_image: np.ndarray,
    border_margin: int = 4,
    max_border_area_ratio: float = 0.25,
) -> np.ndarray:
    """
    Removes black border artifacts resulting from document scanning/cropping.
    Vectorized implementation for instant execution.
    """
    fg = 255 - binary_image
    h, w = fg.shape
    total_pixels = h * w
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    if num_labels <= 1:
        return binary_image.copy()

    x = stats[:, cv2.CC_STAT_LEFT]
    y = stats[:, cv2.CC_STAT_TOP]
    bw = stats[:, cv2.CC_STAT_WIDTH]
    bh = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]

    touches_border = (
        (x <= border_margin)
        | (y <= border_margin)
        | ((x + bw) >= (w - border_margin))
        | ((y + bh) >= (h - border_margin))
    )
    is_border_artifact = touches_border & (
        (area > (total_pixels * 0.02))
        | (bw >= (w - 2 * border_margin))
        | (bh >= (h - 2 * border_margin))
    )
    is_border_artifact[0] = False  # Background

    remove_mask = is_border_artifact[labels]
    clean_fg = fg.copy()
    clean_fg[remove_mask] = 0

    return 255 - clean_fg


def cleanup_binary_document(
    binary_image: np.ndarray,
    apply_closing: bool = False,
    closing_ksize: Tuple[int, int] = (2, 2),
    min_area: int = 12,
    clean_borders: bool = True,
) -> np.ndarray:
    """
    Integrated pipeline for binary morphological and topological cleanup.
    1. Optional gentle closing to bridge micro-fissures in ink.
    2. Connected component area thresholding to eliminate dust and paper foxing specks.
    3. Removal of peripheral scanning borders.
    """
    res = binary_image.copy()

    if clean_borders:
        res = remove_border_artifacts(res)

    if apply_closing:
        res = morphological_closing(res, kernel_size=closing_ksize)

    if min_area > 0:
        res = remove_small_components(res, min_area=min_area)

    return res
