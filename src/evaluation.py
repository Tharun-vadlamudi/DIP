"""
Historical Document Restoration & Binarization
Module: evaluation.py (Member 3)
Description: Quantitative ground-truth evaluation metrics (F-Measure, Precision,
Recall, Specificity, Accuracy, PSNR, MSE, DRD) and comparative visual plotting.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_binary_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    compute_drd: bool = True,
) -> Dict[str, float]:
    """
    Computes standard benchmark metrics between predicted binary image and ground truth.
    Conventions:
      0   = Foreground (Text)
      255 = Background (Paper)
    """
    if pred_binary.shape != gt_binary.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_binary.shape} vs gt {gt_binary.shape}"
        )

    # Boolean foreground masks
    pred_fg = (pred_binary == 0)
    gt_fg = (gt_binary == 0)

    # Confusion matrix components
    tp = int(np.sum(pred_fg & gt_fg))
    fp = int(np.sum(pred_fg & (~gt_fg)))
    fn = int(np.sum((~pred_fg) & gt_fg))
    tn = int(np.sum((~pred_fg) & (~gt_fg)))

    # Metrics
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fm = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # MSE & PSNR on standard [0, 255] intensity scale
    mse = float(np.mean((pred_binary.astype(np.float64) - gt_binary.astype(np.float64)) ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / (mse + 1e-10))) if mse > 0 else 99.0

    metrics = {
        "F-Measure (%)": round(fm * 100.0, 2),
        "Precision (%)": round(prec * 100.0, 2),
        "Recall (%)": round(rec * 100.0, 2),
        "Specificity (%)": round(spec * 100.0, 2),
        "Accuracy (%)": round(acc * 100.0, 2),
        "PSNR (dB)": round(psnr, 2),
        "MSE": round(mse, 2),
    }

    if compute_drd:
        drd = compute_drd_metric(pred_binary, gt_binary)
        metrics["DRD"] = round(drd, 2)

    return metrics


def compute_drd_metric(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """
    Computes Distance Reciprocal Distortion (DRD) based on Lu et al. (IEEE TIP 2013).
    A lower DRD value indicates lower visual distortion.
    Exact vectorized implementation using cv2.filter2D for instantaneous calculation.
    """
    # Foreground text = 1, Background paper = 0
    b = (pred_binary == 0).astype(np.float64)
    gt = (gt_binary == 0).astype(np.float64)

    # 5x5 normalized reciprocal distance weight matrix Wm
    W = np.zeros((5, 5), dtype=np.float64)
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0:
                continue
            W[i + 2, j + 2] = 1.0 / np.sqrt(i * i + j * j)
    W /= np.sum(W)

    # Number of Non-Uniform 8x8 Blocks (NUBN) in Ground Truth
    h, w = gt.shape
    H_trim = h - (h % 8)
    W_trim = w - (w % 8)
    if H_trim >= 8 and W_trim >= 8:
        blocks = gt[:H_trim, :W_trim].reshape(H_trim // 8, 8, W_trim // 8, 8)
        block_sums = blocks.sum(axis=(1, 3))
        nubn = int(np.sum((block_sums > 0) & (block_sums < 64)))
    else:
        nubn = 1

    if nubn == 0:
        nubn = 1

    # Discrepancy mask (flipped pixels)
    diff = (b != gt)
    if not np.any(diff):
        return 0.0

    # 2D convolution of GT with Wm
    conv_gt = cv2.filter2D(gt, -1, W, borderType=cv2.BORDER_CONSTANT)
    distortion_map = np.abs(b - conv_gt)
    total_distortion = float(np.sum(distortion_map[diff]))

    drd = total_distortion / nubn
    return float(drd)


def create_error_map(pred_binary: np.ndarray, gt_binary: np.ndarray) -> np.ndarray:
    """
    Creates an RGB visual error classification map:
    - White [255, 255, 255]: True Negative (Background agreement)
    - Black [0, 0, 0] / Green [34, 139, 34]: True Positive (Correctly detected text)
    - Red [220, 20, 60]: False Positive (Noise / stain wrongly marked as text)
    - Blue [30, 144, 255]: False Negative (Missed faint text stroke)
    """
    h, w = pred_binary.shape
    error_rgb = np.full((h, w, 3), 255, dtype=np.uint8)

    pred_fg = (pred_binary == 0)
    gt_fg = (gt_binary == 0)

    # TP: Green
    tp_mask = pred_fg & gt_fg
    error_rgb[tp_mask] = [34, 139, 34]

    # FP: Red (False text)
    fp_mask = pred_fg & (~gt_fg)
    error_rgb[fp_mask] = [220, 20, 60]

    # FN: Blue (Missed text)
    fn_mask = (~pred_fg) & gt_fg
    error_rgb[fn_mask] = [30, 144, 255]

    return error_rgb


def generate_comparison_plot(
    degraded_img: np.ndarray,
    restored_img: np.ndarray,
    otsu_img: np.ndarray,
    sauvola_img: np.ndarray,
    final_cleaned_img: np.ndarray,
    gt_img: Optional[np.ndarray],
    preprocessed_img: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Document Restoration & Binarization Pipeline",
) -> plt.Figure:
    """
    Generates a high-quality side-by-side comparison figure satisfying Section 13:
    1. Original Degraded
    2. Member 1 Preprocessed & Enhanced (or Sauvola intermediate)
    3. Member 2 Restored Document
    4. Global Otsu Baseline
    5. Final Cleaned Binary Document
    6. Ground Truth & Error Map Overlay
    """
    num_cols = 3
    num_rows = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 11))

    if metrics:
        metric_str = (
            f"F-Measure: {metrics.get('F-Measure (%)', 'N/A')}%  |  "
            f"Precision: {metrics.get('Precision (%)', 'N/A')}%  |  "
            f"Recall: {metrics.get('Recall (%)', 'N/A')}%  |  "
            f"PSNR: {metrics.get('PSNR (dB)', 'N/A')} dB  |  "
            f"DRD: {metrics.get('DRD', 'N/A')}"
        )
        fig.suptitle(f"{title}\n{metric_str}", fontsize=14, fontweight="bold", y=0.98)
    else:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    if preprocessed_img is not None:
        panels = [
            ("1. Original Degraded Document", degraded_img, "gray"),
            ("2. Member 1: Preprocessed & Enhanced", preprocessed_img, "gray"),
            ("3. Member 2: Restored Document", restored_img, "gray"),
            ("4. Member 3: Global Baseline (Otsu)", otsu_img, "gray"),
            ("5. Member 3: Final Cleaned Binary", final_cleaned_img, "gray"),
        ]
    else:
        panels = [
            ("1. Original Degraded Document", degraded_img, "gray"),
            ("2. Member 2: Restored Document", restored_img, "gray"),
            ("3. Member 3: Global Baseline (Otsu)", otsu_img, "gray"),
            ("4. Member 3: Local Sauvola", sauvola_img, "gray"),
            ("5. Member 3: Final Cleaned Binary", final_cleaned_img, "gray"),
        ]

    if gt_img is not None:
        error_map = create_error_map(final_cleaned_img, gt_img)
        panels.append(("6. Ground Truth & Error Overlay (Green=TP, Red=FP, Blue=FN)", error_map, None))
    else:
        panels.append(("6. Ground Truth (Not Available)", np.ones_like(degraded_img) * 255, "gray"))

    for idx, (label, img_data, cmap) in enumerate(panels):
        ax = axes[idx // num_cols, idx % num_cols]
        if cmap is not None:
            ax.imshow(img_data, cmap=cmap)
        else:
            ax.imshow(img_data)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=200, bbox_inches="tight")

    return fig
