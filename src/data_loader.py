"""
Historical Document Restoration & Binarization
Module: data_loader.py
Description: Dataset loading utilities for degraded historical documents,
intermediate member results, and binary ground truth images.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class DataLoader:
    """
    Manages loading and indexing of document datasets and pipeline stages.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Default to workspace DIP directory
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.project_dir = self.base_dir / "Historical_Document_Project"

        # Check root dataset directory first, fallback to Historical_Document_Project/dataset/DIBCO2018
        if (self.base_dir / "dataset" / "degraded").exists():
            self.dataset_dir = self.base_dir / "dataset"
            self.degraded_dir = self.dataset_dir / "degraded"
            self.ground_truth_dir = self.dataset_dir / "ground_truth"
        else:
            self.dataset_dir = self.project_dir / "dataset" / "DIBCO2018"
            self.degraded_dir = self.dataset_dir / "degraded"
            self.ground_truth_dir = self.dataset_dir / "ground_truth" / "gt"

        # Results directory per specification
        self.results_dir = self.base_dir / "results"
        self.results_preprocessing = self.results_dir / "preprocessing"
        self.results_restoration = self.results_dir / "restoration"
        self.results_binarization = self.results_dir / "binarization"
        self.results_evaluation = self.results_dir / "evaluation"

        # Legacy member directories
        self.m1_results_dir = self.project_dir / "member1_results"
        self.m2_results_dir = self.project_dir / "member2_results"
        self.m3_results_dir = self.project_dir / "member3_results"

    def get_document_ids(self) -> List[int]:
        """Returns sorted list of available document numeric IDs."""
        files = list(self.degraded_dir.glob("*.bmp"))
        ids = []
        for f in files:
            stem = f.stem
            if stem.isdigit():
                ids.append(int(stem))
        return sorted(ids)

    def load_degraded_image(self, doc_id: int) -> np.ndarray:
        """Load original degraded document image as grayscale."""
        candidates = [
            self.degraded_dir / f"{doc_id}.bmp",
            self.project_dir / "dataset" / "DIBCO2018" / "degraded" / f"{doc_id}.bmp",
        ]
        path = None
        for p in candidates:
            if p.exists():
                path = p
                break
        if path is None:
            raise FileNotFoundError(f"Degraded document {doc_id} not found in {[str(c) for c in candidates]}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image at {path}")
        return img

    def load_ground_truth(self, doc_id: int) -> Optional[np.ndarray]:
        """Load binary ground truth image (0 = foreground text, 255 = background paper)."""
        candidates = [
            self.ground_truth_dir / f"{doc_id}_gt.bmp",
            self.ground_truth_dir / f"{doc_id}.bmp",
            self.project_dir / "dataset" / "DIBCO2018" / "ground_truth" / "gt" / f"{doc_id}_gt.bmp",
        ]
        path = None
        for p in candidates:
            if p.exists():
                path = p
                break
        if path is None:
            return None
        gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            # Ensure strictly binary [0, 255]
            gt = np.where(gt > 127, 255, 0).astype(np.uint8)
        return gt

    def load_member2_restored(self, doc_id: int, method: str = "gaussian_division") -> np.ndarray:
        """
        Load restored image from Member 2 deliverables.
        method: 'gaussian_division' or 'original_clahe'
        """
        folder = self.m2_results_dir / "final_restoration" / method
        suffix = "_division.bmp" if method == "gaussian_division" else ".bmp"

        # Search matching file
        candidates = list(folder.glob(f"{doc_id}_*{suffix}"))
        if not candidates:
            # Fallback search
            candidates = list(folder.glob(f"{doc_id}_*.bmp"))

        if not candidates:
            raise FileNotFoundError(f"Member 2 restored image for doc {doc_id} ({method}) not found in {folder}")

        img = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
        return img

    def get_all_pairs(self, method: str = "gaussian_division") -> List[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
        """
        Returns list of tuples (doc_id, restored_img, gt_img) for all documents.
        """
        doc_ids = self.get_document_ids()
        pairs = []
        for doc_id in doc_ids:
            restored = self.load_member2_restored(doc_id, method=method)
            gt = self.load_ground_truth(doc_id)
            pairs.append((doc_id, restored, gt))
        return pairs
