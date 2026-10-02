"""
Historical Document Restoration & Binarization
Module: pipeline.py (Full Integration)
Description: Complete end-to-end integrated pipeline executing Member 1,
Member 2, and Member 3 stages on historical document collections.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.preprocessing import preprocess_document
from src.enhancement import enhance_document
from src.background_correction import restore_illumination
from src.binarization import binarize_document, otsu_threshold, sauvola_threshold
from src.morphology import cleanup_binary_document
from src.evaluation import compute_binary_metrics, generate_comparison_plot


class RestorationPipeline:
    """
    Unified end-to-end restoration and binarization pipeline.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.loader = data_loader if data_loader else DataLoader()

    def process_document(
        self,
        doc_id: int,
        binarization_method: str = "sauvola",
        sauvola_w: int = 25,
        sauvola_k: float = 0.34,
        cc_min_area: int = 12,
        apply_closing: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Executes end-to-end processing for a single document:
        Stage 1: Preprocessing & Enhancement (Member 1)
        Stage 2: Illumination Restoration (Member 2)
        Stage 3: Binarization & Morphological Cleanup (Member 3)
        """
        # Load degraded input
        degraded = self.loader.load_degraded_image(doc_id)

        # Stage 1: Member 1 Preprocessing & Contrast Enhancement
        preprocessed = preprocess_document(degraded, noise_filter="median", ksize=3)
        enhanced = enhance_document(preprocessed, method="clahe", clip_limit=2.0)

        # Stage 2: Member 2 Restoration
        restored = restore_illumination(enhanced, sigma=25.0)

        # Stage 3: Member 3 Thresholding
        otsu_bin, _ = otsu_threshold(restored)
        raw_bin = binarize_document(
            restored,
            method=binarization_method,
            window_size=sauvola_w,
            k=sauvola_k,
        )

        # Stage 3: Member 3 Morphological Cleanup
        final_clean = cleanup_binary_document(
            raw_bin,
            apply_closing=apply_closing,
            min_area=cc_min_area,
            clean_borders=True,
        )

        gt = self.loader.load_ground_truth(doc_id)

        return {
            "doc_id": doc_id,
            "degraded": degraded,
            "preprocessed": preprocessed,
            "enhanced": enhanced,
            "restored": restored,
            "otsu": otsu_bin,
            "raw_bin": raw_bin,
            "final_clean": final_clean,
            "gt": gt,
        }

    def run_benchmark(
        self,
        output_dir: Optional[Path] = None,
        use_member2_existing: bool = False,
    ) -> pd.DataFrame:
        """
        Runs comprehensive evaluation across all 10 documents in the dataset.
        Executes end-to-end DIP pipeline across Member 1, Member 2, and Member 3.
        Saves all deliverables into the standard results/ directory and mirrors
        to member results directories for full compatibility.
        """
        if output_dir is None:
            output_dir = self.loader.results_dir
        output_dir = Path(output_dir)

        # Standard results structure
        prep_dir = output_dir / "preprocessing"
        rest_dir = output_dir / "restoration"
        bin_dir = output_dir / "binarization"
        eval_dir = output_dir / "evaluation"
        cmp_dir = eval_dir / "comparisons"

        for d in [
            prep_dir,
            rest_dir,
            bin_dir / "otsu",
            bin_dir / "adaptive_gaussian",
            bin_dir / "niblack",
            bin_dir / "sauvola",
            bin_dir / "final_clean",
            cmp_dir,
            eval_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Also prepare legacy member3_results directory if existing
        m3_dir = self.loader.m3_results_dir
        (m3_dir / "thresholding" / "otsu").mkdir(parents=True, exist_ok=True)
        (m3_dir / "thresholding" / "adaptive_gaussian").mkdir(parents=True, exist_ok=True)
        (m3_dir / "thresholding" / "sauvola").mkdir(parents=True, exist_ok=True)
        (m3_dir / "thresholding" / "niblack").mkdir(parents=True, exist_ok=True)
        (m3_dir / "morphology_cleaned" / "final_binary").mkdir(parents=True, exist_ok=True)
        (m3_dir / "comparisons").mkdir(parents=True, exist_ok=True)
        (m3_dir / "evaluation").mkdir(parents=True, exist_ok=True)

        doc_ids = self.loader.get_document_ids()
        all_metrics = []

        mode_str = "Precomputed Deliverables" if use_member2_existing else "Live Integrated Pipeline (M1 -> M2 -> M3)"
        print(f"Executing Full Document Evaluation across {len(doc_ids)} documents [{mode_str}]...")

        for doc_id in doc_ids:
            print(f"Processing Document {doc_id:2d}...", end=" ", flush=True)

            if use_member2_existing:
                restored = self.loader.load_member2_restored(doc_id, method="gaussian_division")
                degraded = self.loader.load_degraded_image(doc_id)
                preprocessed = preprocess_document(degraded, noise_filter="median", ksize=3)
                enhanced = enhance_document(preprocessed, method="clahe", clip_limit=2.0)
            else:
                stages = self.process_document(doc_id)
                degraded = stages["degraded"]
                preprocessed = stages["preprocessed"]
                enhanced = stages["enhanced"]
                restored = stages["restored"]

            gt = self.loader.load_ground_truth(doc_id)

            # Stage 3 Thresholding Variations
            b_otsu, _ = otsu_threshold(restored)
            b_adapt = binarize_document(restored, method="adaptive_gaussian", window_size=31, c=10)
            b_niblack = binarize_document(restored, method="niblack", window_size=31, k=-0.2)
            b_sauvola = binarize_document(restored, method="sauvola", window_size=25, k=0.34)

            # Morphological & connected component cleanup on Sauvola
            b_final = cleanup_binary_document(
                b_sauvola,
                apply_closing=False,
                min_area=12,
                clean_borders=True,
            )

            # Save Member 1 Preprocessing outputs
            cv2.imwrite(str(prep_dir / f"{doc_id}_preprocessed.bmp"), preprocessed)
            cv2.imwrite(str(prep_dir / f"{doc_id}_enhanced.bmp"), enhanced)

            # Save Member 2 Restoration output
            cv2.imwrite(str(rest_dir / f"{doc_id}_restored.bmp"), restored)

            # Save Member 3 Binarization outputs in results/
            cv2.imwrite(str(bin_dir / "otsu" / f"{doc_id}_otsu.bmp"), b_otsu)
            cv2.imwrite(str(bin_dir / "adaptive_gaussian" / f"{doc_id}_adaptive.bmp"), b_adapt)
            cv2.imwrite(str(bin_dir / "niblack" / f"{doc_id}_niblack.bmp"), b_niblack)
            cv2.imwrite(str(bin_dir / "sauvola" / f"{doc_id}_sauvola.bmp"), b_sauvola)
            cv2.imwrite(str(bin_dir / "final_clean" / f"{doc_id}_final_clean.bmp"), b_final)

            # Mirror to member3_results for backward compatibility
            cv2.imwrite(str(m3_dir / "thresholding" / "otsu" / f"{doc_id}_otsu.bmp"), b_otsu)
            cv2.imwrite(str(m3_dir / "thresholding" / "adaptive_gaussian" / f"{doc_id}_adaptive.bmp"), b_adapt)
            cv2.imwrite(str(m3_dir / "thresholding" / "niblack" / f"{doc_id}_niblack.bmp"), b_niblack)
            cv2.imwrite(str(m3_dir / "thresholding" / "sauvola" / f"{doc_id}_sauvola.bmp"), b_sauvola)
            cv2.imwrite(str(m3_dir / "morphology_cleaned" / "final_binary" / f"{doc_id}_final_clean.bmp"), b_final)

            # Evaluate quantitative metrics against Ground Truth
            m_clean = {}
            if gt is not None:
                m_otsu = compute_binary_metrics(b_otsu, gt, compute_drd=False)
                m_adapt = compute_binary_metrics(b_adapt, gt, compute_drd=False)
                m_nib = compute_binary_metrics(b_niblack, gt, compute_drd=False)
                m_sauv = compute_binary_metrics(b_sauvola, gt, compute_drd=False)
                m_clean = compute_binary_metrics(b_final, gt, compute_drd=True)

                all_metrics.append({
                    "Doc ID": doc_id,
                    "Otsu FM (%)": m_otsu["F-Measure (%)"],
                    "AdaptGauss FM (%)": m_adapt["F-Measure (%)"],
                    "Niblack FM (%)": m_nib["F-Measure (%)"],
                    "Sauvola FM (%)": m_sauv["F-Measure (%)"],
                    "Final Clean FM (%)": m_clean["F-Measure (%)"],
                    "Final Prec (%)": m_clean["Precision (%)"],
                    "Final Rec (%)": m_clean["Recall (%)"],
                    "Final PSNR (dB)": m_clean["PSNR (dB)"],
                    "Final MSE": m_clean["MSE"],
                    "Final DRD": m_clean.get("DRD", 0.0),
                })

            # Save comprehensive side-by-side comparison plot (Deliverable 7 / Section 13)
            cmp_path = cmp_dir / f"document_{doc_id}_pipeline_comparison.png"
            fig = generate_comparison_plot(
                degraded_img=degraded,
                restored_img=restored,
                otsu_img=b_otsu,
                sauvola_img=b_sauvola,
                final_cleaned_img=b_final,
                gt_img=gt,
                preprocessed_img=enhanced,
                metrics=m_clean if m_clean else None,
                save_path=cmp_path,
                title=f"DIBCO 2018 Document #{doc_id} - Classical DIP Pipeline",
            )
            # Mirror to legacy comparisons
            legacy_cmp_path = m3_dir / "comparisons" / f"document_{doc_id}_restoration_binarization.png"
            fig.savefig(str(legacy_cmp_path), dpi=200, bbox_inches="tight")
            plt.close(fig)

            print("Done.")

        df_metrics = pd.DataFrame(all_metrics)

        # Save metrics CSV in both locations
        csv_path = eval_dir / "pipeline_evaluation_summary.csv"
        df_metrics.to_csv(csv_path, index=False)
        legacy_csv_path = m3_dir / "evaluation" / "member3_evaluation_summary.csv"
        df_metrics.to_csv(legacy_csv_path, index=False)
        print(f"\nEvaluation summary saved to: {csv_path}")

        # Generate summary comparison charts
        self._plot_metric_comparisons(df_metrics, eval_dir)
        self._plot_metric_comparisons(df_metrics, m3_dir / "evaluation")

        return df_metrics

    def _plot_metric_comparisons(self, df: pd.DataFrame, eval_dir: Path):
        """Generates visual charts summarizing benchmark results."""
        plt.figure(figsize=(13, 6))
        x = np.arange(len(df["Doc ID"]))
        width = 0.16

        plt.bar(x - 2 * width, df["Otsu FM (%)"], width, label="Otsu Global", color="#7f7f7f")
        plt.bar(x - 1 * width, df["AdaptGauss FM (%)"], width, label="Adaptive Gauss", color="#1f77b4")
        plt.bar(x, df["Niblack FM (%)"], width, label="Niblack", color="#ff7f0e")
        plt.bar(x + 1 * width, df["Sauvola FM (%)"], width, label="Sauvola Raw", color="#2ca02c")
        plt.bar(x + 2 * width, df["Final Clean FM (%)"], width, label="Final Cleaned (Sauvola+Morph)", color="#9467bd")

        plt.xlabel("Document Number", fontweight="bold", fontsize=12)
        plt.ylabel("F-Measure (%)", fontweight="bold", fontsize=12)
        plt.title("Binarization Performance (F-Measure) across DIBCO 2018 Dataset", fontweight="bold", fontsize=14)
        plt.xticks(x, [f"Doc {d}" for d in df["Doc ID"]])
        plt.legend(frameon=True, facecolor="white", edgecolor="gray")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(str(eval_dir / "f_measure_comparison.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    pipeline = RestorationPipeline()
    df = pipeline.run_benchmark(use_member2_existing=False)
    print("\n================== SUMMARY MEANS ACROSS DATASET ==================")
    print(df.mean(numeric_only=True).round(2).to_string())
    print("==================================================================")
