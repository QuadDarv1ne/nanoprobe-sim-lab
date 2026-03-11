# -*- coding: utf-8 -*-
"""Surface image comparison module for Nanoprobe Simulation Lab"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from scipy import ndimage
from scipy.stats import pearsonr

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _calculate_ssim(img1: np.ndarray, img2: np.ndarray, win_size: int = 7) -> float:
    """Calculate SSIM manually (simplified version)"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")

    K1, K2 = 0.01, 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = ndimage.uniform_filter(img1, size=win_size)
    mu2 = ndimage.uniform_filter(img2, size=win_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = ndimage.uniform_filter(img1 ** 2, size=win_size) - mu1_sq
    sigma2_sq = ndimage.uniform_filter(img2 ** 2, size=win_size) - mu2_sq
    sigma12 = ndimage.uniform_filter(img1 * img2, size=win_size) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def _calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR manually"""
    mse_val = np.mean((img1 - img2) ** 2)
    if mse_val == 0:
        return float('inf')
    return float(20 * np.log10(1.0 / np.sqrt(mse_val)))


class SurfaceComparator:
    """Surface image comparator for AFM images"""

    def __init__(self, comparison_method: str = "ssim"):
        self.comparison_method = comparison_method
        self.output_dir = Path("output/surface_comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_surfaces(self, image1: np.ndarray, image2: np.ndarray, normalize: bool = True) -> Dict[str, Any]:
        """Compare two surface images"""
        if image1.shape != image2.shape:
            raise ValueError(f"Shape mismatch: {image1.shape} vs {image2.shape}")

        if normalize:
            img1 = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-10)
            img2 = (image2 - image2.min()) / (image2.max() - image2.min() + 1e-10)
        else:
            img1, img2 = image1, image2

        metrics = {}

        try:
            metrics["ssim"] = _calculate_ssim(img1, img2)
            metrics["psnr"] = _calculate_psnr(img1, img2)
        except Exception:
            pass

        mse_val = float(np.mean((img1 - img2) ** 2))
        metrics["mse"] = mse_val
        metrics["pearson"] = float(pearsonr(img1.flatten(), img2.flatten())[0])
        metrics["mean_diff"] = float(np.mean(np.abs(img1 - img2)))

        scores = []
        if "ssim" in metrics:
            scores.append(metrics["ssim"] * 0.4)
        if "psnr" in metrics:
            scores.append(min(1.0, metrics["psnr"] / 40) * 0.3)
        if "pearson" in metrics:
            scores.append(((metrics["pearson"] + 1) / 2) * 0.3)
        metrics["similarity"] = float(np.sum(scores)) if scores else 0.0

        return metrics

    def compare_files(self, path1: str, path2: str, save: bool = True) -> Dict[str, Any]:
        """Compare two image files"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL not installed")
        
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        if len(arr1.shape) == 3:
            arr1 = arr1.mean(axis=2)
        if len(arr2.shape) == 3:
            arr2 = arr2.mean(axis=2)
        
        metrics = self.compare_surfaces(arr1, arr2)
        
        cid = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = {"id": cid, "path1": path1, "path2": path2, **metrics}
        
        if save:
            with open(self.output_dir / f"{cid}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            
            if MATPLOTLIB_AVAILABLE:
                self._save_visualization(arr1, arr2, metrics, cid)
        
        return results

    def _save_visualization(self, img1, img2, metrics: Dict, cid: str):
        """Save comparison visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(img1, cmap="viridis")
        axes[0].set_title("Image 1")
        axes[0].axis("off")
        axes[1].imshow(img2, cmap="viridis")
        axes[1].set_title("Image 2")
        axes[1].axis("off")
        diff = np.abs(img1 - img2)
        axes[2].imshow(diff, cmap="hot")
        axes[2].set_title(f"Diff (mean={metrics['mean_diff']:.4f})")
        axes[2].axis("off")
        plt.suptitle(f"SSIM: {metrics.get('ssim', 0):.4f}, Similarity: {metrics.get('similarity', 0):.4f}")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{cid}_viz.png", dpi=150, bbox_inches="tight")
        plt.close()


def compare_surfaces(path1: str, path2: str, out: str = "output/surface_comparisons") -> Dict[str, Any]:
    """Quick surface comparison function"""
    c = SurfaceComparator()
    c.output_dir = Path(out)
    return c.compare_files(path1, path2)


if __name__ == "__main__":
    print("=== Surface Comparator Test ===")
    if not PIL_AVAILABLE:
        print("PIL not installed")
    else:
        test_dir = Path("output/surface_comparisons/test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        np.random.seed(42)
        x = np.linspace(-2, 2, 256)
        X, Y = np.meshgrid(x, x)
        base = np.sin(3 * np.sqrt(X**2 + Y**2)) * np.exp(-(X**2 + Y**2) / 2)
        
        img1 = ((base + np.random.randn(256, 256) * 0.05 - base.min()) / (base.max() - base.min()) * 255).astype(np.uint8)
        img2 = ((base * 1.1 + np.random.randn(256, 256) * 0.07 - base.min()) / (base.max() - base.min()) * 255).astype(np.uint8)
        
        Image.fromarray(img1).save(test_dir / "s1.png")
        Image.fromarray(img2).save(test_dir / "s2.png")
        
        r = compare_surfaces(str(test_dir / "s1.png"), str(test_dir / "s2.png"))
        print(f"SSIM: {r.get('ssim', 0):.4f}")
        print(f"Similarity: {r.get('similarity', 0):.4f}")
        print(f"Mean Diff: {r.get('mean_diff', 0):.6f}")
        print("Test completed!")
