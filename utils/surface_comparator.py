"""Surface image comparison module for Nanoprobe Simulation Lab"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

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

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = ndimage.uniform_filter(img1**2, size=win_size) - mu1_sq
    sigma2_sq = ndimage.uniform_filter(img2**2, size=win_size) - mu2_sq
    sigma12 = ndimage.uniform_filter(img1 * img2, size=win_size) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(np.mean(ssim_map))


def _calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR manually"""
    mse_val = np.mean((img1 - img2) ** 2)
    if mse_val == 0:
        return float("inf")
    return float(20 * np.log10(1.0 / np.sqrt(mse_val)))


class SurfaceComparator:
    """Surface image comparator for AFM images"""

    def __init__(self, comparison_method: str = "ssim"):
        """
        Initialize surface comparator

        Args:
            comparison_method: Method for comparison ('ssim', 'mse', 'psnr')
        """
        self.comparison_method = comparison_method
        self.output_dir = Path("output/surface_comparisons")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_surfaces(
        self, image1: np.ndarray, image2: np.ndarray, normalize: bool = True
    ) -> Dict[str, Any]:
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

        # Дополнительные метрики
        metrics["max_diff"] = float(np.max(np.abs(img1 - img2)))
        metrics["std_diff"] = float(np.std(np.abs(img1 - img2)))
        metrics["rms_error"] = float(np.sqrt(np.mean((img1 - img2) ** 2)))

        # Гистограммное сравнение
        hist1, _ = np.histogram(img1.flatten(), bins=64, range=(0, 1))
        hist2, _ = np.histogram(img2.flatten(), bins=64, range=(0, 1))
        hist_corr = float(np.corrcoef(hist1, hist2)[0, 1])
        metrics["histogram_correlation"] = hist_corr if not np.isnan(hist_corr) else 0.0

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

        cid = f"comp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        results = {"id": cid, "path1": path1, "path2": path2, **metrics}

        if save:
            with open(self.output_dir / f"{cid}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

            if MATPLOTLIB_AVAILABLE:
                self._save_visualization(arr1, arr2, metrics, cid)

        return results

    def _save_visualization(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        metrics: Dict,
        cid: str,
        save_difference_map: bool = True,
    ):
        """Save comprehensive comparison visualization"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.25)

        # Основные изображения
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(img1, cmap="viridis", aspect="equal")
        ax1.set_title("Image 1", fontsize=11, fontweight="bold")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(img2, cmap="viridis", aspect="equal")
        ax2.set_title("Image 2", fontsize=11, fontweight="bold")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        # Разница
        diff = img2 - img1
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(
            diff,
            cmap="RdBu",
            aspect="equal",
            vmin=-max(abs(diff.min()), diff.max()),
            vmax=max(abs(diff.min()), diff.max()),
        )
        ax3.set_title(f"Difference\n(mean={metrics['mean_diff']:.4f})", fontsize=10)
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, shrink=0.8)

        # Абсолютная разница (тепловая карта)
        abs_diff = np.abs(diff)
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(abs_diff, cmap="hot", aspect="equal")
        ax4.set_title(f"Absolute Diff\n(max={metrics['max_diff']:.4f})", fontsize=10)
        ax4.axis("off")
        plt.colorbar(im4, ax=ax4, shrink=0.8)

        # 3D визуализация
        ax5 = fig.add_subplot(gs[1, 0], projection="3d")
        x = np.arange(img1.shape[1])
        y = np.arange(img1.shape[0])
        X, Y = np.meshgrid(x, y)
        ax5.plot_surface(X, Y, img1, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
        ax5.set_title("Image 1 (3D)", fontsize=10, fontweight="bold")
        ax5.set_xticks([])
        ax5.set_yticks([])

        ax6 = fig.add_subplot(gs[1, 1], projection="3d")
        ax6.plot_surface(X, Y, img2, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
        ax6.set_title("Image 2 (3D)", fontsize=10, fontweight="bold")
        ax6.set_xticks([])
        ax6.set_yticks([])

        ax7 = fig.add_subplot(gs[1, 2], projection="3d")
        ax7.plot_surface(X, Y, diff, cmap="RdBu", alpha=0.9, linewidth=0, antialiased=True)
        ax7.set_title("Difference (3D)", fontsize=10, fontweight="bold")
        ax7.set_xticks([])
        ax7.set_yticks([])

        # Гистограммы
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.hist(img1.flatten(), bins=50, alpha=0.6, label="Image 1", color="blue", density=True)
        ax8.hist(img2.flatten(), bins=50, alpha=0.6, label="Image 2", color="red", density=True)
        ax8.set_title("Intensity Distribution", fontsize=10, fontweight="bold")
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)

        # Профили интенсивности
        ax9 = fig.add_subplot(gs[2, :2])
        mid_y = img1.shape[0] // 2
        ax9.plot(img1[mid_y, :], label="Image 1", color="blue", linewidth=1.5)
        ax9.plot(img2[mid_y, :], label="Image 2", color="red", linewidth=1.5, linestyle="--")
        ax9.set_title(f"Horizontal Profile (row {mid_y})", fontsize=10, fontweight="bold")
        ax9.set_xlabel("X position")
        ax9.set_ylabel("Intensity")
        ax9.legend(fontsize=8)
        ax9.grid(True, alpha=0.3)

        # Корреляция
        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.scatter(img1.flatten()[::100], img2.flatten()[::100], alpha=0.3, s=5)
        ax10.plot([0, 1], [0, 1], "r--", linewidth=2)
        ax10.set_title(
            f"Pixel Correlation\n(Pearson: {metrics['pearson']:.4f})",
            fontsize=10,
            fontweight="bold",
        )
        ax10.set_xlabel("Image 1")
        ax10.set_ylabel("Image 2")
        ax10.grid(True, alpha=0.3)

        # Общий заголовок
        fig.suptitle(
            f"Surface Comparison: SSIM={metrics.get('ssim', 0):.4f} | "
            f"PSNR={metrics.get('psnr', 0):.2f} dB | "
            f"Similarity={metrics.get('similarity', 0):.4f}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.savefig(self.output_dir / f"{cid}_viz.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Сохранение карты разницы отдельно
        if save_difference_map and MATPLOTLIB_AVAILABLE:
            fig_diff, ax_diff = plt.subplots(figsize=(10, 8))
            im_diff = ax_diff.imshow(abs_diff, cmap="hot", aspect="equal")
            ax_diff.set_title(
                f"Absolute Difference Map\nMax: {metrics['max_diff']:.4f}, Mean: {metrics['mean_diff']:.4f}",
                fontsize=12,
                fontweight="bold",
            )
            ax_diff.axis("off")
            plt.colorbar(im_diff, ax=ax_diff, shrink=0.8)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{cid}_diffmap.png", dpi=150, bbox_inches="tight")
            plt.close()


def compare_surfaces(
    path1: str, path2: str, out: str = "output/surface_comparisons"
) -> Dict[str, Any]:
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

        img1 = (
            (base + np.random.randn(256, 256) * 0.05 - base.min()) / (base.max() - base.min()) * 255
        ).astype(np.uint8)
        img2 = (
            (base * 1.1 + np.random.randn(256, 256) * 0.07 - base.min())
            / (base.max() - base.min())
            * 255
        ).astype(np.uint8)

        Image.fromarray(img1).save(test_dir / "s1.png")
        Image.fromarray(img2).save(test_dir / "s2.png")

        r = compare_surfaces(str(test_dir / "s1.png"), str(test_dir / "s2.png"))
        print(f"SSIM: {r.get('ssim', 0):.4f}")
        print(f"Similarity: {r.get('similarity', 0):.4f}")
        print(f"Mean Diff: {r.get('mean_diff', 0):.6f}")
        print("Test completed!")
