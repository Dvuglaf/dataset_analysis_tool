import numpy as np

from computations.metrics.base import Metric


class BackgroundDominanceMetric:
    """
    Background dominance metric based on optical flow analysis.

    Computes ratio of median to 95th percentile motion magnitude,
    indicating how much of the motion is background vs foreground.
    """

    def __init__(self, patch_size: int = 224):
        """
        Args:
            patch_size: size of patches for analysis
        """
        self.patch_size = patch_size

    def compute(self, optical_flow: np.ndarray) -> Metric:
        """
        Calculate background dominance score from optical flow.

        Args:
            optical_flow: [N, 2, H, W] or [2, H, W] - optical flow array

        Returns:
            float: background dominance score (ratio of median to 95th percentile motion)
        """
        if optical_flow.ndim == 3:
            optical_flow = optical_flow[np.newaxis, ...]

        # Compute flow magnitude
        rad = (optical_flow[:, 0] ** 2 + optical_flow[:, 1] ** 2) ** 0.5  # (N, H, W)
        N, h, w = rad.shape

        patches_y = h // self.patch_size
        patches_x = w // self.patch_size

        if patches_y == 0 or patches_x == 0:
            # If image is smaller than patch_size, use the whole image
            p95 = np.percentile(rad, q=95, axis=(1, 2))
            p95 = np.where(p95 == 0, 1e-8, p95)
            scores = np.median(rad, axis=(1, 2)) / p95
            return Metric(name="background_dominance", value=float(np.mean(scores)), raw_data=scores)

        rad_cropped = rad[:, :patches_y * self.patch_size, :patches_x * self.patch_size]
        patches_rad = rad_cropped.reshape(
            N, patches_y, self.patch_size, patches_x, self.patch_size
        ).mean(axis=(2, 4))

        p95 = np.percentile(patches_rad, q=95, axis=(1, 2))
        p95 = np.where(p95 == 0, 1e-8, p95)
        scores = np.median(patches_rad, axis=(1, 2)) / p95

        return Metric(name="background_dominance", value=float(np.mean(scores)), raw_data=scores)
