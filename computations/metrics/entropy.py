from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode

from computations.metrics.base import BaseMetric, Metric, VideoSegment


def rgb_to_luma(frames):
    r, g, b = frames[:, 0], frames[:, 1], frames[:, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).to(torch.uint8)


class EntropyMetrics(BaseMetric):
    """
    Entropy metric for video frames.

    Calculates Shannon entropy of frame luminance values to measure
    information content and visual complexity.
    """

    DEFAULT_BINS = 256

    def __init__(self, batch_size: int, device_id: str = "cpu", resize_height: int = 720):
        self.batch_size: int = batch_size
        self.device_id = device_id
        self.resize_height = resize_height
        self.resize = Resize(size=self.resize_height, interpolation=InterpolationMode.BICUBIC)

    @property
    def name(self) -> str:
        return "entropy"

    @torch.no_grad()
    def _calc_entropy(
        self,
        gray_frames: list[torch.Tensor],
        bins: int = DEFAULT_BINS,
        eps: float = 1e-8
    ) -> list[float]:
        """
        Calculate Shannon entropy for a batch of video frames.
        
        Entropy measures the amount of information/randomness in the image intensity
        distribution. Higher entropy indicates more complex/detailed frames.
        
        Args:
            frames: List of RGB frames as numpy arrays with shape [H, W, C]
            bins: Number of histogram bins (default: 256 for uint8 images)
            eps: Small value to avoid log(0), only used for numerical stability
            
        Returns:
            List of entropy values (in nats), one per frame
            
        Raises:
            ValueError: If frames list is empty, bins invalid, or frames have inconsistent shapes
        """
        # Validate inputs
        if bins <= 0 or bins > 256:
            raise ValueError(f"Bins must be in range (0, 256], got {bins}")

        luma_frames = torch.from_numpy(np.array(gray_frames))

        # Fully vectorized entropy calculation for entire batch (no loops!)
        batch_size = luma_frames.size(0)
        num_pixels = luma_frames.size(1) * luma_frames.size(2)
        
        # Flatten spatial dimensions: [B, H, W] -> [B, H*W]
        flattened = (luma_frames * (bins - 1)).clamp(0, bins - 1).long()
        
        # Create batch indices for scatter operations: [B, num_pixels]
        batch_indices = torch.arange(batch_size, device=self.device_id).unsqueeze(1).expand(-1, num_pixels)
        
        # Flatten everything for single scatter_add operation
        flat_batch_idx = batch_indices.reshape(-1)  # [B * num_pixels]
        flat_pixel_values = flattened.reshape(-1)   # [B * num_pixels]
        
        # Compute histograms for all frames with single vectorized scatter_add
        # Shape: [B, bins]
        histograms = torch.zeros(batch_size, bins, device=self.device_id, dtype=torch.float32)
        
        # Combined indices: batch_idx * bins + pixel_value
        combined_indices = flat_batch_idx * bins + flat_pixel_values
        ones = torch.ones_like(combined_indices, dtype=torch.float32)
        
        # Single scatter_add for all frames at once
        flat_histograms = torch.zeros(batch_size * bins, device=self.device_id, dtype=torch.float32)
        flat_histograms.scatter_add_(0, combined_indices, ones)
        histograms = flat_histograms.reshape(batch_size, bins)
        
        # Normalize to probability distributions: [B, bins]
        hist_sums = histograms.sum(dim=1, keepdim=True)  # [B, 1]
        
        # Handle edge case: replace zero sums with 1 to avoid division by zero
        hist_sums = torch.where(hist_sums > 0, hist_sums, torch.ones_like(hist_sums))
        prob_distributions = histograms / hist_sums  # [B, bins]
        
        # Calculate Shannon entropy: H = -sum(p * log(p)) - fully vectorized
        # Only compute for non-zero probabilities to avoid log(0)
        # Use where() to handle zeros without explicit masking
        safe_probs = torch.where(
            prob_distributions > 0,
            prob_distributions,
            torch.ones_like(prob_distributions)  # Replace zeros with 1 (log(1) = 0)
        )
        
        # Compute log only once for all non-zero values
        log_probs = torch.log(safe_probs)
        
        # Zero out contributions from zero probabilities
        log_probs = torch.where(prob_distributions > 0, log_probs, torch.zeros_like(log_probs))
        
        # Compute entropy for each frame: [B, bins] -> [B]
        entropies_tensor = -(prob_distributions * log_probs).sum(dim=1)
        
        # Convert to list and return
        return entropies_tensor.tolist()

    def _process_batch(self, batch: list[np.ndarray], _) -> list[float]:
        return self._calc_entropy(batch)
