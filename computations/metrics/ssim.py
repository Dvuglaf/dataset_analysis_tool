import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import Resize, InterpolationMode

from computations.metrics.base import BaseMetric, Metric, VideoSegment


class FastSSIMModule(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        mu_x = x.mean(dim=(2, 3))
        mu_y = y.mean(dim=(2, 3))

        var_x = x.var(dim=(2, 3), unbiased=False)
        var_y = y.var(dim=(2, 3), unbiased=False)

        cov = ((x - mu_x[:, :, None, None]) *
               (y - mu_y[:, :, None, None])).mean(dim=(2, 3))

        C1 = 1e-4
        C2 = 9e-4

        ssim = (
            (2*mu_x*mu_y + C1) * (2*cov + C2)
            / ((mu_x**2 + mu_y**2 + C1) *
               (var_x + var_y + C2) + self.eps)
        )

        return ssim.mean(dim=1)


class SSIMMetrics(BaseMetric):
    """Метрика SSIM"""

    def __init__(self, batch_size: int, device_id: str = "cpu", resize_height: int = 720):
        self.batch_size: int = batch_size
        self.device_id: str = device_id
        self.resize_height = resize_height
        self.ssim_metric = FastSSIMModule().to(self.device_id)
        self.resize = Resize(size=self.resize_height, interpolation=InterpolationMode.BICUBIC)

    @property
    def name(self) -> str:
        return "ssim"

    def _ensure_model(self) -> None:
        pass

    def _process_batch(self, batch: list[torch.Tensor], previous_batch: list[np.ndarray] | None = None) -> np.ndarray:
        if len(batch) < 2:
            return np.array([])

        batch = np.array(batch)

        batch_first = batch[:-1, ...]
        batch_second = batch[1:, ...]

        # [B, H, W, C] -> [B, C, H, W] и нормализация
        batch_first_t = torch.from_numpy(batch_first.astype(np.float32))
        batch_second_t = torch.from_numpy(batch_second.astype(np.float32))

        with torch.no_grad():
            ssim_batch = self.ssim_metric(batch_first_t, batch_second_t)

        return np.array(ssim_batch, dtype=np.float32)
