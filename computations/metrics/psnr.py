import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode


from computations.metrics.base import BaseMetric, Metric, VideoSegment


class PeakSignalToNoiseRationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        mse = torch.mean((x - y) ** 2, dim=(1, 2, 3))
        mse = torch.clamp(mse, min=1e-10)
        return -10 * torch.log10(mse)


class PSNRMetrics(BaseMetric):
    """Метрика PSNR"""

    def __init__(self, batch_size: int, device_id: str = "cpu", resize_height: int = 720):
        self.batch_size: int = batch_size
        self.device_id: str = device_id
        self.resize_height = resize_height
        self.psnr_metric = PeakSignalToNoiseRationModule().to(self.device_id)
        self.resize = Resize(size=self.resize_height, interpolation=InterpolationMode.BICUBIC)

    @property
    def name(self) -> str:
        return "psnr"

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
            psnr_batch = self.psnr_metric(batch_first_t, batch_second_t)

        return np.array(psnr_batch, dtype=np.float32)
