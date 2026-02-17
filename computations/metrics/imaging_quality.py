import traceback
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from computations.metrics.base import BaseMetric, Metric, VideoSegment
from pyiqa.archs.musiq_arch import MUSIQ


def musiq_transform(images, preprocess_mode='longer', max_size: int = 512):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h, w) > max_size:
            scale = max_size / min(h, w)
            images = transforms.Resize(size=(int(scale * h), int(scale * w)))(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(max_size)(images)
    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            images = transforms.Resize(size=(int(scale * h), int(scale * w)))(images)
    elif preprocess_mode == 'None':
        return images / 255.
    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images


class ImagingQualityMetrics(BaseMetric):
    """Метрика ImagingQuality"""

    def __init__(self, model, batch_size: int, device_id: str = "cpu"):
        self.batch_size: int = batch_size
        self.device_id: str = device_id
        self.model = model

    @property
    def name(self) -> str:
        return "imaging_quality"

    def _process_batch(self, ready_frames: list[torch.Tensor], _) -> list[float]:
        ready_frames = np.array(ready_frames, dtype=np.float32)
        tensor = torch.tensor(ready_frames, dtype=torch.float32)
        tensor = musiq_transform(tensor, preprocess_mode='longer', max_size=512)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                tensor = tensor.to(self.device_id, non_blocking=True)
                quality = self.model(tensor).detach().cpu().tolist()

        return quality
