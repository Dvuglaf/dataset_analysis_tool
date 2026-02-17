from pathlib import Path
from typing import List, Any

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

from metrics.base import BaseMetric, Metric, VideoSegment


def dino_transform_image(resize_size: int = 256, crop_size: int = 224):
    return Compose([
        Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        CenterCrop(crop_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class TemporalDinoConsistency(BaseMetric):
    """Метрика TemporalDinoConsistency"""

    def __init__(self, batch_size: int, device_id: str = "cpu", resize_size: int = 256, crop_size: int = 224):
        self.batch_size: int = batch_size
        self.device_id = device_id
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.transform = dino_transform_image(self.resize_size, self.crop_size)
        self.model = None

    @property
    def name(self) -> str:
        return "temporal_dino_consistency"

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = torch.hub.load(
                repo_or_dir='facebookresearch/dino:main',
                source='github',
                model='dino_vitb16'
            ).to(self.device_id)

    def _process_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        batch = np.array(batch, dtype=np.float32)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2) / 255

        with torch.no_grad():
            tensor = self.transform(tensor)
            tensor = tensor.to(self.device_id)
            image_features = self.model(tensor).detach().cpu().numpy()

        return image_features

    def _aggregate_results(self, all_results: List[Any]) -> np.ndarray:
        return np.array(all_results)

    def compute(self, video_path: Path) -> Metric:
        try:
            all_features = self._process_video_batched(video_path, min_batch_size=1)
            aggregated_features = self._aggregate_results(all_features)
            return Metric(name=self.name, value=aggregated_features, raw_data=all_features)
        except Exception as e:
            raise
