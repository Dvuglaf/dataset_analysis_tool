import traceback
from pathlib import Path
from typing import List, Any

import clip
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

from computations.metrics.base import BaseMetric, Metric, VideoSegment


def clip_transform(n_px, resize_size: int = 256):
    return Compose([
        Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class ClipFeaturesMetric(BaseMetric):
    """Метрика TemporalClipConsistency"""

    def __init__(self, model, batch_size: int, device_id: str = "cpu", resize_size: int = 256, crop_size: int = 224):
        self.batch_size: int = batch_size
        self.device_id = device_id
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.transform = clip_transform(self.crop_size, self.resize_size)
        self.model = model

    @property
    def name(self) -> str:
        return "temporal_clip_consistency"

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model, _ = clip.load("ViT-B/32", device=self.device_id)

    def _process_batch(self, batch: list[np.ndarray], _) -> Any:
        tensor = torch.from_numpy(np.array(batch))

        with torch.no_grad():
            tensor = self.transform(tensor)
            tensor = tensor.to(self.device_id)
            image_features = self.model.encode_image(tensor).detach().cpu().numpy()

        return image_features

    def _aggregate_results(self, all_results: List[Any]) -> np.ndarray:
        return np.array(all_results)
