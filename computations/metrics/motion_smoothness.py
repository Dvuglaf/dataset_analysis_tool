import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from computations.metrics.base import BaseMetric, Metric, VideoSegment
from computations.third_party.amt.utils.build_utils import build_from_cfg
from computations.third_party.amt.utils.utils import InputPadder


class FrameProcess:
    def extract_frame(self, frame_list, start_from=0):
        return frame_list[start_from::2]


class MotionSmoothness:
    def __init__(self, config, ckpt, device, batch_size=256, target_height: int = 480):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.target_height = target_height
        self.initialization()
        self.load_model()

    def load_model(self):
        cfg_path = self.config
        ckpt_path = self.ckpt
        network_cfg = OmegaConf.load(cfg_path).network
        network_name = network_cfg.name
        print(f'Loading [{network_name}] from [{ckpt_path}]...')
        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.device), weights_only=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def initialization(self):
        if "cuda" in self.device:
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024 ** 2
            self.anchor_memory_bias = 2500 * 1024 ** 2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            # Do not resize in cpu mode
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(1 / 2).float().view(1, 1, 1, 1).to(self.device)
        self.fp = FrameProcess()

    def _resize_frames(self, frames):
        if not frames:
            raise ValueError("Input frames list is empty")

        _, h, w = frames[0].shape
        if h <= self.target_height:
            return torch.tensor(np.array(frames)).to(self.device), h, w

        scale = self.target_height / h
        new_w = max(1, int(round(w * scale)))

        tensor = torch.tensor(np.array(frames)).to(self.device)
        tensor = F.interpolate(
            tensor,
            size=(self.target_height, new_w),
            mode='bilinear',
            align_corners=False
        )

        return tensor, self.target_height, new_w

    def motion_score_batch(self, frames):
        iters = int(self.niters)

        inputs, h, w = self._resize_frames(self.fp.extract_frame(frames, start_from=0))
        inputs = inputs.div(255.0)
        assert len(inputs) > 1, f"The number of input should be more than one (current {len(inputs)})"
        # inputs = check_dim_and_resize(inputs)
        scale = self.anchor_resolution / (h * w) * np.sqrt(
            (self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        if scale < 1:
            print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(inputs)

        pair = inputs[:-1], inputs[1:]

        in_0 = pair[0].to(self.device)
        in_1 = pair[1].to(self.device)

        embt = torch.ones(in_0.shape[0], 1, 1, 1).to(self.device) * 1 / 2

        with torch.no_grad():
            imgt_pred = self.model(in_0, in_1, embt, scale_factor=scale, eval=True)['imgt_pred']

        # -----------------------  cal_vfi_score -----------------------
        outputs: torch.Tensor = padder.unpad(imgt_pred)
        outputs = outputs.clamp(0.0, 1.0)

        original_frames, _, _ = self._resize_frames(self.fp.extract_frame(frames, start_from=1))
        original_frames = original_frames.div(255.0)
        original_frames = original_frames.clamp(0.0, 1.0)
        if len(frames) % 2 == 0:
            original_frames = original_frames[:-1]

        vfi_score = self.vfi_score(original_frames, outputs)
        norm = (255.0 - vfi_score) / 255.0
        return norm

    def vfi_score(self, ori_frames, interpolate_frames):
        diff = torch.abs(ori_frames - interpolate_frames).mean().item()
        return diff


class MotionSmoothnessMetrics(BaseMetric):
    """Метрика Motion Smoothness"""

    def __init__(self, model, batch_size: int, device_id: str = "cpu", target_height: int = 480):
        self.model = model
        self.batch_size: int = batch_size
        self.device_id: str = device_id
        self.target_height = target_height

    @property
    def name(self) -> str:
        return "motion_smoothness"

    def _ensure_model(self) -> None:
        pass

    def _process_batch(self, ready_frames: list[torch.Tensor], _) -> float:
        return self.model.motion_score_batch(ready_frames)
