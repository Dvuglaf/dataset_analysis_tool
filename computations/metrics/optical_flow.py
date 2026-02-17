import traceback
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode

from metrics.base import BaseMetric, Metric, VideoSegment


class DynamicDegree:
    def __init__(self, args, device, resize_height: int = 720):
        from third_party.RAFT.core.raft import RAFT
        from third_party.RAFT.core.utils_core.utils import InputPadder
        self.args = args
        self.device = device
        self.resize_height = resize_height
        self.InputPadder = InputPadder

        self.model = RAFT(self.args)
        self.resize = Resize(size=self.resize_height, interpolation=InterpolationMode.BICUBIC)

        state = torch.load(self.args.model, map_location=self.device)
        if any(k.startswith("module.") for k in state.keys()):
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                name = k[7:] if k.startswith("module.") else k
                new_state[name] = v
            state = new_state
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def get_score_gpu(self, img, flo):
        scores = []
        for img, flo in zip(img, flo):
            img = img.permute(1,2,0)
            flo = flo.permute(1,2,0)
            u = flo[:,:,0]
            v = flo[:,:,1]
            rad = torch.sqrt(torch.square(u) + torch.square(v))
            h, w = rad.shape
            rad_flat = rad.flatten()
            cut_index = int(h*w*0.05)
            if cut_index == 0:
                cut_index = 1
            max_rad = torch.mean(torch.abs(torch.topk(rad_flat, cut_index).values)).item()

            scores.append(max_rad)
        return scores

    def infer_from_batch(self, frames):
        """Обработка батча пар фреймов"""

        if type(frames) is list:
            frames = np.array(frames)

        frames = torch.from_numpy(frames.astype(np.uint8)).permute(0, 3, 1, 2).float()

        B, C, H, W = frames.shape

        if H > self.resize_height:
            frames = self.resize(frames)

        frames = frames.to(self.device)
        pair = frames[:len(frames) - 1], frames[1:]

        with torch.no_grad():
            padder = self.InputPadder(frames[0].shape)

            images1, images2 = padder.pad(pair[0], pair[1])

            images1, images2 = images1.to(self.device), images2.to(self.device)
            _, flow_up = self.model(images1, images2, iters=20, test_mode=True)

            scores = self.get_score_gpu(images1, flow_up)

            flow_up_mean = torch.mean(flow_up, dim=0)

            return scores, flow_up_mean.cpu().numpy()


class OpticalFlowMetrics(BaseMetric):
    """Метрика OpticalFlow"""

    def __init__(self, raft_cfg, batch_size: int, mode: str = "optical_flow_mean", device_id: str = "cpu", resize_height: int = 720):
        self.raft_cfg = raft_cfg
        self.batch_size: int = batch_size
        self.mode = mode
        self.device_id = device_id
        self.resize_height = resize_height
        self.model = None
        self.interval = 3

    @property
    def name(self) -> str:
        return "optical_flow"

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = DynamicDegree(
                args=self.raft_cfg,
                device=self.device_id,
                resize_height=self.resize_height
            )

    # def _process_video(self, video_path: Path) -> tuple[list, list]:
    #     """Общая логика обработки видео для optical flow."""
    #     self._ensure_model()
    #
    #     all_scores = []
    #     all_flow = []
    #     batch = []
    #
    #     for idx, frame in enumerate(frame_generator(video_path)):
    #         if idx % self.interval == 0:
    #             batch.append(frame)
    #
    #         if len(batch) > 0 and (len(batch) % (self.batch_size + 1) == 0):
    #             scores, flow = self.model.infer_from_batch(batch)
    #             all_scores.extend(scores)
    #             all_flow.append(flow)
    #             batch.clear()
    #
    #     if len(batch) > 1:
    #         scores, flow = self.model.infer_from_batch(batch)
    #         all_scores.extend(scores)
    #         all_flow.append(flow)
    #
    #     return all_scores, all_flow

    def _aggregate_flow_results(self, all_scores: list, all_flow: list):
        """Агрегация результатов в зависимости от режима."""
        if self.mode == "score":
            return np.mean(all_scores).item()

        elif self.mode == "optical_flow":
            all_flow = [np.expand_dims(x, 0) for x in all_flow]
            metric_value = np.concatenate(all_flow, axis=0)

            block_size = 32
            num_flows = metric_value.shape[0]
            num_blocks = (num_flows + block_size - 1) // block_size

            averaged_blocks = []
            for i in range(num_blocks):
                start_idx = i * block_size
                end_idx = min((i + 1) * block_size, num_flows)
                block = metric_value[start_idx:end_idx]
                block_mean = np.mean(block, axis=0)
                averaged_blocks.append(np.expand_dims(block_mean, 0))

            return np.concatenate(averaged_blocks, axis=0)

        elif self.mode == "optical_flow_mean":
            all_flow = [np.expand_dims(x, 0) for x in all_flow]
            return np.concatenate(all_flow, axis=0).mean(axis=0)

        else:
            all_flow = [np.expand_dims(x, 0) for x in all_flow]
            flow_mean = np.concatenate(all_flow, axis=0).mean(axis=0)
            scores_mean = np.mean(all_scores).item()
            return np.array((flow_mean, scores_mean), dtype=object)

    def compute(self, video_path: Path) -> Metric:
        try:
            all_scores, all_flow = self._process_video(video_path)
            metric_value = self._aggregate_flow_results(all_scores, all_flow)

            is_array = self.mode != "score"
            return Metric(name=self.name, value=metric_value, raw_data=all_scores if self.mode == "score" else all_flow)

        except Exception as e:
            raise
