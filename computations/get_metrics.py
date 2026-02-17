from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from pyiqa.archs.musiq_arch import MUSIQ
from torchvision.transforms import InterpolationMode, Resize, CenterCrop
import torch.nn.functional as F

from easydict import EasyDict as edict

from third_party.amt.utils.utils import InputPadder
from third_party.amt.utils.build_utils import build_from_cfg


THIRD_PARTY_DIR = Path("third_party/")
DEVICE = "cpu"
RESIZE_HEIGHT = 720
IMAGE_QUALITY_MODEL_PATH = THIRD_PARTY_DIR / "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
OPTICAL_FLOW_MODEL_PATH = THIRD_PARTY_DIR / "raft_model/models/raft-things.pth"
MOTION_SMOOTHNESS_CFG = THIRD_PARTY_DIR / "amt/cfgs/AMT-S.yaml"
MOTION_SMOOTHNESS_MODEL_PATH = THIRD_PARTY_DIR / "amt_model/amt-s.pth"


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
            _, flow_up = self.model(images1, images2, iters=12, test_mode=True)

            scores = self.get_score_gpu(images1, flow_up)

            flow_up_mean = torch.mean(flow_up, dim=0)

            return scores, flow_up_mean.cpu().numpy()


def rgb_to_luma(frames):
    r, g, b = frames[:, 0], frames[:, 1], frames[:, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).to(torch.uint8)


def musiq_transform(images, preprocess_mode='longer', max_size: int = 512):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h, w) > max_size:
            scale = max_size / min(h, w)
            images = Resize(size=(int(scale * h), int(scale * w)))(images)
            if preprocess_mode == 'shorter_centercrop':
                images = CenterCrop(max_size)(images)
    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            images = Resize(size=(int(scale * h), int(scale * w)))(images)
    elif preprocess_mode == 'None':
        return images / 255.
    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.

def motion_smoothness_score(frames, model):
    def resize_frames(frames, target_height = 480):
        h, w = frames[0].shape[:2]
        if h <= target_height:
            tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float().to(DEVICE)
            return tensor, h, w

        scale = target_height / h
        new_w = max(1, int(round(w * scale)))

        tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float().to(DEVICE)
        tensor = F.interpolate(
            tensor,
            size=(target_height, new_w),
            mode='bilinear',
            align_corners=False
        )

        return tensor, target_height, new_w

    anchor_resolution = 8192 * 8192
    anchor_memory = 1
    anchor_memory_bias = 0
    vram_avail = 1

    inputs, h, w = resize_frames(frames)
    inputs = inputs.div(255.0)
    assert len(inputs) > 1, f"The number of input should be more than one (current {len(inputs)})"

    scale = anchor_resolution / (h * w) * np.sqrt(
        (vram_avail - anchor_memory_bias) / anchor_memory)
    scale = 1 if scale > 1 else scale
    scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
    if scale < 1:
        print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
    padding = int(16 / scale)
    padder = InputPadder(inputs[0].shape, padding)
    inputs = padder.pad(inputs)

    pair = inputs[:-1], inputs[1:]

    in_0 = pair[0].to(DEVICE)
    in_1 = pair[1].to(DEVICE)

    # embt = torch.tensor(1 / 2).float().view(in_0.shape[0], 1, 1, 1).to(DEVICE)
    embt = torch.ones(in_0.shape[0], 1, 1, 1).to(DEVICE) * 1 / 2

    with torch.no_grad():
        imgt_pred = model(in_0, in_1, embt, scale_factor=scale, eval=True)['imgt_pred']


    # -----------------------  cal_vfi_score -----------------------
    outputs: torch.Tensor = padder.unpad(imgt_pred)
    outputs = outputs.clamp(0.0, 1.0)

    original_frames, _, _ = resize_frames(frames)
    original_frames = original_frames.div(255.0)
    original_frames = original_frames.clamp(0.0, 1.0)
    if len(frames) % 2 == 0:
        original_frames = original_frames[:-1]

    vfi_score = get_vfi_score(original_frames, outputs)
    norm = (255.0 - vfi_score) / 255.0
    return norm


def get_vfi_score(ori_frames, interpolate_frames):
    diff = torch.abs(ori_frames - interpolate_frames).mean().item()
    return diff

"""
    (1) Video Quality
"""


def get_entropy_metric(
        frames: np.ndarray,
        bins: int = 256
) -> np.ndarray:
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

    resize_transform = Resize(size=RESIZE_HEIGHT, interpolation=InterpolationMode.BICUBIC)

    if bins <= 0 or bins > 256:
        raise ValueError(f"Bins must be in range (0, 256], got {bins}")

    # Convert list to numpy array - raises ValueError if shapes inconsistent
    try:
        frames_array = np.array(frames)
    except ValueError as e:
        raise ValueError(f"Frames have inconsistent shapes: {e}") from e

    batch_size, height, width, channels = frames_array.shape

    # Efficient resizing: batch operation instead of loop
    if height > RESIZE_HEIGHT:
        # Convert to tensor format [B, C, H, W] for batch resizing
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)
        # Resize entire batch at once (much faster than per-frame)
        frames_tensor = resize_transform(frames_tensor)
        frames_tensor = frames_tensor.to(DEVICE, non_blocking=True)
    else:
        # Direct tensor conversion without resizing
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).to(
            DEVICE, non_blocking=True
        )

    # Convert RGB to luminance (grayscale) - reduces data while preserving structure
    luma_frames = rgb_to_luma(frames_tensor)  # [B, H, W]

    # Fully vectorized entropy calculation for entire batch (no loops!)
    batch_size = luma_frames.size(0)
    num_pixels = luma_frames.size(1) * luma_frames.size(2)

    # Flatten spatial dimensions: [B, H, W] -> [B, H*W]
    flattened = luma_frames.reshape(batch_size, -1).long()  # [B, num_pixels]

    # Create batch indices for scatter operations: [B, num_pixels]
    batch_indices = torch.arange(batch_size, device=DEVICE).unsqueeze(1).expand(-1, num_pixels)

    # Flatten everything for single scatter_add operation
    flat_batch_idx = batch_indices.reshape(-1)  # [B * num_pixels]
    flat_pixel_values = flattened.reshape(-1)  # [B * num_pixels]

    # Combined indices: batch_idx * bins + pixel_value
    combined_indices = flat_batch_idx * bins + flat_pixel_values
    ones = torch.ones_like(combined_indices, dtype=torch.float32)

    # Single scatter_add for all frames at once
    flat_histograms = torch.zeros(batch_size * bins, device=DEVICE, dtype=torch.float32)
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
    return np.array(entropies_tensor.cpu())


def get_image_quality_metric(
        frames: np.array,
        max_size: int = 512,
        model_path: str | Path | None = "/Users/dvuglaf/.cache/torch/hub/pyiqa/musiq_ava_ckpt-e8d3f067.pth"
) -> np.ndarray:
    if model_path is None:
        model = MUSIQ().to(DEVICE)
    else:
        model = MUSIQ(pretrained_model_path=model_path).to(DEVICE)
    model.eval()
    frames = np.array(frames, dtype=np.float32)
    tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
    tensor = musiq_transform(images=tensor, max_size=max_size)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            tensor = tensor.to(DEVICE, non_blocking=True)
            quality = model(tensor).detach().cpu().tolist()

    return np.array(quality)


"""
    (2) Motion & Temporal Coherence
"""


# def get_optical_flow_metric(
#         frames: np.ndarray,
#         model_path: str | Path = OPTICAL_FLOW_MODEL_PATH
# ) -> np.ndarray:
#     raft_cfg = edict({
#         "model": str(model_path),
#         "small": False,
#         "mixed_precision": False,
#         "alternate_corr": False
#     })
#     model = DynamicDegree(args=raft_cfg, device=DEVICE, resize_height=RESIZE_HEIGHT)
#     scores, _ = model.infer_from_batch(frames)
#     return np.array(scores)


def get_optical_flow_metric(
        frames: np.ndarray
):
    flows = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        cur = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(  # noqa
            prev, cur,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=1,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        flows.append(flow)
        prev = cur

    return np.array(flows)


def get_motion_smoothness_metric(
        frames: np.ndarray,
        cfg_path: str | Path = MOTION_SMOOTHNESS_CFG,
        model_path: str | Path = MOTION_SMOOTHNESS_MODEL_PATH,
):
    network_cfg = OmegaConf.load(cfg_path).network
    network_name = network_cfg.name
    model = build_from_cfg(network_cfg)
    ckpt = torch.load(model_path, map_location=torch.device(DEVICE), weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return motion_smoothness_score(frames, model)
