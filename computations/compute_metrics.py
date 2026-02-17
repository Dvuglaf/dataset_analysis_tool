import os
from pathlib import Path
from typing import Union

import numpy as np

from metrics.entropy import EntropyMetrics
from metrics.imaging_quality import ImagingQualityMetrics
from metrics.motion_smoothness import MotionSmoothnessMetrics
from metrics.optical_flow import OpticalFlowMetrics
from metrics.psnr import PSNRMetrics
from metrics.ssim import SSIMMetrics
from metrics.temporal_clip_consistency import TemporalClipConsistency
from metrics.temporal_dino_consistency import TemporalDinoConsistency
from metrics.VideoMAE import VideoMAEFeatureMetric
from metrics.background_dominance import BackgroundDominanceMetric
from metrics.ics_metrics import ICSMetric
from metrics.cluster_metrics import ClusterMetric
from metrics.base import VideoSegment

THIRD_PARTY_DIR = Path("./third_party")
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICE = "cpu"

IMAGING_QUALITY_MODEL_PATH = THIRD_PARTY_DIR / "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
MOTION_SMOOTHNESS_CFG = THIRD_PARTY_DIR / "amt/cfgs/AMT-S.yaml"
MOTION_SMOOTHNESS_MODEL_PATH = THIRD_PARTY_DIR / "amt_model/amt-s.pth"
OPTICAL_FLOW_MODEL_PATH = THIRD_PARTY_DIR / "raft_model/models/raft-things.pth"


# Cached metric instances
_entropy_metric = None
_psnr_metric = None
_ssim_metric = None
_imaging_quality_metric = None
_motion_smoothness_metric = None
_optical_flow_metric = None
_temporal_clip_consistency_metric = None
_temporal_dino_consistency_metric = None
_videomae_metric = None
_yoloe_metric = None
_cotracker_metric = None
_background_dominance_metric = None
_ics_metric = None
_cluster_metric = None


def get_entropy(
    video_segment: VideoSegment,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE
) -> np.ndarray:
    global _entropy_metric
    if _entropy_metric is None:
        _entropy_metric = EntropyMetrics(batch_size=batch_size, device_id=device)
    else:
        _entropy_metric.batch_size = batch_size
    return _entropy_metric.compute(video_segment)


def get_psnr(
    video_segment: VideoSegment,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE
) -> float:
    """
    PSNR: batch_size влияет только на размер подачи в модель,
    а количество значений в raw_data определяется только числом кадров сегмента.
    Для избежания проблем со старыми закешированными инстансами
    всегда создаём новый объект метрики.
    """
    metric = PSNRMetrics(batch_size=batch_size, device_id=device)
    return metric.compute(video_segment)


def get_ssim(
    video_segment: VideoSegment,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE
) -> float:
    """
    SSIM: batch_size влияет только на размер подачи в модель,
    а количество значений в raw_data определяется только числом кадров сегмента.
    Для избежания проблем со старыми закешированными инстансами
    всегда создаём новый объект метрики.
    """
    metric = SSIMMetrics(batch_size=batch_size, device_id=device)
    return metric.compute(video_segment)


def get_imaging_quality(
    video_segment: VideoSegment,
    model_path: str | Path = "musiq_ava_ckpt-e8d3f067.pth",
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE
) -> float:
    global _imaging_quality_metric
    if _imaging_quality_metric is None:
        _imaging_quality_metric = ImagingQualityMetrics(
            model_path=str(model_path),
            batch_size=batch_size,
            device_id=device
        )
    else:
        _imaging_quality_metric.batch_size = batch_size
    return _imaging_quality_metric.compute(video_segment)


def get_motion_smoothness(
    video_segment: VideoSegment,
    cfg: str | Path = MOTION_SMOOTHNESS_CFG,
    model_path: str | Path = MOTION_SMOOTHNESS_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE
) -> float:
    global _motion_smoothness_metric
    if _motion_smoothness_metric is None:
        _motion_smoothness_metric = MotionSmoothnessMetrics(
            cfg=str(cfg),
            model_path=str(model_path),
            batch_size=batch_size,
            device_id=device
        )
    else:
        _motion_smoothness_metric.batch_size = batch_size
    return _motion_smoothness_metric.compute(video_segment)


def get_optical_flow(
    video_segment: VideoSegment,
    model_path: str | Path = OPTICAL_FLOW_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mode: str = "optical_flow_mean",
    device: str = DEFAULT_DEVICE
) -> Union[float, np.ndarray]:
    from easydict import EasyDict as edict
    global _optical_flow_metric
    if _optical_flow_metric is None:
        raft_cfg = edict({
            "model": str(model_path),
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False
        })
        _optical_flow_metric = OpticalFlowMetrics(
            raft_cfg=raft_cfg,
            batch_size=batch_size,
            mode=mode,
            device_id=device
        )
    else:
        _optical_flow_metric.batch_size = batch_size
    return _optical_flow_metric.compute(video_segment)


def get_temporal_clip_consistency(
    video_segment: VideoSegment,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE
) -> np.ndarray:
    global _temporal_clip_consistency_metric
    if _temporal_clip_consistency_metric is None:
        _temporal_clip_consistency_metric = TemporalClipConsistency(
            batch_size=batch_size,
            device_id=device
        )
    else:
        _temporal_clip_consistency_metric.batch_size = batch_size
    return _temporal_clip_consistency_metric.compute(video_segment)


def get_temporal_dino_consistency(
    video_segment: VideoSegment,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = DEFAULT_DEVICE
) -> np.ndarray:
    global _temporal_dino_consistency_metric
    if _temporal_dino_consistency_metric is None:
        _temporal_dino_consistency_metric = TemporalDinoConsistency(
            batch_size=batch_size,
            device_id=device
        )
    else:
        _temporal_dino_consistency_metric.batch_size = batch_size
    return _temporal_dino_consistency_metric.compute(video_segment)


def get_videomae_features(
    video_segment: VideoSegment,
    model_path: str | Path,
    model_key: str = "finetune_videomaev2_giant_patch14_224",
    sequence_size: int = 16,
    batch_size: int = 8,
    resize_hw: tuple[int, int] = (224, 224),
    center_crop: bool = True,
    device: str = DEFAULT_DEVICE
) -> np.ndarray:
    global _videomae_metric
    if _videomae_metric is None:
        _videomae_metric = VideoMAEFeatureMetric(
            model_path=str(model_path),
            model_key=model_key,
            sequence_size=sequence_size,
            batch_size=batch_size,
            resize_hw=resize_hw,
            center_crop=center_crop,
            device_id=device
        )
    return _videomae_metric.compute(video_segment)


def get_cotracker(
    video_segment: VideoSegment,
    grid_size: int = 15,
    frame_interval: int = 1,
    device: str = DEFAULT_DEVICE
) -> np.ndarray:
    global _cotracker_metric
    if _cotracker_metric is None:
        _cotracker_metric = RealCoTrackerMetric(
            grid_size=grid_size,
            frame_interval=frame_interval,
            device_id=device
        )
    return _cotracker_metric.compute(video_segment)


def reset_cache():
    """Reset all cached metric instances."""
    global _entropy_metric, _psnr_metric, _ssim_metric
    global _imaging_quality_metric, _motion_smoothness_metric
    global _optical_flow_metric, _temporal_clip_consistency_metric
    global _temporal_dino_consistency_metric, _videomae_metric
    global _yoloe_metric, _cotracker_metric
    global _background_dominance_metric, _ics_metric, _cluster_metric

    _entropy_metric = None
    _psnr_metric = None
    _ssim_metric = None
    _imaging_quality_metric = None
    _motion_smoothness_metric = None
    _optical_flow_metric = None
    _temporal_clip_consistency_metric = None
    _temporal_dino_consistency_metric = None
    _videomae_metric = None
    _yoloe_metric = None
    _cotracker_metric = None
    _background_dominance_metric = None
    _ics_metric = None
    _cluster_metric = None


def get_background_dominance(
    optical_flow: np.ndarray,
    patch_size: int = 224
) -> float:
    """
    Calculate background dominance score from optical flow.

    Args:
        optical_flow: [N, 2, H, W] or [2, H, W] - optical flow array
        patch_size: size of patches for analysis

    Returns:
        float: background dominance score (ratio of median to 95th percentile motion)
    """
    global _background_dominance_metric
    if _background_dominance_metric is None or _background_dominance_metric.patch_size != patch_size:
        _background_dominance_metric = BackgroundDominanceMetric(patch_size=patch_size)
    return _background_dominance_metric.compute(optical_flow)


def get_ics_metrics(
    embeddings: np.ndarray,
    k: int = 10,
    n_segments: int = 30,
    similarity_threshold: float = 0.85,
    metric: str = 'cosine',
    normalize: bool = True,
    batch_size: int = 1000,
    device: str = DEFAULT_DEVICE
) -> dict:
    """
    Compute ICS (Inter-Class Similarity) metrics using k-NN.

    Args:
        embeddings: [N_videos, N_frames, d_feature] video embeddings
        k: number of nearest neighbors (if None, uses N-1)
        n_segments: number of segments to split each video into
        similarity_threshold: threshold for merging similar segments
        metric: distance metric ('cosine' or 'l2')
        normalize: whether to L2 normalize embeddings
        batch_size: batch size for GPU processing
        device: device for computation

    Returns:
        dict with ICS metrics:
            - mean_knn_dist: mean k-NN distance
            - min_observed: minimum observed distance
            - max_observed: maximum observed distance
            - total_segments: number of segments after processing
    """
    global _ics_metric
    _ics_metric = ICSMetric(
        k=k,
        n_segments=n_segments,
        similarity_threshold=similarity_threshold,
        metric=metric,
        normalize=normalize,
        batch_size=batch_size,
        device_id=device
    )
    return _ics_metric.compute(embeddings)


def get_cluster_metrics(
    embeddings: np.ndarray,
    eps: float = 5,
    min_samples: int = 5,
    n_segments: int = 30,
    merge_threshold: float = 0.85,
    metric: str = 'cosine',
    normalize: bool = True,
    batch_size: int = 1000,
    device: str = DEFAULT_DEVICE
) -> dict:
    """
    Compute cluster metrics using DBSCAN on GPU-computed distance matrix.

    Args:
        embeddings: [N_videos, N_frames, d_feature] video embeddings
        eps: DBSCAN eps parameter (max distance between neighbors)
        min_samples: DBSCAN min_samples parameter
        n_segments: number of segments to split each video into
        merge_threshold: threshold for merging similar segments
        metric: distance metric ('cosine' or 'l2')
        normalize: whether to L2 normalize embeddings (required for cosine)
        batch_size: batch size for GPU processing
        device: device for computation

    Returns:
        dict with cluster metrics:
            - mean_cluster_radius: mean radius of clusters
            - mean_inter_cluster_distance: mean distance between cluster centers
            - r_R_score: ratio of radius to inter-cluster distance
            - num_clusters: number of clusters found
            - noise_ratio: fraction of noise points
            - total_segments: number of segments after processing
            - eps: DBSCAN eps parameter used
            - min_samples: DBSCAN min_samples parameter used
    """
    global _cluster_metric
    _cluster_metric = ClusterMetric(
        eps=eps,
        min_samples=min_samples,
        n_segments=n_segments,
        merge_threshold=merge_threshold,
        normalize=normalize,
        batch_size=batch_size,
        device_id=device
    )
    return _cluster_metric.compute(embeddings)


if __name__ == "__main__":
    path_to_video = "/Users/dvuglaf/Downloads/2024.04.09_15.20.37.mov"
