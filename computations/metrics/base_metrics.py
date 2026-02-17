import cv2
import numpy as np

from computations.metrics.base import Metric


def get_brightness_metric(gray_frames) -> Metric:
    per_frame_metric_value = np.array([float(frame.mean()) for frame in gray_frames])
    return Metric(name="brightness", value=float(per_frame_metric_value.mean()), raw_data=per_frame_metric_value)


def get_contrast_metric(gray_frames) -> Metric:
    per_frame_metric_value = np.array([float(frame.std()) for frame in gray_frames])
    return Metric(name="contrast", value=float(per_frame_metric_value.mean()), raw_data=per_frame_metric_value)


def get_saturation_metric(hsv_frames) -> Metric:
    per_frame_metric_value = np.array([float(frame[:, :, 1].mean()) for frame in hsv_frames])
    return Metric(name="saturation", value=float(per_frame_metric_value.mean()), raw_data=per_frame_metric_value)


def get_blur_metric(gray_frames) -> Metric:
    per_frame_metric_value = np.array([float(cv2.Laplacian(frame, cv2.CV_64F).var()) for frame in gray_frames])
    return Metric(name="blur", value=float(per_frame_metric_value.mean()), raw_data=per_frame_metric_value)


def get_artifacts_metric(gray_frames) -> Metric:
    per_frame_metric_value = []
    for frame in gray_frames:
        _, artifacts = cv2.threshold(frame, 250, 255, cv2.THRESH_BINARY)
        artifacts_ratio = np.sum(artifacts) / (255 * artifacts.size)
        per_frame_metric_value.append(float(artifacts_ratio))

    per_frame_metric_value = np.array(per_frame_metric_value)
    return Metric(name="artifacts", value=float(per_frame_metric_value.mean()), raw_data=per_frame_metric_value)


def get_optical_flow_metric(gray_frames: list[np.ndarray]):
    flows = []
    prev = gray_frames[0]
    for i in range(1, len(gray_frames)):
        cur = gray_frames[i]
        flow = cv2.calcOpticalFlowFarneback(  # noqa
            prev, cur,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=1,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        flows.append(flow)
        prev = cur

    flows = np.array(flows)
    dx = flows[..., 0]
    dy = flows[..., 1]
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    mean_magnitude = float(np.mean(magnitude))

    return Metric(name="optical_flow", value=mean_magnitude, raw_data=np.mean(np.array(flows), axis=0))


def get_temporal_clip_consintency(clip_features):
    pass
