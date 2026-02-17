from pathlib import Path

import cv2
import numpy as np
import torch


def numpy_to_tensor_fast(frames: list[np.ndarray] | np.ndarray, device: str):
    """
    Быстрое преобразование numpy в tensor с pinned memory
    """
    if type(frames) is list:
        frames = np.array(frames)

    tensor = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    if "cuda" in device:
        return tensor.pin_memory().to(device, non_blocking=True)
    else:
        return tensor.to(device)


def frame_generator(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
    finally:
        cap.release()

def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    return fps
