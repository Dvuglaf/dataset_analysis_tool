from pathlib import Path

import cv2
import numpy as np

from custom_types import VideoWrapper


def threshold_based_frame_processing(path_to_video: str | Path,
                                     threshold: float = 0.05,
                                     ) -> dict[int, int]:
    def get_mse_metric(a: np.ndarray, b: np.ndarray) -> float:
        return np.mean((a - b) ** 2) / (255.0 ** 2)  # noqa

    def find_drop_frames(video_wrapper: VideoWrapper):
        drop_indices = []
        mse_values = []
        prev_frame = video_wrapper[0]
        for i, curr_frame in enumerate(video_wrapper.frames):
            if i == 200:
                break
            if i == 0:
                continue
            dist = get_mse_metric(prev_frame, curr_frame)
            mse_values.append(dist)
            if dist < threshold:
                drop_indices.append(i)
            else:
                prev_frame = curr_frame

        return drop_indices, mse_values

    video = VideoWrapper(path_to_video)
    to_drop_indices, mse_values = find_drop_frames(video)

    return to_drop_indices, mse_values


def fade_frame_processing(path_to_video: str | Path,
                          slope_thresh: float = 1.0,
                          min_length: int = 5,
                          smooth_window: int = 6
                          ):
    video_wrapper = VideoWrapper(path_to_video)
    means = []
    for frame in video_wrapper.frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(np.mean(gray))

    if len(means) < 2:
        return []

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        means = np.convolve(means, kernel, mode='same')

    diffs = np.diff(means)

    def extract(mask):
        segs, start = [], None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            if not v and start is not None:
                if i - start >= min_length:
                    segs.append((start, i))
                start = None
        if start is not None and len(mask) - start >= min_length:
            segs.append((start, len(mask)))
        return segs

    inc = diffs >= slope_thresh
    dec = diffs <= -slope_thresh
    seg_in = extract(inc)
    seg_out = extract(dec)

    res = []
    for s, e in seg_in:
        res.append({'type': 'in', 'start': s, 'end': e})
    for s, e in seg_out:
        res.append({'type': 'out', 'start': s, 'end': e})
    return sorted(res, key=lambda x: x['start'])


def brightness_frame_processing(path_to_video: str | Path,
                                low_threshold: float,
                                high_threshold: float):
    video_wrapper = VideoWrapper(path_to_video)

    brightnesses = []
    for frame in video_wrapper.frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses.append(np.mean(frame))

    return np.mean(brightnesses) < low_threshold or np.mean(brightnesses) > high_threshold


if __name__ == "__main__":
    video_path = "/Users/dvuglaf/Downloads/forklift_eng_0.mp4"
    # result = fade_frame_processing(video_path, slope_thresh=1.0, min_length=5, smooth_window=6)
    result = brightness_frame_processing(video_path, low_threshold=50.0, high_threshold=200.0)
    print(result)
    # drop_indices, mse_values = threshold_based_frame_processing(video_path, threshold=0.05)
    # print(len(drop_indices), len(mse_values))
    # print(f"Frames to drop: {drop_indices}")
    # print(f"MAE values: {[i for i in range(len(mse_values)) if mse_values[i] < 0.05]}")
