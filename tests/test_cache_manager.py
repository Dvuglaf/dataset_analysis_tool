import numpy as np
import pytest
from pathlib import Path

from cache_manager import CacheManager
from computations.metrics.base import Metric
from custom_types import SceneSegment


# ================= Fixtures =================
@pytest.fixture
def temp_video(tmp_path: Path) -> Path:
    path = tmp_path / "test_video.mp4"
    path.write_bytes(b"fake_video_data_12345")
    return path


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def manager(cache_dir: Path):
    return CacheManager(cache_dir)


@pytest.fixture
def dummy_scene():
    return SceneSegment(
        start_frame=0,
        end_frame=100,
        label="scene_1",
        start_time_s=0.0,
        end_time_s=4.0
    )


@pytest.fixture
def simple_metric():
    return Metric(
        name="sharpness",
        value=0.85,
        raw_data=np.array([[0.1, 0.2], [0.3, 0.4]])
    )


@pytest.fixture
def array_metric():
    return Metric(
        name="motion",
        value=np.array([0.1, 0.2]),
        raw_data=np.array([[0.1, 0.2], [0.3, 0.4]])
    )


# ================= Tests =================
def test_video_hash(manager, temp_video):
    h1 = manager._get_video_hash(temp_video)
    h2 = manager._get_video_hash(temp_video)

    assert h1 == h2
    assert len(h1) == 64


def test_empty_cache_returns_none(manager, temp_video):
    result = manager.get_video_cached_metric(temp_video,"sharpness")
    assert result is None


def test_save_and_load_video_metric(manager, temp_video, simple_metric):
    manager.save_video_metrics_to_cache(temp_video, [simple_metric])
    result = manager.get_video_cached_metric(temp_video, "sharpness")

    assert result is not None
    assert result.name == "sharpness"
    assert result.value == 0.85


def test_save_and_load_video_numpy_metric(manager, temp_video, array_metric):
    manager.save_video_metrics_to_cache(temp_video, [array_metric])
    result = manager.get_video_cached_metric(temp_video, "motion")

    assert result is not None
    np.testing.assert_array_equal(result.value, array_metric.value)


def test_scene_cache_not_exists(manager, temp_video, dummy_scene):
    result = manager.get_scene_cached_metrics(temp_video, dummy_scene, "sharpness")

    assert result is None


def test_save_and_load_scene_metric(manager, temp_video, dummy_scene, simple_metric):
    manager.save_scene_metrics_to_cache(temp_video, dummy_scene, [simple_metric])

    result = manager.get_scene_cached_metrics(temp_video, dummy_scene, "sharpness")

    assert result is not None
    assert result.name == "sharpness"
    assert result.value == 0.85


def test_multiple_metrics(manager, temp_video):

    m1 = Metric(name="m1", value=1.0, raw_data=np.array([1]))
    m2 = Metric(name="m2", value=2.0, raw_data=np.array([2]))

    manager.save_video_metrics_to_cache(temp_video, [m1, m2])

    r1 = manager.get_video_cached_metric(temp_video, "m1")
    r2 = manager.get_video_cached_metric(temp_video, "m2")

    assert r1.value == 1.0
    assert r2.value == 2.0


def test_cache_files_created(manager, temp_video, array_metric, cache_dir):
    manager.save_video_metrics_to_cache(temp_video, [array_metric])
    files = list(cache_dir.iterdir())

    # json + npy
    assert len(files) >= 2


def test_restore_metric_value_array(manager, tmp_path):
    arr = np.array([1, 2, 3])
    file = tmp_path / "test.npy"
    np.save(file, arr)
    restored = manager._restore_metric_value(str(file))
    np.testing.assert_array_equal(restored, arr)


def test_restore_metric_value_float(manager):
    value = manager._restore_metric_value(1.23)
    assert value == 1.23


def test_scene_cache_filename(manager, temp_video, dummy_scene, simple_metric):
    manager.save_scene_metrics_to_cache(temp_video, dummy_scene, [simple_metric])

    h = manager._get_video_hash(temp_video)

    expected = f"{h}_{dummy_scene.start_frame}_{dummy_scene.end_frame}.json"
    files = [f.name for f in manager.path_to_cache_dir.iterdir()]

    assert expected in files
