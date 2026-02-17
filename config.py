from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox, QCheckBox
from scenedetect import ContentDetector, ThresholdDetector, AdaptiveDetector

from cache_manager import CacheManager
from metric_pipeline import MetricsProcessor
from widgets import ValueSlider


SCENES = {
    "pyscenedetect (Adaptive)": {
        "help": "Detects scene changes using PySceneDetect library (Adaptive Detector).",
        "object": AdaptiveDetector,
        "parameters": {
            "adaptive_threshold": {
                "type": float,
                "default": 3.0,
                "min": 0.0,
                "max": 100.0,
                "widget": ValueSlider,
                "help": "Threshold (float) that score ratio must exceed to trigger a new scene."
            },
            "min_scene_len": {
                "type": int,
                "default": 15,
                "min": 1,
                "max": 1000,
                "widget": ValueSlider,
                "help": "Minimum number of frames between scene cuts."
            },
            "window_width": {
                "type": int,
                "default": 2,
                "min": 1,
                "max": 1000,
                "widget": ValueSlider,
                "help": "Size of window (number of frames) before and after each frame to average together "
                        "in order to detect deviations from the mean."
            },
            "min_content_val": {
                "type": float,
                "default": 15.0,
                "min": 0.0,
                "max": 100.0,
                "widget": ValueSlider,
                "help": "Minimum threshold that the content_val must exceed in order to register as a new scene."
            }
        }
    },
    "pyscenedetect (Content)": {
        "help": "Detects scene changes using PySceneDetect library (Content Detector).",
        "object": ContentDetector,
        "parameters": {
            "threshold": {
                "type": float,
                "default": 27.0,
                "min": 0.0,
                "max": 100.0,
                "widget": ValueSlider,
                "help": "Threshold the average change in pixel intensity must exceed to trigger a cut."
            },
            "min_scene_len": {
                "type": int,
                "default": 15,
                "min": 1,
                "max": 1000,
                "widget": ValueSlider,
                "help": "Minimum number of frames between scene cuts."
            }
        }
    },
    "pyscenedetect (Threshold)": {
        "help": "Detects fast cuts/slow fades in from and out to a given threshold level. "
                "Detects both fast cuts and slow fades so long as an appropriate threshold is chosen "
                "(especially taking into account the minimum grey/black level).",
        "object": ThresholdDetector,
        "parameters": {
            "threshold": {
                "type": float,
                "default": 12,
                "min": 0.0,
                "max": 100.0,
                "widget": ValueSlider,
                "help": "8-bit intensity value that each pixel value (R, G, and B) "
                        "must be <= to in order to trigger a fade in/out."
            },
            "min_scene_len": {
                "type": int,
                "default": 15,
                "min": 1,
                "max": 1000,
                "widget": ValueSlider,
                "help": "Minimum number of frames between scene cuts."
            },
            "fade_bias": {
                "type": float,
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "widget": ValueSlider,
                "help": "Float between -1.0 and +1.0 representing the percentage of timecode skew "
                        "for the start of a scene (-1.0 causing a cut at the fade-to-black, "
                        "0.0 in the middle, "
                        "and +1.0 causing the cut to be right at the position where the threshold is passed)."
            },
            "add_final_scene": {
                "type": bool,
                "default": True,
                "widget": QCheckBox,
                "help": "Boolean indicating if the video ends on a "
                        "fade-out to generate an additional scene at this timecode."
            }
        }
    },
    "Basic fixed": {
        "help": "Split into fixed-length scenes based on a specified time interval.",
        "object": None,
        "parameters": {
            "scene_length": {
                "type": float,
                "default": 5.0,
                "min": 1.0,
                "max": 15.0,
                "widget": ValueSlider,
                "help": "Length of each scene in seconds."
            },
            "adjust_last_scene": {
                "type": bool,
                "default": True,
                "widget": QCheckBox,
                "help": "If enabled, adjusts the last scene to fit the remaining duration of the video."
            },
        }
    },
}


FILTERS = {
    "PSNR": {
        "help": "Removes duplicate frames based on Peak Signal-to-Noise Ratio (PSNR) metric.",
        "parameters": {
            "threshold": {
                "type": float,
                "default": 30.0,
                "min": 0.0,
                "max": 100.0,
                "widget": QDoubleSpinBox,
                "help": "PSNR threshold below which frames are considered duplicates."
            }
        }
    },
    "Fade": {
        "help": "Detects and removes fade-in and fade-out frames from the video.",
        "parameters": {
            "slope_threshold": {
                "type": float,
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
                "widget": QDoubleSpinBox,
                "help": "Slope threshold to identify fade transitions."
            },
            "min_duration": {
                "type": int,
                "default": 5,
                "min": 1,
                "max": 100,
                "widget": QSpinBox,
                "help": "Minimum duration (in frames) of fade to be considered for removal."
            },
            "smooth_window": {
                "type": int,
                "default": 6,
                "min": 1,
                "max": 64,
                "widget": QSpinBox,
                "help": "Size of the smoothing window applied to detect fades."
            }
        }
    },
    "Brightness": {
        "help": "Filters out frames that are too dark or too bright based on brightness levels.",
        "parameters": {
            "min_brightness": {
                "type": float,
                "default": 0.2,
                "min": 0.0,
                "max": 1.0,
                "widget": QDoubleSpinBox,
                "help": "Minimum brightness level below which frames are removed."
            },
            "max_brightness": {
                "type": float,
                "default": 0.8,
                "min": 0.0,
                "max": 1.0,
                "widget": QDoubleSpinBox,
                "help": "Maximum brightness level above which frames are removed."
            }
        }
    }
}

cache_manager = CacheManager(".analyzer_cache/")
