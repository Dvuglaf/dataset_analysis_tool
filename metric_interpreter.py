"""
Video Dataset Metric Interpreter v3
====================================
Two-level interpretation system:
  Level 1 — per-metric threshold categorization (hardcoded from benchmark analysis)
  Level 2 — composite cross-metric conclusions (actionable insights)

Thresholds calibrated against benchmark distributions:
  Kinetics-400, Something-Something v2, UCF101, HMDB51, Assembly101

Benchmark reference (means):
  ┌────────────────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
  │ Metric                         │ Assem101 │  HMDB51  │  Kin400  │  UCF101  │   SSv2   │
  ├────────────────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
  │ imaging_quality (MUSIQ)        │  59.17   │  62.84   │  47.11   │  52.90   │  46.80   │
  │ entropy                        │   4.68   │   5.02   │   4.66   │   4.75   │   4.62   │
  │ blur (Laplacian var)           │   99.8   │  486.3   │  577.3   │  754.3   │  466.6   │
  │ opticflow_score                │   0.43   │  29.85   │  12.26   │  14.41   │  10.02   │
  │ motion_smoothness              │   1.000  │   0.989  │   0.985  │   0.986  │   0.987  │
  │ psnr                           │  35.52   │  25.92   │  27.08   │  25.72   │  26.91   │
  │ ssim                           │   0.955  │   0.889  │   0.869  │   0.849  │   0.881  │
  │ clip_temporal_consistency      │   0.995  │   0.989  │   0.981  │   0.984  │   0.984  │
  │ dino_temporal_consistency      │   0.994  │   0.983  │   0.968  │   0.973  │   0.976  │
  │ background_dominance           │   0.277  │   0.403  │   0.515  │   0.509  │   0.417  │
  │ clip_ics                       │   0.158  │   0.381  │   0.377  │   0.384  │   0.260  │
  │ dino_ics                       │   0.413  │   0.765  │   0.786  │   0.757  │   0.774  │
  │ num_clusters                   │    18    │    53    │   (n/a)  │   100    │   159    │
  │ r_R_score                      │   0.086  │   0.106  │   (n/a)  │   0.090  │   0.091  │
  │ noise_ratio                    │   0.011  │   0.345  │   (n/a)  │   0.276  │   0.054  │
  │ duration_sec (track lifetime)  │   1.33   │   1.21   │   1.54   │   1.80   │   1.75   │
  │ persistence                    │   0.82   │   0.72   │   0.75   │   0.73   │   0.80   │
  │ norm_avg_velocity              │  0.0048  │  0.0102  │  0.0127  │  0.0114  │  0.0159  │
  │ norm_displacement              │  0.037   │  0.066   │  0.084   │  0.062   │  0.089   │
  │ frames_count                   │   46.5   │   22.7   │   26.9   │   27.9   │   15.7   │
  │ domain_concentration (top1 %)  │   77.0   │   31.1   │   25.0   │   52.6   │   34.6   │
  └────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Literal

import numpy as np

from computations.metrics.base import Metric


# Metrics can be list of objects with .name and .value, or dict[str, float]


# ──────────────────────────────────────────────────────────────────────
# Level 1: Threshold definitions (hardcoded)
# ──────────────────────────────────────────────────────────────────────

class Level(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ThresholdResult:
    metric_name: str
    display_name: str
    value: float
    level: Level
    label: str
    description: str
    low_upper: float
    high_lower: float


@dataclass
class ThresholdSpec:
    metric_name: str
    display_name: str
    subtitle: str | None
    low_upper: float
    high_lower: float
    label_low: str
    label_mid: str
    label_high: str
    desc_low: str
    desc_mid: str
    desc_high: str
    range: tuple[float, float]
    mode: Literal["direct", "inverse", "target"] = "direct"
    target: float = None  # Нужно только для mode="target"
    higher_is_better: Optional[bool] = None


# fmt: off
THRESHOLDS: dict[str, ThresholdSpec] = {
    # ── Axis 1: Visual Quality ──────────────────────────────────────
    # MUSIQ: benchmarks 46.8–62.8
    "imaging_quality": ThresholdSpec(
        metric_name="imaging_quality",
        display_name="Perceptual Quality (MUSIQ)",
        subtitle=None,
        low_upper=45.0,                          # below all benchmarks
        high_lower=60.0,                         # Assembly101/HMDB51 zone
        label_low="Low quality",
        label_mid="Moderate quality",
        label_high="High quality",
        desc_low="Below all standard benchmarks. Likely compression artifacts, "
                 "low resolution, or poor capture conditions.",
        desc_mid="Within the typical range of established video benchmarks "
                 "(Kinetics-400 / UCF101 level).",
        desc_high="Above most benchmarks. Clean capture, controlled conditions "
                  "(Assembly101 / HMDB51 level).",
        higher_is_better=True,
        range=(0.0, 100.0),
    ),
    # Entropy: benchmarks 4.62–5.02, tight range
    "entropy": ThresholdSpec(
        metric_name="entropy",
        display_name="Frame Entropy",
        subtitle=None,
        low_upper=4.5,                            # below all benchmarks
        high_lower=5.0,                            # only HMDB51 above
        label_low="Low entropy",
        label_mid="Moderate entropy",
        label_high="High entropy",
        desc_low="Visually uniform frames — flat backgrounds, low texture complexity. "
                 "May lack informative content for feature learning.",
        desc_mid="Typical intensity distribution complexity across benchmarks.",
        desc_high="Texture-rich scenes with high intensity variation. "
                  "Beneficial for learning discriminative visual features.",
        higher_is_better=True,
        range=(0.0, 7)
    ),
    # Blur (Laplacian variance): benchmarks 100–754 (higher = more high-freq detail)
    # Assembly101=100, SSv2=467, HMDB51=486, Kinetics=577, UCF101=754
    # NOTE: Low values can indicate either actual blur OR visually simple scenes
    "blur": ThresholdSpec(
        metric_name="blur",
        display_name="High-Frequency Detail",
        subtitle="Laplacian variance",
        low_upper=200.0,                           # below all main benchmarks
        high_lower=600.0,                           # Kinetics/UCF101 territory
        label_low="Low detail",
        label_mid="Moderate detail",
        label_high="High detail",
        desc_low="Low high-frequency content. Can indicate motion blur, defocus, "
                 "low resolution, OR visually simple scenes (uniform backgrounds, "
                 "smooth surfaces). Check entropy to disambiguate.",
        desc_mid="Typical high-frequency detail for video benchmarks "
                 "(SSv2 / HMDB51 / Kinetics-400 level).",
        desc_high="Rich high-frequency content — sharp edges, complex textures "
                  "(UCF101 level and above).",
        higher_is_better=True,
        range=(0.0, 1000.0),
    ),
    # ── Brightness: mean grayscale intensity [0..255] ───────────────
    "brightness": ThresholdSpec(
        metric_name="brightness",
        display_name="Frame Brightness",
        subtitle=None,
        low_upper=70.0,  # very dark
        high_lower=180.0,  # very bright
        label_low="Underexposed",
        label_mid="Well exposed",
        label_high="Overexposed",
        desc_low="Low average luminance. Frames appear dark or underexposed, "
                 "potentially hiding visual details.",
        desc_mid="Balanced exposure with sufficient visibility of scene content.",
        desc_high="High luminance values. Possible overexposure or washed-out highlights.",
        higher_is_better=None,
        range=(0.0, 255.0),
        mode="target",
        target=128.
    ),

    # ── Contrast: grayscale standard deviation ─────────────────────
    "contrast": ThresholdSpec(
        metric_name="contrast",
        display_name="Intensity Contrast",
        subtitle=None,
        low_upper=20.0,  # flat images
        high_lower=60.0,  # strong contrast
        label_low="Low contrast",
        label_mid="Moderate contrast",
        label_high="High contrast",
        desc_low="Low intensity variation. Frames appear flat or washed out, "
                 "with weak separation between objects and background.",
        desc_mid="Typical contrast levels supporting good object and texture visibility.",
        desc_high="Strong intensity variation. High visual separation and pronounced edges.",
        higher_is_better=None,
        range=(0.0, 128),
        mode="target",
        target=65.
    ),

    # ── Saturation: mean HSV S channel [0..255] ─────────────────────
    "saturation": ThresholdSpec(
        metric_name="saturation",
        display_name="Color Saturation",
        subtitle=None,
        low_upper=40.0,  # desaturated
        high_lower=140.0,  # vivid colors
        label_low="Desaturated",
        label_mid="Natural colors",
        label_high="Highly saturated",
        desc_low="Low color saturation. Frames appear grayscale-like or visually dull.",
        desc_mid="Balanced and natural color representation.",
        desc_high="Strongly saturated colors. May enhance visual appeal, "
                  "but can also indicate color distortion.",
        higher_is_better=None,
        range=(0.0, 255.0),
        mode="target",
        target=100.
    ),

    # ── Artifacts: bright pixel ratio [0..1] ────────────────────────
    "artifacts": ThresholdSpec(
        metric_name="artifacts",
        display_name="Artifacts",
        subtitle="Compression / Highlight artifacts",
        low_upper=0.01,  # almost clean
        high_lower=0.05,  # heavy artifacts
        label_low="Clean",
        label_mid="Minor artifacts",
        label_high="Severe artifacts",
        desc_low="Minimal presence of saturated or corrupted pixels. "
                 "Indicates clean visual signal.",
        desc_mid="Noticeable but limited artifacts. May result from compression "
                 "or sensor noise.",
        desc_high="High proportion of saturated or corrupted pixels. "
                  "Strong visual degradation likely affecting analysis quality.",
        higher_is_better=None,
        range=(0.0, 0.07),
        mode="inverse"
    ),

    # ── Axis 2: Motion & Temporal Coherence ─────────────────────────
    # Optical flow: benchmarks 0.43–29.8, huge spread
    "optical_flow": ThresholdSpec(
        metric_name="optical_flow",
        display_name="Optical Flow Magnitude",
        subtitle=None,
        low_upper=2.0,                            # Assembly101 zone (~0.4)
        high_lower=20.0,                          # above Kinetics/UCF/SSv2
        label_low="Minimal motion",
        label_mid="Moderate motion",
        label_high="High motion",
        desc_low="Near-static content. Models may not learn meaningful "
                 "temporal features from this data.",
        desc_mid="Typical dynamics for action recognition benchmarks "
                 "(SSv2 / Kinetics-400 / UCF101 range).",
        desc_high="Intensive dynamics — active manipulation or fast movement. "
                  "Requires models with strong temporal modeling.",
        higher_is_better=None,
        range=(0.0, 30)
    ),
    # Motion smoothness: benchmarks 0.985–1.000, very tight
    "motion_smoothness": ThresholdSpec(
        metric_name="motion_smoothness",
        display_name="Motion Smoothness",
        subtitle=None,
        low_upper=0.975,                           # below all benchmarks
        high_lower=0.995,                          # Assembly101 territory
        label_low="Low smoothness",
        label_mid="Moderate smoothness",
        label_high="High smoothness",
        desc_low="Jerky transitions — possible scene cuts, corrupted frames, "
                 "or abrupt camera movement. May also indicate high dynamics.",
        desc_mid="Typical transition quality across benchmarks.",
        desc_high="Very smooth inter-frame transitions. "
                  "Controlled or near-static recording conditions.",
        higher_is_better=True,
        range=(0.9, 1.0)
    ),
    # PSNR: benchmarks 25.7–35.5 (Assembly101 outlier at 35.5, rest 25.7–27.1)
    "psnr": ThresholdSpec(
        metric_name="psnr",
        display_name="Inter-frame PSNR",
        subtitle="Peak Signal-to-Noise Ratio",
        low_upper=24.0,                            # below all benchmarks
        high_lower=30.0,                            # Assembly101 territory
        label_low="Low PSNR",
        label_mid="Moderate PSNR",
        label_high="High PSNR",
        desc_low="Large pixel-level changes between frames — fast motion, "
                 "scene changes, or compression artifacts.",
        desc_mid="Typical inter-frame stability for action video.",
        desc_high="Minimal frame-to-frame pixel changes. "
                  "Static or near-static content.",
        higher_is_better=None,
        range=(0.0, 50.0),
        mode="inverse"
    ),
    # SSIM: benchmarks 0.849–0.955
    "ssim": ThresholdSpec(
        metric_name="ssim",
        display_name="Inter-frame SSIM",
        subtitle="Structural Similarity Index",
        low_upper=0.82,                            # below all benchmarks
        high_lower=0.93,                            # Assembly101 territory
        label_low="Low SSIM",
        label_mid="Moderate SSIM",
        label_high="High SSIM",
        desc_low="Significant structural changes between frames — "
                 "high dynamics or quality issues.",
        desc_mid="Typical structural consistency for video benchmarks.",
        desc_high="High structural frame-to-frame similarity. "
                  "Stable or near-static content.",
        higher_is_better=None,
        range=(-1.0, 1.0),
        mode="inverse"
    ),
    # DINO temporal consistency: benchmarks 0.968–0.994
    "temporary_consistency_dino_score": ThresholdSpec(
        metric_name="temporary_consistency_dino_score",
        display_name="DINO Temporal Consistency",
        subtitle=None,
        low_upper=0.96,                            # below all benchmarks
        high_lower=0.99,                            # Assembly101 territory
        label_low="Low consistency",
        label_mid="Moderate consistency",
        label_high="High consistency",
        desc_low="Structural feature disruptions between frames — "
                 "possible scene breaks, drastic view changes, or very high dynamics.",
        desc_mid="Standard temporal coherence range "
                 "(Kinetics-400 / UCF101 / SSv2 level).",
        desc_high="Near-constant structural features. Very stable, "
                  "low-dynamics scenes.",
        higher_is_better=None,
        range=(0.0, 1.0)
    ),
    # CLIP temporal consistency: benchmarks 0.981–0.995
    "temporary_consistency_clip_score": ThresholdSpec(
        metric_name="temporary_consistency_clip_score",
        display_name="CLIP Temporal Consistency",
        subtitle=None,
        low_upper=0.97,                            # below all benchmarks
        high_lower=0.99,                            # Assembly101 territory
        label_low="Low consistency",
        label_mid="Moderate consistency",
        label_high="High consistency",
        desc_low="Semantic content shifts between frames — "
                 "possible topic/context changes within clips.",
        desc_mid="Standard semantic stability across frames.",
        desc_high="Semantically static content — minimal scene-level variation.",
        higher_is_better=None,
        range=(0.0, 1.0)
    ),
    # Background dominance: benchmarks 0.277–0.515
    "background_dominance": ThresholdSpec(
        metric_name="background_dominance",
        display_name="Background Dominance",
        subtitle=None,
        low_upper=0.35,                            # Assembly101 zone
        high_lower=0.50,                            # Kinetics/UCF101 zone
        label_low="Object-centric",
        label_mid="Mixed motion",
        label_high="Camera-centric",
        desc_low="Motion is localized in objects, stable background. "
                 "Local attention / object-centric models are preferred.",
        desc_mid="Balanced contribution of camera and object motion.",
        desc_high="Global camera motion dominates. "
                  "Full-frame optical flow features may be more effective.",
        higher_is_better=None,
        range=(0.0, 1.0),
    ),

    # ── Axis 3: Diversity & Distinguishability ──────────────────────
    # CLIP ICS: benchmarks 0.158–0.384
    "clip_ics": ThresholdSpec(
        metric_name="clip_ics",
        display_name="Inter-Clip Diversity (CLIP)",
        subtitle=None,
        low_upper=0.25,                            # Assembly101/SSv2 zone
        high_lower=0.37,                            # HMDB51/Kinetics/UCF101 zone
        label_low="Low diversity",
        label_mid="Moderate diversity",
        label_high="High diversity",
        desc_low="Semantically repetitive content. "
                 "May need supplementation with external data.",
        desc_mid="Moderate semantic variety across clips.",
        desc_high="In-the-wild level diversity "
                  "(HMDB51 / Kinetics-400 / UCF101 level).",
        higher_is_better=True,
        range=(0.0, 1.0)
    ),
    # DINO ICS: benchmarks 0.413–0.786 (Assembly101 outlier at 0.413, rest 0.757–0.786)
    "dino_ics": ThresholdSpec(
        metric_name="dino_ics",
        display_name="Inter-Clip Diversity (DINO)",
        subtitle=None,
        low_upper=0.60,                            # Assembly101 clearly below
        high_lower=0.77,                            # top benchmarks
        label_low="Low diversity",
        label_mid="Moderate diversity",
        label_high="High diversity",
        desc_low="Visually homogeneous dataset — similar scenes/backgrounds.",
        desc_mid="Moderate visual variety across clips.",
        desc_high="High visual diversity comparable to in-the-wild benchmarks.",
        higher_is_better=True,
        range=(0.0, 1.0)
    ),
    # Num clusters: benchmarks 18–159
    "num_clusters": ThresholdSpec(
        metric_name="num_clusters",
        display_name="Action Taxonomy (clusters)",
        subtitle=None,
        low_upper=40.0,                            # Assembly101 zone (18)
        high_lower=130.0,                           # SSv2 zone (159)
        label_low="Narrow taxonomy",
        label_mid="Moderate taxonomy",
        label_high="Rich taxonomy",
        desc_low="Few distinct action categories detected. "
                 "Limited behavioral variety.",
        desc_mid="Moderate number of action clusters.",
        desc_high="Rich action taxonomy — diverse behavioral patterns.",
        higher_is_better=True,
        range=(0, 300)
    ),
    # r/R score: benchmarks 0.086–0.106, very tight range (0.02 spread)
    "r_R_score": ThresholdSpec(
        metric_name="r_R_score",
        display_name="Cluster Separability (r/R)",
        subtitle=None,
        low_upper=0.09,                            # best separation
        high_lower=0.11,                            # worst among benchmarks + margin
        label_low="Well-separated",
        label_mid="Moderate separation",
        label_high="Poorly separated",
        desc_low="Compact, well-separated action clusters — "
                 "actions are clearly distinguishable.",
        desc_mid="Typical cluster separability for video benchmarks.",
        desc_high="Overlapping clusters — action categories may be ambiguous. "
                  "Consider refining class definitions.",
        higher_is_better=False,
        range=(0, 1.0),
        mode="inverse"
    ),
    # Noise ratio: benchmarks 0.011–0.345, huge spread
    "noise_ratio": ThresholdSpec(
        metric_name="noise_ratio",
        display_name="Clustering Noise Ratio",
        subtitle=None,
        low_upper=0.06,                            # Assembly101/SSv2 zone
        high_lower=0.25,                            # UCF101/HMDB51 zone
        label_low="Low noise",
        label_mid="Moderate noise",
        label_high="High noise",
        desc_low="Most clips cluster well — clean dataset structure.",
        desc_mid="Some unclassifiable content present.",
        desc_high="Significant portion of clips don't fit any cluster. "
                  "Possible data quality issues or very heterogeneous content.",
        higher_is_better=False,
        range=(0.0, 1.0),
        mode="inverse"
    ),

    # ── Axis 4: Object Tracking (YOLO-based) ────────────────────────
    # duration_sec: benchmarks 1.21–1.80 (mean track lifetime in seconds)
    "duration_sec": ThresholdSpec(
        metric_name="duration_sec",
        display_name="Mean Track Lifetime (sec)",
        subtitle=None,
        low_upper=1.0,                            # below all benchmarks
        high_lower=2.0,                            # above all benchmarks
        label_low="Short tracks",
        label_mid="Moderate tracks",
        label_high="Long tracks",
        desc_low="Objects appear briefly — fast scene changes, small objects, "
                 "or tracking fragmentation.",
        desc_mid="Typical object persistence for action recognition benchmarks.",
        desc_high="Objects persist for extended periods — slow scenes, "
                  "long takes, or large dominant objects.",
        higher_is_better=None,
        range=(0.0, 5.0)
    ),
    # persistence: benchmarks 0.72–0.82 (detection continuity ratio)
    "persistence": ThresholdSpec(
        metric_name="persistence",
        display_name="Track Persistence",
        subtitle=None,
        low_upper=0.65,                            # below all benchmarks
        high_lower=0.85,                            # Assembly101 territory
        label_low="Low persistence",
        label_mid="Moderate persistence",
        label_high="High persistence",
        desc_low="Frequent detection gaps within tracks — occlusions, "
                 "motion blur causing missed detections, or low detector confidence.",
        desc_mid="Typical detection continuity for video benchmarks.",
        desc_high="Near-continuous tracking with minimal gaps. "
                  "Stable, clearly visible objects.",
        higher_is_better=True,
        range=(0.0, 1.0)
    ),
    # norm_avg_velocity: benchmarks 0.0048–0.0159 (Assembly101 outlier low)
    "norm_avg_velocity": ThresholdSpec(
        metric_name="norm_avg_velocity",
        display_name="Normalized Object Velocity",
        subtitle=None,
        low_upper=0.006,                           # Assembly101 zone
        high_lower=0.015,                           # SSv2 zone
        label_low="Slow objects",
        label_mid="Moderate object speed",
        label_high="Fast objects",
        desc_low="Objects move slowly relative to frame size. "
                 "Stationary or near-stationary object dynamics.",
        desc_mid="Typical object velocity for action recognition benchmarks.",
        desc_high="Fast object motion relative to frame. "
                  "May challenge tracking and temporal modeling.",
        higher_is_better=None,
        range=(0.0, 0.05)
    ),
    # norm_displacement: benchmarks 0.037–0.089 (Assembly101 outlier low)
    "norm_displacement_v": ThresholdSpec(
        metric_name="norm_displacement_v",
        display_name="Normalized Object Displacement",
        subtitle=None,
        low_upper=0.05,                            # Assembly101 + lower HMDB/UCF zone
        high_lower=0.10,                            # above most benchmarks
        label_low="Low displacement",
        label_mid="Moderate displacement",
        label_high="High displacement",
        desc_low="Objects stay near their origin — localized actions, "
                 "stationary camera with in-place manipulation.",
        desc_mid="Typical spatial displacement for action video.",
        desc_high="Objects traverse significant screen area. "
                  "Large-scale movement or camera tracking.",
        higher_is_better=None,
        range=(0.0, 0.2)
    ),
    # frames_count: benchmarks 15.7–46.5 (Assembly101 outlier high due to long videos)
    "frames_count": ThresholdSpec(
        metric_name="frames_count",
        display_name="Mean Detections per Track",
        subtitle=None,
        low_upper=15.0,                            # SSv2 lower bound
        high_lower=35.0,                            # above most benchmarks
        label_low="Few detections",
        label_mid="Moderate detections",
        label_high="Many detections",
        desc_low="Short detection sequences per object — brief clips "
                 "or highly fragmented tracking.",
        desc_mid="Typical detection count for video benchmarks.",
        desc_high="Long detection sequences — extended footage, "
                  "long-lived objects, or high frame rate.",
        higher_is_better=None,
        range=(0, 100)
    ),
    # domain_concentration: top-1 domain % from YOLO semantic classification
    # benchmarks: Asm101=77%, HMDB=31%, Kin400=25%, UCF101=53%, SSv2=35%
    "domain_concentration": ThresholdSpec(
        metric_name="domain_concentration",
        display_name="Domain Concentration (top-1 %)",
        subtitle=None,
        low_upper=30.0,                            # Kinetics/HMDB zone — diverse
        high_lower=50.0,                            # UCF101/Assembly101 zone — concentrated
        label_low="Diverse domain mix",
        label_mid="Moderate concentration",
        label_high="Domain-concentrated",
        desc_low="Detected objects spread across many semantic domains. "
                 "In-the-wild style content variety.",
        desc_mid="Moderate domain focus with secondary domains present.",
        desc_high="Single domain dominates the dataset. "
                  "Typical for specialized datasets.",
        higher_is_better=None,
        range=(0.0, 100.0)
    ),
}
# fmt: on


def evaluate_metric(metric_name: str, value: float) -> ThresholdResult:
    """Evaluate a single metric against hardcoded thresholds."""
    spec = THRESHOLDS[metric_name]
    if value < spec.low_upper:
        level, label, desc = Level.LOW, spec.label_low, spec.desc_low
    elif value > spec.high_lower:
        level, label, desc = Level.HIGH, spec.label_high, spec.desc_high
    else:
        level, label, desc = Level.MEDIUM, spec.label_mid, spec.desc_mid

    return ThresholdResult(
        metric_name=metric_name,
        display_name=spec.display_name,
        value=value,
        level=level,
        label=label,
        description=desc,
        low_upper=spec.low_upper,
        high_lower=spec.high_lower,
    )


def evaluate_all(metrics: dict[str, float]) -> list[ThresholdResult]:
    results = []
    for name, value in metrics.items():
        if name in THRESHOLDS:
            results.append(evaluate_metric(name, value))
    return results


# ──────────────────────────────────────────────────────────────────────
# Per-group verdict for analysis tab (Scene / Video mode)
# ──────────────────────────────────────────────────────────────────────

# Names as returned by pipeline vs THRESHOLDS keys
METRIC_NAME_ALIASES: dict[str, str] = {
    "flow": "optical_flow",
    "smoothness": "motion_smoothness",
}


@dataclass
class GroupVerdict:
    """Verdict for one metric group to show in the analysis tab description card."""
    range_label: str          # "High" | "Medium" | "Low"
    title_line: str           # e.g. "High Quality Scene"
    description_lines: list[str]  # 2–3 lines from desc_low / desc_mid / desc_high
    warning_line: Optional[str] = None  # optional 4th line, e.g. one anomalous metric
    icon_level: str = "mid"   # "high" | "mid" | "low" for icon/color
    no_objects_detected: bool = False  # True only for objects group when all metrics are None
    empty: bool = False       # True when no metrics to interpret (show nothing or placeholder)


def _metrics_to_dict(metrics: Union[list, dict]) -> dict[str, float]:
    """Normalize metrics to dict[str, float]. Accept list of objects with .name/.value or dict."""
    out: dict[str, float] = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if v is not None and isinstance(v, (int, float)):
                out[k] = float(v)
        return out
    for m in metrics or []:
        if m is None:
            continue
        name = getattr(m, "name", None)
        value = getattr(m, "value", None)
        if name is not None and value is not None and isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def get_group_verdict(
    metrics: Union[list, dict],
    group_name: str,
    mode: str,
) -> GroupVerdict:
    """
    Build a short verdict for one metric group (quality, motion, objects, diversity)
    using THRESHOLDS (desc_low, desc_mid, desc_high). For use in analysis_tab description card.

    group_name: "quality" | "motion" | "objects" | "diversity"
    mode: "scene" | "video"
    """
    values = _metrics_to_dict(metrics)
    group_display = {
        "quality": "Quality",
        "motion": "Motion",
        "objects": "Objects",
        "diversity": "Diversity",
    }.get(group_name, group_name.capitalize())
    mode_display = "Scene" if mode == "scene" else "Video"

    # Objects: if all values are None, no objects detected
    objects_keys = {"duration_sec", "persistence", "norm_avg_velocity", "frames_count"}
    if group_name == "diversity":
        if not values:
            # Had any object metrics but all None?
            if isinstance(metrics, list):
                object_metrics = [m for m in metrics if m is not None and getattr(m, "name", None) in objects_keys]
                if object_metrics and all(getattr(m, "value", None) is None for m in object_metrics):
                    return GroupVerdict(
                        range_label="",
                        title_line="No objects detected",
                        description_lines=[
                            "Object detection was run but no objects were found in this content.",
                        ],
                        icon_level="mid",
                        no_objects_detected=True,
                    )
            elif isinstance(metrics, dict) and any(k in objects_keys for k in metrics):
                return GroupVerdict(
                    range_label="",
                    title_line="No objects detected",
                    description_lines=[
                        "Object detection was run but no objects were found in this content.",
                    ],
                    icon_level="mid",
                    no_objects_detected=True,
                )

    # Map pipeline names to THRESHOLDS keys
    normalized: dict[str, float] = {}
    for name, val in values.items():
        key = METRIC_NAME_ALIASES.get(name, name)
        if key in THRESHOLDS:
            normalized[key] = val

    if not normalized:
        return GroupVerdict(
            range_label="",
            title_line="",
            description_lines=[],
            empty=True,
        )

    # Evaluate each metric
    results: list[ThresholdResult] = []
    for name, value in normalized.items():
        try:
            results.append(evaluate_metric(name, value))
        except Exception:
            continue

    if not results:
        return GroupVerdict(range_label="", title_line="", description_lines=[], empty=True)

    # For motion: PSNR and SSIM mean "frame similarity" — high value = less motion. Invert their
    # level when aggregating so high PSNR/SSIM contribute to LOW motion, not HIGH.
    def _motion_level(r: ThresholdResult) -> Level:
        if group_name != "motion":
            return r.level
        if r.metric_name in ("psnr", "ssim"):
            if r.level == Level.HIGH:
                return Level.LOW
            if r.level == Level.LOW:
                return Level.HIGH
        if r.metric_name == "artifacts":
            if r.level == Level.HIGH:
                return Level.LOW
            if r.level == Level.LOW:
                return Level.HIGH
        return r.level

    low_c = sum(1 for r in results if _motion_level(r) == Level.LOW)
    mid_c = sum(1 for r in results if _motion_level(r) == Level.MEDIUM)
    high_c = sum(1 for r in results if _motion_level(r) == Level.HIGH)
    if high_c >= low_c and high_c >= mid_c:
        agg_level = Level.HIGH
        range_label = "High"
        icon_level = "high"
    elif low_c >= mid_c and low_c >= high_c:
        agg_level = Level.LOW
        range_label = "Low"
        icon_level = "low"
    else:
        agg_level = Level.MEDIUM
        range_label = "Medium"
        icon_level = "mid"

    # Descriptions: use up to 2–3 from metrics that match aggregate level (or first available)
    same_level = [r for r in results if r.level == agg_level]
    other_level = [r for r in results if r.level != agg_level]
    desc_lines: list[str] = []
    seen: set[str] = set()
    for r in (same_level or results)[:2]:
        if r.description and r.description.strip() not in seen:
            seen.add(r.description.strip())
            desc_lines.append(r.description.strip())
    if len(desc_lines) < 3 and other_level:
        for r in other_level[:1]:
            if r.description and r.description.strip() not in seen:
                desc_lines.append(r.description.strip())
                break

    # Warning: one metric strongly disagrees with aggregate
    warning_line: Optional[str] = None
    if other_level and agg_level != Level.MEDIUM:
        # Pick one outlier to mention
        outlier = other_level[0]
        spec = THRESHOLDS.get(outlier.metric_name)
        if spec:
            warning_line = f"Note: {outlier.display_name} is {outlier.label} (value: {outlier.value:.3f})."

    title_line = f"{range_label} {group_display} {mode_display}" if range_label else f"{group_display} {mode_display}"

    return GroupVerdict(
        range_label=range_label,
        title_line=title_line,
        description_lines=desc_lines[:3],
        warning_line=warning_line,
        icon_level=icon_level,
    )


def get_description(
        metrics: dict[str, float] | list[Metric],
        group_name: str,
        score: float,
        mode: str
) -> GroupVerdict | None:

    if isinstance(metrics, list):
        metrics = {metric.name: metric.value for metric in metrics}

    if len(metrics) == 0:
        return None

    def get_metric_description(name: str, value: float):
        spec = THRESHOLDS[name]
        if value < spec.low_upper:
            return spec.desc_low
        elif value < spec.high_lower:
            return spec.desc_mid
        else:
            return spec.desc_high

    def construct_title_line(rng: Literal["High", "Medium", "Low"]) -> str:
        return f"{rng} {group_name.title()} {mode.title()}"

    def range_to_icon_level(rng: Literal["High", "Medium", "Low"]) -> Literal["high", "mid", "low"]:
        if rng == "High":
            return "high"
        elif rng == "Low":
            return "low"
        else:
            return "mid"

    if score < 0.33:
        range_label = "Low"
    elif score < 0.66:
        range_label = "Medium"
    else:
        range_label = "High"

    warning = None
    description = []

    if group_name == "quality":
        general_metrics = ("entropy", "blur")
        for g_metric_name in general_metrics:
            g_value = metrics[g_metric_name]
            description.append(get_metric_description(g_metric_name, g_value))

    if group_name == "motion":
        general_metrics = ("optical_flow", "psnr")
        for g_metric_name in general_metrics:
            g_value = metrics[g_metric_name]
            description.append(get_metric_description(g_metric_name, g_value))

        background_dominance_v = metrics["background_dominance"]
        warning = get_metric_description("background_dominance", background_dominance_v)

    objects_keys = {"duration_sec", "persistence", "norm_avg_velocity", "frames_count"}
    if group_name == "diversity":
        general_metrics = ("duration_sec", "persistence", "clip_ics")
        object_values = [value for key, value in metrics.items() if key in objects_keys]
        if not object_values or all(value is None for value in object_values):
            warning = "No objects detected!"
        for g_metric_name in general_metrics:
            try:
                g_value = metrics[g_metric_name]
            except KeyError:
                continue
            try:
                description.append(get_metric_description(g_metric_name, g_value))
            except TypeError:
                warning = "No objects detected!"


    return GroupVerdict(
        range_label=range_label,
        title_line=construct_title_line(range_label),  # noqa
        description_lines=description,  # 2–3 lines from desc_low / desc_mid / desc_high
        warning_line=warning,  # optional 4th line, e.g. one anomalous metric
        icon_level=range_to_icon_level(range_label)  # noqa
    )


import numpy as np
import math


def get_gaussian_score(value, target, range_limit):
    """
    Рассчитывает оценку по Гауссу.
    range_limit: расстояние от target до границы, где score должен быть ~0.1
    """
    if value == target:
        return 1.0

    # Вычисляем сигму так, чтобы на границе оценка была 0.1
    # f(limit) = exp(-(limit^2)/(2*sigma^2)) = 0.1
    # Отсюда sigma = limit / sqrt(2 * ln(10))
    sigma = range_limit / math.sqrt(2 * math.log(10))

    return math.exp(-( (value - target)**2 ) / (2 * sigma**2))


def get_group_score(metrics: list[Metric] | dict[str, float]) -> float:
    scores = []

    if isinstance(metrics, list):
        metrics = {metric.name: metric.value for metric in metrics}

    for metric, value in metrics.items():
        if metric == "background_dominance":
            continue
        if value is None: continue

        spec = THRESHOLDS.get(metric)
        low, high = spec.range
        val = max(low, min(high, value))

        if spec.mode == "target":
            target = getattr(spec, 'target', (low + high) / 2)
            # Определяем расстояние до ближайшей границы как предел колокола
            dist_to_limit = min(target - low, high - target)
            score = get_gaussian_score(val, target, dist_to_limit)

        elif spec.mode == "inverse":
            score = 1.0 - (val - low) / (high - low)

        else:  # direct
            score = (val - low) / (high - low)

        scores.append(score)

    return np.mean(scores) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────
# Level 2: Composite conclusions
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CompositeConclusion:
    title: str
    category: str       # motion_profile, pretraining_suitability, warning, recommendation
    severity: str       # info, warning, critical
    description: str
    recommendation: str
    involved_metrics: list[str] = field(default_factory=list)


def _lev(metric_name: str, value: float) -> Level:
    spec = THRESHOLDS[metric_name]
    if value < spec.low_upper:
        return Level.LOW
    elif value > spec.high_lower:
        return Level.HIGH
    return Level.MEDIUM


def compute_composite_conclusions(metrics: dict[str, float]) -> list[CompositeConclusion]:
    conclusions: list[CompositeConclusion] = []

    def g(key):
        return metrics.get(key)

    def lev(key):
        v = g(key)
        return _lev(key, v) if v is not None and key in THRESHOLDS else None

    of_val    = g("opticflow_score")
    dino_tc   = g("temporary_consistency_dino_score")
    clip_tc   = g("temporary_consistency_clip_score")
    bg_dom    = g("background_dominance")
    musiq     = g("imaging_quality")
    entropy   = g("entropy")
    blur      = g("blur")
    smooth    = g("motion_smoothness")
    clip_ics  = g("clip_ics")
    dino_ics  = g("dino_ics")
    n_clust   = g("num_clusters")
    r_R       = g("r_R_score")
    noise     = g("noise_ratio")
    psnr_val  = g("psnr")
    ssim_val  = g("ssim")
    # tracking
    dur       = g("duration_sec")
    persist   = g("persistence")
    velocity  = g("norm_avg_velocity")
    displ     = g("norm_displacement")
    frames    = g("frames_count")
    dom_conc  = g("domain_concentration")

    # ── Pattern 1: Motion character ─────────────────────────────────
    if all(v is not None for v in [of_val, dino_tc, bg_dom]):
        of_l, tc_l, bg_l = lev("opticflow_score"), lev("temporary_consistency_dino_score"), lev("background_dominance")

        if of_l == Level.HIGH and tc_l in (Level.MEDIUM, Level.HIGH) and bg_l == Level.LOW:
            conclusions.append(CompositeConclusion(
                title="Structured object manipulation",
                category="motion_profile", severity="info",
                description=f"High optical flow ({of_val:.1f}) with preserved temporal consistency "
                            f"(DINO: {dino_tc:.3f}) and object-centric motion (BG dom: {bg_dom:.3f}). "
                            f"Active, structured object motion against a stable background.",
                recommendation="Models with local/object-centric attention are preferred over "
                               "global motion approaches (e.g., full-frame two-stream).",
                involved_metrics=["opticflow_score", "temporary_consistency_dino_score", "background_dominance"],
            ))
        elif of_l == Level.HIGH and tc_l == Level.LOW and bg_l == Level.HIGH:
            conclusions.append(CompositeConclusion(
                title="Chaotic camera-driven dynamics",
                category="motion_profile", severity="warning",
                description=f"High optical flow ({of_val:.1f}) with low temporal consistency "
                            f"(DINO: {dino_tc:.3f}) and camera-dominant motion ({bg_dom:.3f}). "
                            f"Erratic camera motion dominates the scene.",
                recommendation="Consider stabilization preprocessing or temporal augmentation. "
                               "Filtering clips with extreme camera motion may improve training.",
                involved_metrics=["opticflow_score", "temporary_consistency_dino_score", "background_dominance"],
            ))
        elif of_l == Level.HIGH and tc_l == Level.LOW and bg_l in (Level.LOW, Level.MEDIUM):
            conclusions.append(CompositeConclusion(
                title="High dynamics with consistency drop",
                category="motion_profile", severity="info",
                description=f"High optical flow ({of_val:.1f}) with below-benchmark temporal consistency "
                            f"(DINO: {dino_tc:.3f}). The consistency drop is likely a natural consequence "
                            f"of intensive motion rather than a data quality issue.",
                recommendation="Verify that clips don't contain undetected scene cuts. "
                               "If consistency drop is motion-driven, standard temporal models "
                               "should handle this — no special preprocessing needed.",
                involved_metrics=["opticflow_score", "temporary_consistency_dino_score"],
            ))
        elif of_l == Level.LOW and tc_l == Level.HIGH:
            conclusions.append(CompositeConclusion(
                title="Quasi-static scenes",
                category="motion_profile", severity="warning",
                description=f"Low optical flow ({of_val:.1f}) with high temporal consistency "
                            f"(DINO: {dino_tc:.3f}). Near-static footage with minimal meaningful motion.",
                recommendation="Temporal reasoning may not be learnable from this data. "
                               "Consider motion augmentation or supplementing with dynamic data.",
                involved_metrics=["opticflow_score", "temporary_consistency_dino_score"],
            ))
        elif of_l == Level.MEDIUM and tc_l == Level.MEDIUM:
            conclusions.append(CompositeConclusion(
                title="Balanced motion profile",
                category="motion_profile", severity="info",
                description=f"Moderate optical flow ({of_val:.1f}) with standard temporal consistency "
                            f"(DINO: {dino_tc:.3f}). Balanced dynamics comparable to Kinetics-400 / UCF101.",
                recommendation="Standard action recognition architectures should work well.",
                involved_metrics=["opticflow_score", "temporary_consistency_dino_score"],
            ))

    # ── Pattern 2: High dynamics explain low frame similarity ───────
    # Suppresses false alarms when PSNR/SSIM/smoothness/blur are low due to motion
    if of_val is not None and lev("opticflow_score") == Level.HIGH:
        low_similarity_metrics = []
        if psnr_val is not None and lev("psnr") == Level.LOW:
            low_similarity_metrics.append(f"PSNR ({psnr_val:.1f})")
        if ssim_val is not None and lev("ssim") == Level.LOW:
            low_similarity_metrics.append(f"SSIM ({ssim_val:.3f})")
        if smooth is not None and lev("motion_smoothness") == Level.LOW:
            low_similarity_metrics.append(f"smoothness ({smooth:.4f})")
        if blur is not None and lev("blur") == Level.LOW:
            low_similarity_metrics.append(f"high-freq detail ({blur:.1f})")

        if low_similarity_metrics:
            conclusions.append(CompositeConclusion(
                title="Low frame similarity explained by high dynamics",
                category="recommendation", severity="info",
                description=f"Below-benchmark {', '.join(low_similarity_metrics)} "
                            f"are a natural consequence of high optical flow ({of_val:.1f}). "
                            f"This is expected for highly dynamic content and does not "
                            f"indicate data quality problems.",
                recommendation="No corrective action needed — these metrics reflect "
                               "genuine motion intensity, not artifacts.",
                involved_metrics=["opticflow_score", "psnr", "ssim", "motion_smoothness", "blur"],
            ))

    # Low blur disambiguation: scene simplicity vs actual quality issue
    if blur is not None and of_val is not None:
        if lev("blur") == Level.LOW and lev("opticflow_score") in (Level.LOW, Level.MEDIUM):
            # Low high-freq detail without high dynamics — is it blur or simple scenes?
            # Two signals of "scene simplicity": low entropy OR decent MUSIQ
            is_simple_scene = False
            if entropy is not None and lev("entropy") == Level.LOW:
                is_simple_scene = True
            if musiq is not None and lev("imaging_quality") in (Level.MEDIUM, Level.HIGH):
                is_simple_scene = True

            if is_simple_scene:
                conclusions.append(CompositeConclusion(
                    title="Low detail explained by scene characteristics",
                    category="recommendation", severity="info",
                    description=f"Low Laplacian variance ({blur:.1f}) in a dataset with "
                                f"{'good perceptual quality (MUSIQ: ' + f'{musiq:.1f})' if musiq else ''}"
                                f"{' and ' if musiq and entropy else ''}"
                                f"{'moderate/low entropy (' + f'{entropy:.3f})' if entropy else ''}"
                                f" suggests visually simple scenes (uniform backgrounds, "
                                f"smooth surfaces) rather than actual blur.",
                    recommendation="This is a scene property, not a defect. No filtering needed, "
                                   "but feature learning may benefit from texture augmentation.",
                    involved_metrics=["blur", "imaging_quality", "entropy", "opticflow_score"],
                ))
            else:
                # Low blur + low MUSIQ + not high dynamics → genuine quality problem
                ent_str = f"{entropy:.3f}" if entropy is not None else "N/A"
                musiq_str = f"{musiq:.1f}" if musiq is not None else "N/A"
                conclusions.append(CompositeConclusion(
                    title="Low sharpness — possible quality issue",
                    category="warning", severity="warning",
                    description=f"Low Laplacian variance ({blur:.1f}) with low perceptual quality "
                                f"(MUSIQ: {musiq_str}) and without high dynamics "
                                f"(OF: {of_val:.1f}) suggests genuine quality issues: "
                                f"defocus, low resolution, or compression artifacts.",
                    recommendation="Consider quality-based filtering to remove the most "
                                   "blurry clips, or improve capture conditions.",
                    involved_metrics=["blur", "imaging_quality", "opticflow_score"],
                ))

    # ── Pattern 3: Pretraining suitability ──────────────────────────
    if all(v is not None for v in [musiq, clip_ics, n_clust]):
        q_l, d_l, c_l = lev("imaging_quality"), lev("clip_ics"), lev("num_clusters")

        if q_l in (Level.MEDIUM, Level.HIGH) and d_l in (Level.MEDIUM, Level.HIGH) and c_l in (Level.MEDIUM, Level.HIGH):
            conclusions.append(CompositeConclusion(
                title="Good pretraining source",
                category="pretraining_suitability", severity="info",
                description=f"Acceptable quality (MUSIQ: {musiq:.1f}), diversity "
                            f"(CLIP ICS: {clip_ics:.3f}), and taxonomy ({n_clust:.0f} clusters).",
                recommendation="Suitable as a pretraining source or to supplement "
                               "existing pretraining data with domain-specific content.",
                involved_metrics=["imaging_quality", "clip_ics", "num_clusters"],
            ))
        elif q_l in (Level.MEDIUM, Level.HIGH) and d_l == Level.LOW:
            conclusions.append(CompositeConclusion(
                title="Decent quality but low diversity",
                category="pretraining_suitability", severity="warning",
                description=f"Acceptable quality (MUSIQ: {musiq:.1f}) but limited diversity "
                            f"(CLIP ICS: {clip_ics:.3f}). Visually clean but repetitive.",
                recommendation="Supplement with diverse external data for pretraining. "
                               "Acceptable as a fine-tuning source if domain matches.",
                involved_metrics=["imaging_quality", "clip_ics"],
            ))
        elif q_l == Level.LOW and d_l in (Level.MEDIUM, Level.HIGH):
            conclusions.append(CompositeConclusion(
                title="Diverse but low quality",
                category="pretraining_suitability", severity="warning",
                description=f"Good diversity (CLIP ICS: {clip_ics:.3f}) but low quality "
                            f"(MUSIQ: {musiq:.1f}). Artifacts may contaminate representations.",
                recommendation="Apply quality-based filtering before training. "
                               "Consider using only for fine-tuning with a quality-pretrained backbone.",
                involved_metrics=["imaging_quality", "clip_ics"],
            ))
        elif q_l == Level.LOW and d_l == Level.LOW:
            conclusions.append(CompositeConclusion(
                title="Poor pretraining candidate",
                category="pretraining_suitability", severity="critical",
                description=f"Low quality (MUSIQ: {musiq:.1f}) and low diversity "
                            f"(CLIP ICS: {clip_ics:.3f}). Not suitable for pretraining.",
                recommendation="Major data pipeline revision needed: improve capture quality, "
                               "add diverse sources, or apply aggressive filtering.",
                involved_metrics=["imaging_quality", "clip_ics"],
            ))

    # ── Pattern 4: Warnings / anomalies ─────────────────────────────
    # High flow + low smoothness (only when flow is NOT high — otherwise Pattern 2 covers it)
    if all(v is not None for v in [of_val, smooth]):
        if lev("opticflow_score") == Level.MEDIUM and lev("motion_smoothness") == Level.LOW:
            conclusions.append(CompositeConclusion(
                title="Possible motion artifacts",
                category="warning", severity="warning",
                description=f"Moderate motion ({of_val:.1f}) with low smoothness ({smooth:.4f}) "
                            f"suggests unfiltered scene cuts or encoding artifacts.",
                recommendation="Run scene boundary detection to split clips at hard cuts. "
                               "Check for corrupted frames.",
                involved_metrics=["opticflow_score", "motion_smoothness"],
            ))

    # Low entropy + high MUSIQ
    if all(v is not None for v in [entropy, musiq]):
        if lev("entropy") == Level.LOW and lev("imaging_quality") == Level.HIGH:
            conclusions.append(CompositeConclusion(
                title="Visually clean but texturally flat",
                category="warning", severity="info",
                description=f"High quality (MUSIQ: {musiq:.1f}) but low entropy ({entropy:.3f}) "
                            f"— clean capture of visually uniform scenes.",
                recommendation="Feature learning may produce less discriminative representations. "
                               "Consider texture augmentation or mixing with richer data.",
                involved_metrics=["entropy", "imaging_quality"],
            ))

    # High noise + few clusters
    if all(v is not None for v in [noise, n_clust]):
        if lev("noise_ratio") == Level.HIGH and lev("num_clusters") == Level.LOW:
            conclusions.append(CompositeConclusion(
                title="Poorly structured content",
                category="warning", severity="critical",
                description=f"High noise ratio ({noise:.3f}) with few clusters ({n_clust:.0f}) "
                            f"— most clips don't form coherent categories.",
                recommendation="Review class definitions and annotation consistency. "
                               "Consider re-segmenting clips.",
                involved_metrics=["noise_ratio", "num_clusters"],
            ))

    # Semantic instability with visual stability
    if all(v is not None for v in [clip_tc, dino_tc]):
        if lev("temporary_consistency_clip_score") == Level.LOW \
                and lev("temporary_consistency_dino_score") in (Level.MEDIUM, Level.HIGH):
            conclusions.append(CompositeConclusion(
                title="Semantic instability with visual stability",
                category="warning", severity="warning",
                description=f"Low CLIP consistency ({clip_tc:.3f}) with preserved DINO consistency "
                            f"({dino_tc:.3f}) — frames look similar but semantic content shifts.",
                recommendation="Verify clip segmentation — clips may span multiple semantic events.",
                involved_metrics=["temporary_consistency_clip_score", "temporary_consistency_dino_score"],
            ))

    # Object-centric but static
    if all(v is not None for v in [bg_dom, of_val]):
        if lev("background_dominance") == Level.LOW and lev("opticflow_score") == Level.LOW:
            conclusions.append(CompositeConclusion(
                title="Object-centric but static",
                category="warning", severity="warning",
                description=f"Object-centric framing (BG dom: {bg_dom:.3f}) but minimal motion "
                            f"({of_val:.1f}). Objects present but barely moving.",
                recommendation="Temporal models may not learn useful dynamics. "
                               "Consider speed augmentation or spatial-only models.",
                involved_metrics=["background_dominance", "opticflow_score"],
            ))

    # ── Pattern 5: Cluster quality ──────────────────────────────────
    if all(v is not None for v in [r_R, n_clust, noise]):
        rR_l, noise_l, clust_l = lev("r_R_score"), lev("noise_ratio"), lev("num_clusters")

        if rR_l == Level.LOW and noise_l == Level.LOW and clust_l == Level.HIGH:
            conclusions.append(CompositeConclusion(
                title="Well-structured action taxonomy",
                category="recommendation", severity="info",
                description=f"Rich taxonomy ({n_clust:.0f} clusters), good separability "
                            f"(r/R: {r_R:.4f}), low noise ({noise:.3f}).",
                recommendation="Clean categorical structure. Classification-based approaches "
                               "should work effectively.",
                involved_metrics=["r_R_score", "num_clusters", "noise_ratio"],
            ))
        elif rR_l == Level.HIGH and clust_l in (Level.MEDIUM, Level.HIGH):
            conclusions.append(CompositeConclusion(
                title="Overlapping action categories",
                category="warning", severity="warning",
                description=f"Many clusters ({n_clust:.0f}) but poor separability "
                            f"(r/R: {r_R:.4f}). Actions overlap significantly.",
                recommendation="Consider merging similar classes, hierarchical classification, "
                               "or contrastive learning to improve separation.",
                involved_metrics=["r_R_score", "num_clusters"],
            ))

    # ── Pattern 6: Object tracking dynamics ────────────────────────

    # Tabletop / in-place manipulation (Assembly101-like)
    if all(v is not None for v in [velocity, displ]):
        if lev("norm_avg_velocity") == Level.LOW and lev("norm_displacement") == Level.LOW:
            persist_note = ""
            if persist is not None and lev("persistence") in (Level.MEDIUM, Level.HIGH):
                persist_note = f" with stable tracking (persistence: {persist:.2f})"
            conclusions.append(CompositeConclusion(
                title="In-place object manipulation",
                category="motion_profile", severity="info",
                description=f"Slow objects (velocity: {velocity:.4f}) with minimal displacement "
                            f"({displ:.3f}){persist_note}. "
                            f"Objects stay in place — "
                            f"tabletop or workstation-style footage.",
                recommendation="Spatial models may suffice. Fine-grained hand/object "
                               "interaction models are more suitable than global motion models.",
                involved_metrics=["norm_avg_velocity", "norm_displacement", "persistence"],
            ))

    # Tracking-challenging dynamics
    if all(v is not None for v in [velocity, persist]):
        if lev("norm_avg_velocity") == Level.HIGH and lev("persistence") == Level.LOW:
            conclusions.append(CompositeConclusion(
                title="Tracking-challenging dynamics",
                category="warning", severity="warning",
                description=f"Fast objects (velocity: {velocity:.4f}) with low tracking "
                            f"persistence ({persist:.2f}). Objects frequently lost by detector "
                            f"during rapid motion.",
                recommendation="Consider stronger data augmentation for motion blur, "
                               "or using a more robust tracker. ID switches may contaminate "
                               "temporal features.",
                involved_metrics=["norm_avg_velocity", "persistence"],
            ))

    # Cross-validation: YOLO velocity vs optical flow
    if all(v is not None for v in [velocity, of_val, bg_dom]):
        vel_l = lev("norm_avg_velocity")
        of_l = lev("opticflow_score")

        if vel_l == Level.LOW and of_l == Level.HIGH:
            conclusions.append(CompositeConclusion(
                title="Camera motion dominates over object motion",
                category="motion_profile", severity="info",
                description=f"High optical flow ({of_val:.1f}) but slow tracked objects "
                            f"(velocity: {velocity:.4f}). The dominant motion in the scene "
                            f"comes from camera movement, not object dynamics.",
                recommendation="Full-frame motion features (optical flow, ego-motion compensation) "
                               "will be more informative than object-centric tracking.",
                involved_metrics=["norm_avg_velocity", "opticflow_score", "background_dominance"],
            ))
        elif vel_l == Level.HIGH and of_l in (Level.MEDIUM, Level.HIGH):
            conclusions.append(CompositeConclusion(
                title="Object dynamics confirmed by tracking",
                category="recommendation", severity="info",
                description=f"Both optical flow ({of_val:.1f}) and object velocity "
                            f"({velocity:.4f}) indicate active dynamics. Object-level motion "
                            f"contributes significantly to the overall scene dynamics.",
                recommendation="Object-centric temporal models should capture these dynamics "
                               "effectively. Tracking-based features can complement flow.",
                involved_metrics=["norm_avg_velocity", "opticflow_score"],
            ))

    # ── Pattern 7: Domain specialization ────────────────────────────

    if dom_conc is not None and clip_ics is not None:
        dc_l = lev("domain_concentration")
        div_l = lev("clip_ics")

        if dc_l == Level.HIGH and div_l == Level.LOW:
            conclusions.append(CompositeConclusion(
                title="Narrow-domain specialized dataset",
                category="recommendation", severity="info",
                description=f"High domain concentration ({dom_conc:.0f}% top-1 domain) "
                            f"with low inter-clip diversity (CLIP ICS: {clip_ics:.3f}). "
                            f"Content is heavily specialized.",
                recommendation="Best used for fine-tuning, not pretraining. "
                               "Supplement with diverse data if pretraining is needed.",
                involved_metrics=["domain_concentration", "clip_ics"],
            ))
        elif dc_l == Level.LOW and div_l in (Level.MEDIUM, Level.HIGH):
            conclusions.append(CompositeConclusion(
                title="Multi-domain diverse content",
                category="recommendation", severity="info",
                description=f"Low domain concentration ({dom_conc:.0f}% top-1 domain) "
                            f"with good diversity (CLIP ICS: {clip_ics:.3f}). "
                            f"Objects span multiple semantic domains.",
                recommendation="Suitable for general-purpose pretraining. "
                               "Broad domain coverage supports transfer learning.",
                involved_metrics=["domain_concentration", "clip_ics"],
            ))

    return conclusions


# ──────────────────────────────────────────────────────────────────────
# Full report
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DatasetReport:
    dataset_name: str
    threshold_results: list[ThresholdResult]
    composite_conclusions: list[CompositeConclusion]

    def summary(self) -> str:
        lines = [f"=== Dataset Report: {self.dataset_name} ===\n"]

        lines.append("── Level 1: Per-Metric Assessment ──\n")
        for r in self.threshold_results:
            lines.append(
                f"  {r.display_name}: {r.value:.4f} → [{r.label}] "
                f"(thresholds: <{r.low_upper:.4f} | >{r.high_lower:.4f})"
            )
            lines.append(f"    {r.description}\n")

        lines.append("── Level 2: Composite Conclusions ──\n")
        if not self.composite_conclusions:
            lines.append("  No composite patterns detected.\n")
        for c in self.composite_conclusions:
            sev = {"info": "ℹ", "warning": "⚠", "critical": "‼"}[c.severity]
            lines.append(f"  {sev} {c.title} [{c.category}]")
            lines.append(f"    {c.description}")
            lines.append(f"    → {c.recommendation}\n")

        return "\n".join(lines)


def generate_report(dataset_name: str, metrics: dict[str, float]) -> DatasetReport:
    return DatasetReport(
        dataset_name=dataset_name,
        threshold_results=evaluate_all(metrics),
        composite_conclusions=compute_composite_conclusions(metrics),
    )
