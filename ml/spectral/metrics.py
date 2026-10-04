from __future__ import annotations

import numpy as np


# ============================================================
# Evidence helpers
# ============================================================

def calculate_confidence(
    mean_change_score: float,
    mean_spectral_distance: float,
    mean_absolute_ndvi_change: float,
) -> float:
    """
    Calculate an evidence-confidence score.

    IMPORTANT:
    This is NOT a trained-model probability and should
    not be interpreted as statistical accuracy.

    It summarizes agreement between:
        - normalized change evidence
        - physical spectral distance
        - absolute NDVI change
    """

    # Detection evidence is already normalized to [0, 1].
    detection_evidence = float(
        np.clip(
            mean_change_score,
            0.0,
            1.0,
        )
    )

    # Convert physical spectral distance into a bounded
    # evidence scale. Around 0.30 represents strong
    # multispectral temporal separation for this pipeline.
    spectral_evidence = float(
        np.clip(
            mean_spectral_distance / 0.30,
            0.0,
            1.0,
        )
    )

    # NDVI change around 0.30 represents strong vegetation
    # change.
    ndvi_evidence = float(
        np.clip(
            mean_absolute_ndvi_change / 0.30,
            0.0,
            1.0,
        )
    )

    confidence = (
        0.40 * detection_evidence
        + 0.35 * spectral_evidence
        + 0.25 * ndvi_evidence
    )

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )


def calculate_severity(
    mean_change_score: float,
    spectral_distance: float,
    ndvi_change: float,
) -> str:
    """
    Classify physical change severity.

    Severity is based on physical evidence rather than
    simply on the fact that a region crossed the detection
    threshold.

    This is a rule-based interpretation layer, not a
    learned severity model.
    """

    absolute_ndvi = abs(
        float(ndvi_change)
    )

    spectral_distance = float(
        spectral_distance
    )

    change_score = float(
        mean_change_score
    )

    # --------------------------------------------------------
    # Critical
    # --------------------------------------------------------
    #
    # Strong multispectral change combined with strong
    # vegetation evidence, OR extremely strong spectral
    # separation.
    #
    if (
        (
            spectral_distance >= 0.30
            and absolute_ndvi >= 0.25
        )
        or spectral_distance >= 0.45
        or (
            change_score >= 0.92
            and absolute_ndvi >= 0.35
        )
    ):
        return "critical"

    # --------------------------------------------------------
    # High
    # --------------------------------------------------------

    if (
        (
            spectral_distance >= 0.20
            and absolute_ndvi >= 0.15
        )
        or spectral_distance >= 0.30
        or (
            change_score >= 0.85
            and absolute_ndvi >= 0.20
        )
    ):
        return "high"

    # --------------------------------------------------------
    # Medium
    # --------------------------------------------------------

    if (
        (
            spectral_distance >= 0.12
            and absolute_ndvi >= 0.08
        )
        or spectral_distance >= 0.18
        or (
            change_score >= 0.80
            and absolute_ndvi >= 0.10
        )
    ):
        return "medium"

    # --------------------------------------------------------
    # Low
    # --------------------------------------------------------

    return "low"


def classify_change_direction(
    ndvi_change: float,
    spectral_distance: float,
) -> str:
    """
    Classify the dominant direction of spectral change.
    """

    ndvi_change = float(
        ndvi_change
    )

    spectral_distance = float(
        spectral_distance
    )

    if ndvi_change <= -0.15:
        return "vegetation_loss"

    if ndvi_change >= 0.15:
        return "vegetation_gain"

    if spectral_distance >= 0.20:
        return "spectral_change"

    return "mixed_or_uncertain"


# ============================================================
# Region metrics
# ============================================================

def calculate_region_metrics(
    region: dict,
    change_score: np.ndarray,
    ndvi_before: np.ndarray,
    ndvi_after: np.ndarray,
    spectral_distance: np.ndarray,
    region_mask: np.ndarray,
    resolution_m: float,
) -> dict:
    """
    Calculate quantitative metrics for a detected change zone.

    Parameters
    ----------
    region:
        Connected-component region metadata.

    change_score:
        Normalized detection score in [0, 1].

    ndvi_before:
        Physical NDVI before the temporal change.

    ndvi_after:
        Physical NDVI after the temporal change.

    spectral_distance:
        Physical multispectral Euclidean distance.

    region_mask:
        Boolean mask containing the exact changed pixels
        represented by this region.

    resolution_m:
        Actual raster pixel resolution in meters.
    """

    if resolution_m <= 0:
        raise ValueError(
            "resolution_m must be positive."
        )

    if region_mask.shape != ndvi_before.shape:
        raise ValueError(
            "region_mask and NDVI arrays must have "
            "identical spatial dimensions."
        )

    pixel_count = int(
        np.count_nonzero(
            region_mask
        )
    )

    if pixel_count == 0:
        raise ValueError(
            "Region contains no pixels."
        )

    # --------------------------------------------------------
    # Extract region evidence
    # --------------------------------------------------------

    region_change = change_score[
        region_mask
    ]

    region_ndvi_before = ndvi_before[
        region_mask
    ]

    region_ndvi_after = ndvi_after[
        region_mask
    ]

    region_spectral_distance = (
        spectral_distance[
            region_mask
        ]
    )

    # Keep only finite values.

    finite_change = region_change[
        np.isfinite(region_change)
    ]

    finite_ndvi_before = (
        region_ndvi_before[
            np.isfinite(
                region_ndvi_before
            )
        ]
    )

    finite_ndvi_after = (
        region_ndvi_after[
            np.isfinite(
                region_ndvi_after
            )
        ]
    )

    finite_spectral = (
        region_spectral_distance[
            np.isfinite(
                region_spectral_distance
            )
        ]
    )

    if (
        finite_change.size == 0
        or finite_ndvi_before.size == 0
        or finite_ndvi_after.size == 0
        or finite_spectral.size == 0
    ):
        raise ValueError(
            "Region contains insufficient finite evidence."
        )

    # --------------------------------------------------------
    # Physical metrics
    # --------------------------------------------------------

    mean_change_score = float(
        np.mean(
            finite_change
        )
    )

    mean_spectral_distance = float(
        np.mean(
            finite_spectral
        )
    )

    median_spectral_distance = float(
        np.median(
            finite_spectral
        )
    )

    mean_ndvi_before = float(
        np.mean(
            finite_ndvi_before
        )
    )

    mean_ndvi_after = float(
        np.mean(
            finite_ndvi_after
        )
    )

    mean_ndvi_change = (
        mean_ndvi_after
        - mean_ndvi_before
    )

    mean_absolute_ndvi_change = float(
        np.mean(
            np.abs(
                region_ndvi_after[
                    np.isfinite(
                        region_ndvi_after
                    )
                ]
                -
                region_ndvi_before[
                    np.isfinite(
                        region_ndvi_before
                    )
                ]
            )
        )
    )

    # --------------------------------------------------------
    # Physical area
    # --------------------------------------------------------

    pixel_area_m2 = (
        resolution_m
        * resolution_m
    )

    area_m2 = (
        pixel_count
        * pixel_area_m2
    )

    area_hectares = (
        area_m2
        / 10000.0
    )

    # --------------------------------------------------------
    # Evidence confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        mean_change_score=(
            mean_change_score
        ),
        mean_spectral_distance=(
            mean_spectral_distance
        ),
        mean_absolute_ndvi_change=(
            mean_absolute_ndvi_change
        ),
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity = calculate_severity(
        mean_change_score=(
            mean_change_score
        ),
        spectral_distance=(
            mean_spectral_distance
        ),
        ndvi_change=(
            mean_ndvi_change
        ),
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = classify_change_direction(
        ndvi_change=(
            mean_ndvi_change
        ),
        spectral_distance=(
            mean_spectral_distance
        ),
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "region_id": int(
            region["region_id"]
        ),

        "pixel_count": pixel_count,

        "area_m2": float(
            area_m2
        ),

        "area_hectares": float(
            area_hectares
        ),

        "centroid": {

            "row": float(
                region["centroid_row"]
            ),

            "col": float(
                region["centroid_col"]
            ),

        },

        "bbox": region[
            "bbox"
        ],

        "change_evidence": {

            "mean_change_score": (
                mean_change_score
            ),

            # IMPORTANT:
            # This is now the physical multispectral
            # distance, NOT a normalized value.

            "mean_spectral_distance": (
                mean_spectral_distance
            ),

            "median_spectral_distance": (
                median_spectral_distance
            ),

        },

        "ndvi_evidence": {

            "before": (
                mean_ndvi_before
            ),

            "after": (
                mean_ndvi_after
            ),

            "change": (
                float(
                    mean_ndvi_change
                )
            ),

            "absolute_change": (
                mean_absolute_ndvi_change
            ),

        },

        "confidence": (
            confidence
        ),

        "severity": (
            severity
        ),

        "direction": (
            direction
        ),

    }