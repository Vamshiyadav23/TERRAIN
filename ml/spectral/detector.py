from __future__ import annotations

import numpy as np


def robust_percentile_threshold(
    change_score: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = 95.0,
) -> float:
    

    values = change_score[valid_mask]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        raise ValueError(
            "No valid pixels available for threshold selection."
        )

    threshold = float(
        np.percentile(
            values,
            percentile,
        )
    )

    return threshold


def detect_change(
    change_score: np.ndarray,
    valid_mask: np.ndarray,
    percentile: float = 95.0,
    minimum_threshold: float = 0.0,
) -> dict:
    """
    Convert a continuous change-score map into a binary
    change mask using an adaptive percentile threshold.

    Parameters
    ----------
    change_score:
        2D array containing pixel-level change magnitude.

    valid_mask:
        2D boolean array indicating usable pixels.

    percentile:
        Percentile used to derive the adaptive threshold.

    minimum_threshold:
        Optional lower bound for the threshold.

    Returns
    -------
    dict containing:

        change_score
        threshold
        change_mask
        changed_pixels
        valid_pixels
        changed_percentage
    """

    if change_score.ndim != 2:
        raise ValueError(
            "change_score must be a 2D array."
        )

    if valid_mask.ndim != 2:
        raise ValueError(
            "valid_mask must be a 2D array."
        )

    if change_score.shape != valid_mask.shape:
        raise ValueError(
            "change_score and valid_mask "
            "must have the same shape."
        )

    threshold = robust_percentile_threshold(
        change_score,
        valid_mask,
        percentile,
    )

    threshold = max(
        threshold,
        minimum_threshold,
    )

    change_mask = (
        (change_score >= threshold)
        & valid_mask
    )

    changed_pixels = int(
        change_mask.sum()
    )

    valid_pixels = int(
        valid_mask.sum()
    )

    changed_percentage = (
        changed_pixels
        / valid_pixels
        * 100.0
        if valid_pixels > 0
        else 0.0
    )

    return {
        "change_score": change_score,
        "threshold": threshold,
        "change_mask": change_mask,
        "changed_pixels": changed_pixels,
        "valid_pixels": valid_pixels,
        "changed_percentage": changed_percentage,
    }


def normalize_change_score(
    change_score: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Normalize a change-score map to [0, 1] over valid pixels.

    This is useful for visualization and confidence scoring.
    """

    result = np.zeros_like(
        change_score,
        dtype=np.float32,
    )

    values = change_score[
        valid_mask
    ]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return result

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    if maximum <= minimum:
        result[valid_mask] = 0.0
        return result

    result[valid_mask] = (
        (
            change_score[valid_mask]
            - minimum
        )
        / (
            maximum
            - minimum
        )
    )

    result = np.clip(
        result,
        0.0,
        1.0,
    )

    return result