from __future__ import annotations

import numpy as np


def build_common_roi(
    before_valid_mask: np.ndarray,
    after_valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Build the common valid region shared by the
    BEFORE and AFTER Sentinel-2 observations.
    """

    if before_valid_mask.ndim != 2:
        raise ValueError(
            "before_valid_mask must be a 2D array."
        )

    if after_valid_mask.ndim != 2:
        raise ValueError(
            "after_valid_mask must be a 2D array."
        )

    if (
        before_valid_mask.shape
        != after_valid_mask.shape
    ):
        raise ValueError(
            "BEFORE and AFTER valid masks "
            "must have identical shapes."
        )

    roi = (
        before_valid_mask.astype(bool)
        & after_valid_mask.astype(bool)
    )

    return roi


def calculate_roi_statistics(
    roi_mask: np.ndarray,
) -> dict:
    """
    Calculate basic statistics for the common ROI.
    """

    if roi_mask.ndim != 2:
        raise ValueError(
            "roi_mask must be a 2D array."
        )

    total_pixels = int(
        roi_mask.size
    )

    roi_pixels = int(
        roi_mask.sum()
    )

    coverage_percentage = (
        roi_pixels
        / total_pixels
        * 100.0
        if total_pixels > 0
        else 0.0
    )

    return {
        "total_pixels": total_pixels,
        "roi_pixels": roi_pixels,
        "coverage_percentage": coverage_percentage,
    }


def apply_roi(
    array: np.ndarray,
    roi_mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Apply an ROI mask to a 2D array.

    Pixels outside the ROI are replaced with fill_value.
    """

    if array.ndim != 2:
        raise ValueError(
            "array must be a 2D array."
        )

    if roi_mask.ndim != 2:
        raise ValueError(
            "roi_mask must be a 2D array."
        )

    if array.shape != roi_mask.shape:
        raise ValueError(
            "array and roi_mask must have "
            "identical shapes."
        )

    result = np.full(
        array.shape,
        fill_value,
        dtype=array.dtype,
    )

    result[roi_mask] = array[
        roi_mask
    ]

    return result


def calculate_roi_percentage(
    mask: np.ndarray,
    roi_mask: np.ndarray,
) -> float:
    """
    Calculate the percentage of ROI pixels represented
    by a boolean mask.
    """

    if mask.shape != roi_mask.shape:
        raise ValueError(
            "mask and roi_mask must have "
            "identical shapes."
        )

    roi_pixels = int(
        roi_mask.sum()
    )

    if roi_pixels == 0:
        return 0.0

    masked_pixels = int(
        (
            mask
            & roi_mask
        ).sum()
    )

    return (
        masked_pixels
        / roi_pixels
        * 100.0
    )