from __future__ import annotations

import numpy as np


def common_valid_mask(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
) -> np.ndarray:
    """
    Pixels valid in both before and after scenes.
    """

    if before_mask.shape != after_mask.shape:
        raise ValueError(
            "Before and after masks must have the same shape."
        )

    return (
        before_mask.astype(bool)
        & after_mask.astype(bool)
    )


def robust_band_normalization(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robustly normalize each spectral band using common valid pixels.

    The transformation is applied independently per band.
    Percentiles reduce the influence of extreme pixels.
    """

    if before.shape != after.shape:
        raise ValueError(
            "Before and after arrays must have identical shapes."
        )

    if before.ndim != 3:
        raise ValueError(
            "Spectral arrays must have shape (bands, height, width)."
        )

    if valid_mask.shape != before.shape[1:]:
        raise ValueError(
            "Valid mask shape must match image spatial dimensions."
        )

    normalized_before = before.astype(
        np.float32,
        copy=True,
    )

    normalized_after = after.astype(
        np.float32,
        copy=True,
    )

    for band in range(before.shape[0]):

        b_values = before[band][valid_mask]
        a_values = after[band][valid_mask]

        b_values = b_values[
            np.isfinite(b_values)
        ]

        a_values = a_values[
            np.isfinite(a_values)
        ]

        if len(b_values) < 100 or len(a_values) < 100:
            raise ValueError(
                f"Insufficient valid pixels for band {band}."
            )

        b_low, b_high = np.percentile(
            b_values,
            [
                lower_percentile,
                upper_percentile,
            ],
        )

        a_low, a_high = np.percentile(
            a_values,
            [
                lower_percentile,
                upper_percentile,
            ],
        )

        if (
            b_high <= b_low
            or a_high <= a_low
        ):
            raise ValueError(
                f"Invalid percentile range for band {band}."
            )

        # Map each image into a comparable robust range.
        normalized_before[band] = (
            normalized_before[band] - b_low
        ) / (b_high - b_low)

        normalized_after[band] = (
            normalized_after[band] - a_low
        ) / (a_high - a_low)

    normalized_before = np.clip(
        normalized_before,
        0.0,
        1.0,
    )

    normalized_after = np.clip(
        normalized_after,
        0.0,
        1.0,
    )

    return (
        normalized_before,
        normalized_after,
    )


def calculate_ndvi(
    spectral: np.ndarray,
) -> np.ndarray:
    """
    Calculate NDVI from Sentinel-2 B04 (red)
    and B08 (NIR).

    Expected band order:
        0 = B02
        1 = B03
        2 = B04
        3 = B08
    """

    if spectral.shape[0] < 4:
        raise ValueError(
            "Expected at least four spectral bands."
        )

    red = spectral[2]
    nir = spectral[3]

    denominator = nir + red

    ndvi = np.divide(
        nir - red,
        denominator,
        out=np.zeros_like(nir, dtype=np.float32),
        where=np.abs(denominator) > 1e-8,
    )

    return ndvi.astype(np.float32)


def spectral_distance(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Calculate Euclidean spectral distance across
    B02, B03, B04 and B08.
    """

    difference = after - before

    distance = np.sqrt(
        np.sum(
            np.square(difference),
            axis=0,
        )
    )

    distance = distance.astype(
        np.float32
    )

    distance[~valid_mask] = np.nan

    return distance


def ndvi_change(
    before_ndvi: np.ndarray,
    after_ndvi: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Calculate temporal NDVI change.
    """

    if before_ndvi.shape != after_ndvi.shape:
        raise ValueError(
            "NDVI arrays must have identical shapes."
        )

    change = (
        after_ndvi - before_ndvi
    ).astype(np.float32)

    change[~valid_mask] = np.nan

    return change

def harmonize_after_to_before(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: np.ndarray,
    stable_percentile: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robustly harmonize AFTER reflectance to the BEFORE scene.

    The initial spectral difference is used to identify pixels
    that are relatively stable between the two dates.

    For each spectral band, a robust affine transformation is
    estimated:

        before ~= a * after + b

    The transformation is then applied to the AFTER image.

    Returns:
        harmonized_after
        stable_mask
    """

    if before.shape != after.shape:
        raise ValueError(
            "Before and after arrays must have identical shapes."
        )

    if before.ndim != 3:
        raise ValueError(
            "Spectral arrays must have shape (bands, height, width)."
        )

    if valid_mask.shape != before.shape[1:]:
        raise ValueError(
            "Valid mask shape must match image spatial dimensions."
        )

    before_f = before.astype(np.float32, copy=False)
    after_f = after.astype(np.float32, copy=False)

    # Initial spectral distance in physical reflectance space.
    initial_difference = after_f - before_f

    initial_distance = np.sqrt(
        np.sum(
            np.square(initial_difference),
            axis=0,
        )
    )

    initial_distance[
        ~valid_mask
    ] = np.nan

    finite_distance = initial_distance[
        np.isfinite(initial_distance)
    ]

    if finite_distance.size < 100:
        raise ValueError(
            "Insufficient valid pixels for temporal harmonization."
        )

    # Use relatively stable pixels for calibration.
    stable_threshold = np.percentile(
        finite_distance,
        stable_percentile,
    )

    stable_mask = (
        valid_mask
        & np.isfinite(initial_distance)
        & (initial_distance <= stable_threshold)
    )

    if np.count_nonzero(stable_mask) < 100:
        raise ValueError(
            "Insufficient stable pixels for temporal harmonization."
        )

    harmonized_after = after_f.copy()

    for band in range(before.shape[0]):

        x = after_f[band][stable_mask]
        y = before_f[band][stable_mask]

        finite = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        x = x[finite]
        y = y[finite]

        if len(x) < 100:
            raise ValueError(
                f"Insufficient stable pixels for band {band}."
            )

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x_centered = x - x_mean
        y_centered = y - y_mean

        denominator = np.sum(
            x_centered ** 2
        )

        if denominator <= 1e-12:
            slope = 1.0
        else:
            slope = (
                np.sum(
                    x_centered * y_centered
                )
                / denominator
            )

        intercept = (
            y_mean
            - slope * x_mean
        )

        # Prevent pathological calibration.
        slope = float(
            np.clip(
                slope,
                0.5,
                2.0,
            )
        )

        intercept = float(
            np.clip(
                intercept,
                -0.25,
                0.25,
            )
        )

        harmonized_after[band] = (
            slope * after_f[band]
            + intercept
        )

    harmonized_after = np.clip(
        harmonized_after,
        0.0,
        1.0,
    )

    return (
        harmonized_after.astype(np.float32),
        stable_mask,
    )