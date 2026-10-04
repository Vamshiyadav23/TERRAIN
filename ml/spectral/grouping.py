from __future__ import annotations

import numpy as np
from scipy import ndimage


# ============================================================
# Connected components
# ============================================================

def connected_components(
    change_mask: np.ndarray,
    min_pixels: int = 20,
    connectivity: int = 8,
) -> list[dict]:
    """
    Extract exact connected change regions.

    Each returned region contains the exact pixel coordinates
    belonging to the component.

    Parameters
    ----------
    change_mask:
        Boolean 2D change mask.

    min_pixels:
        Minimum number of pixels required for a region.

    connectivity:
        4 or 8 pixel connectivity.
    """

    if change_mask.ndim != 2:
        raise ValueError(
            "change_mask must be a 2D array."
        )

    if min_pixels < 1:
        raise ValueError(
            "min_pixels must be >= 1."
        )

    if connectivity == 4:

        structure = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )

    elif connectivity == 8:

        structure = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

    else:

        raise ValueError(
            "connectivity must be either 4 or 8."
        )

    labeled, component_count = (
        ndimage.label(
            change_mask.astype(bool),
            structure=structure,
        )
    )

    regions = []

    for component_id in range(
        1,
        component_count + 1,
    ):

        rows, cols = np.where(
            labeled == component_id
        )

        pixel_count = len(rows)

        if pixel_count < min_pixels:
            continue

        min_row = int(
            rows.min()
        )

        max_row = int(
            rows.max()
        )

        min_col = int(
            cols.min()
        )

        max_col = int(
            cols.max()
        )

        centroid_row = float(
            rows.mean()
        )

        centroid_col = float(
            cols.mean()
        )

        # Exact coordinates are retained.
        pixels = np.column_stack(
            (
                rows,
                cols,
            )
        ).astype(
            np.int32
        )

        regions.append(
            {
                "region_id": len(regions) + 1,

                "pixel_count": int(
                    pixel_count
                ),

                "centroid_row": (
                    centroid_row
                ),

                "centroid_col": (
                    centroid_col
                ),

                "bbox": {
                    "min_row": min_row,
                    "min_col": min_col,
                    "max_row": max_row,
                    "max_col": max_col,
                },

                "pixels": pixels,
            }
        )

    return regions


# ============================================================
# Region mask
# ============================================================

def region_to_mask(
    region: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert a region's exact pixel coordinates into
    a boolean image mask.
    """

    mask = np.zeros(
        shape,
        dtype=bool,
    )

    pixels = region.get(
        "pixels"
    )

    if pixels is None:
        raise ValueError(
            "Region does not contain exact pixel coordinates."
        )

    if len(pixels) == 0:
        return mask

    rows = pixels[:, 0]
    cols = pixels[:, 1]

    mask[
        rows,
        cols,
    ] = True

    return mask


# ============================================================
# Nearby region grouping
# ============================================================

def merge_nearby_regions(
    regions: list[dict],
    max_distance_pixels: float = 8.0,
) -> list[dict]:
    """
    Merge spatially nearby regions.

    IMPORTANT:
    Exact pixel coordinates are preserved.

    Two regions are merged when their pixel sets contain
    pixels whose Euclidean distance is <= max_distance_pixels.

    The implementation uses bounding-box distance as a fast
    pre-filter and exact pixel distance for the final test.
    """

    if len(regions) <= 1:
        return regions

    if max_distance_pixels < 0:
        raise ValueError(
            "max_distance_pixels must be >= 0."
        )

    remaining = list(
        regions
    )

    merged_regions = []

    while remaining:

        current = remaining.pop(
            0
        )

        changed = True

        while changed:

            changed = False

            keep = []

            for candidate in remaining:

                if _regions_are_nearby(
                    current,
                    candidate,
                    max_distance_pixels,
                ):

                    current = _merge_two_regions(
                        current,
                        candidate,
                    )

                    changed = True

                else:

                    keep.append(
                        candidate
                    )

            remaining = keep

        merged_regions.append(
            current
        )

    # Re-number after merging.

    for index, region in enumerate(
        merged_regions,
        start=1,
    ):

        region["region_id"] = index

    return merged_regions


def _regions_are_nearby(
    region_a: dict,
    region_b: dict,
    max_distance_pixels: float,
) -> bool:
    """
    Determine whether two regions are spatially close.
    """

    bbox_a = region_a["bbox"]
    bbox_b = region_b["bbox"]

    bbox_distance = _bbox_distance(
        bbox_a,
        bbox_b,
    )

    if bbox_distance > max_distance_pixels:
        return False

    pixels_a = region_a["pixels"]
    pixels_b = region_b["pixels"]

    # Exact pixel distance.

    max_distance_squared = (
        max_distance_pixels
        * max_distance_pixels
    )

    # Compare the smaller region against the larger one
    # to reduce computation.

    if len(pixels_a) > len(pixels_b):
        pixels_a, pixels_b = (
            pixels_b,
            pixels_a,
        )

    for pixel in pixels_a:

        row = pixel[0]
        col = pixel[1]

        row_delta = (
            pixels_b[:, 0]
            - row
        )

        col_delta = (
            pixels_b[:, 1]
            - col
        )

        distance_squared = (
            row_delta * row_delta
            +
            col_delta * col_delta
        )

        if np.any(
            distance_squared
            <= max_distance_squared
        ):
            return True

    return False


def _bbox_distance(
    bbox_a: dict,
    bbox_b: dict,
) -> float:
    """
    Minimum Euclidean distance between two bounding boxes.
    """

    if (
        bbox_a["max_row"]
        < bbox_b["min_row"]
    ):

        row_gap = (
            bbox_b["min_row"]
            - bbox_a["max_row"]
        )

    elif (
        bbox_b["max_row"]
        < bbox_a["min_row"]
    ):

        row_gap = (
            bbox_a["min_row"]
            - bbox_b["max_row"]
        )

    else:

        row_gap = 0

    if (
        bbox_a["max_col"]
        < bbox_b["min_col"]
    ):

        col_gap = (
            bbox_b["min_col"]
            - bbox_a["max_col"]
        )

    elif (
        bbox_b["max_col"]
        < bbox_a["min_col"]
    ):

        col_gap = (
            bbox_a["min_col"]
            - bbox_b["max_col"]
        )

    else:

        col_gap = 0

    return float(
        np.sqrt(
            row_gap * row_gap
            +
            col_gap * col_gap
        )
    )


def _merge_two_regions(
    region_a: dict,
    region_b: dict,
) -> dict:
    """
    Merge two regions while preserving exact pixels.
    """

    pixels = np.vstack(
        (
            region_a["pixels"],
            region_b["pixels"],
        )
    )

    # Remove duplicates just in case.
    pixels = np.unique(
        pixels,
        axis=0,
    )

    rows = pixels[:, 0]
    cols = pixels[:, 1]

    return {
        "region_id": 0,

        "pixel_count": int(
            len(pixels)
        ),

        "centroid_row": float(
            rows.mean()
        ),

        "centroid_col": float(
            cols.mean()
        ),

        "bbox": {
            "min_row": int(
                rows.min()
            ),
            "min_col": int(
                cols.min()
            ),
            "max_row": int(
                rows.max()
            ),
            "max_col": int(
                cols.max()
            ),
        },

        "pixels": pixels.astype(
            np.int32
        ),
    }