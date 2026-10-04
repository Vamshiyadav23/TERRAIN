from __future__ import annotations

from importlib import metadata
import json
import sys
from pathlib import Path

import numpy as np
from rasterio.transform import Affine

from ml.spectral.polygons import regions_to_feature_collection

# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Internal imports
# ============================================================

from ml.spectral.detector import (
    detect_change,
    normalize_change_score,
)

from ml.spectral.grouping import (
    connected_components,
    merge_nearby_regions,
    region_to_mask,
)

from ml.spectral.metrics import (
    calculate_region_metrics,
)

from ml.spectral.preprocessing import (
    calculate_ndvi,
    common_valid_mask,
    harmonize_after_to_before,
    ndvi_change,
    spectral_distance,
)

from ml.spectral.roi import (
    calculate_roi_statistics,
)


# ============================================================
# Paths
# ============================================================

SATELLITE_DATA_DIR = (
    PROJECT_ROOT
    / "backend"
    / "satellite_data"
)

BEFORE_DIR = (
    SATELLITE_DATA_DIR
    / "before"
)

AFTER_DIR = (
    SATELLITE_DATA_DIR
    / "after"
)

METADATA_PATH = (
    SATELLITE_DATA_DIR
    / "metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "spectral_zone_report.json"
)


# ============================================================
# Configuration
# ============================================================

BAND_NAMES = [
    "B02",
    "B03",
    "B04",
    "B08",
]

# IMPORTANT:
# Do not hard-code 10 m.
# The acquired 256x256 raster has its actual spatial
# resolution stored in metadata.json.
RESOLUTION_M = None

THRESHOLD_PERCENTILE = 95.0

MIN_REGION_PIXELS = 20

CONNECTIVITY = 8

MERGE_DISTANCE_PIXELS = 8.0


# ============================================================
# Utility functions
# ============================================================

def load_array(
    path: Path,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return np.load(path)


def find_existing_mask(
    prefix: str,
) -> Path:

    if prefix == "before":
        candidates = [
            BEFORE_DIR / "before_valid_mask.npy",
        ]

    elif prefix == "after":
        candidates = [
            AFTER_DIR / "after_valid_mask.npy",
        ]

    else:
        raise ValueError(
            f"Unknown scene prefix: {prefix}"
        )

    for candidate in candidates:

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find valid mask for '{prefix}'."
    )


def load_metadata() -> dict:

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def find_resolution(
    metadata: dict,
) -> float:

    """
    Recursively locate resolution_m in metadata.

    Expected value:
        resolution_m: [x_resolution, y_resolution]

    For the current acquired raster this should be
    approximately 3.984375 m/pixel.
    """

    found = []

    def search(
        value,
    ):

        if isinstance(value, dict):

            if "resolution_m" in value:
                found.append(
                    value["resolution_m"]
                )

            for child in value.values():
                search(child)

        elif isinstance(value, list):

            for child in value:
                search(child)

    search(metadata)

    if not found:
        raise ValueError(
            "Could not find 'resolution_m' in "
            "backend/satellite_data/metadata.json."
        )

    resolution = found[0]

    if isinstance(
        resolution,
        (list, tuple),
    ):

        if len(resolution) == 0:
            raise ValueError(
                "resolution_m is empty."
            )

        x_resolution = float(
            resolution[0]
        )

        if len(resolution) > 1:

            y_resolution = float(
                resolution[1]
            )

        else:

            y_resolution = x_resolution

    else:

        x_resolution = float(
            resolution
        )

        y_resolution = x_resolution

    if (
        x_resolution <= 0
        or y_resolution <= 0
    ):
        raise ValueError(
            "Invalid spatial resolution in metadata."
        )

    # Use the mean pixel dimension.
    return float(
        (x_resolution + y_resolution) / 2.0
    )


def percentile_scale(
    values: np.ndarray,
    mask: np.ndarray,
    percentile: float = 95.0,
) -> float:

    valid_values = values[
        mask
        & np.isfinite(values)
    ]

    if valid_values.size < 100:
        raise ValueError(
            "Insufficient valid values for "
            "robust scaling."
        )

    scale = float(
        np.percentile(
            valid_values,
            percentile,
        )
    )

    return max(
        scale,
        1e-8,
    )


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline() -> dict:

    print("=" * 70)

    print(
        "TERRAIN SENTINEL-2 "
        "SPECTRAL ZONE PIPELINE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    resolution_m = find_resolution(
        metadata
    )

    print()
    print(
        f"Spatial resolution: "
        f"{resolution_m:.6f} m/pixel"
    )

    # --------------------------------------------------------
    # Load spectral arrays
    # --------------------------------------------------------

    before_path = (
        BEFORE_DIR
        / "before_spectral.npy"
    )

    after_path = (
        AFTER_DIR
        / "after_spectral.npy"
    )

    before = load_array(
        before_path
    ).astype(
        np.float32
    )

    after = load_array(
        after_path
    ).astype(
        np.float32
    )

    print()
    print(
        f"BEFORE: {before_path}"
    )

    print(
        f"AFTER : {after_path}"
    )

    print(
        f"Input shape: {before.shape}"
    )

    # --------------------------------------------------------
    # Validate arrays
    # --------------------------------------------------------

    if before.shape != after.shape:

        raise ValueError(
            "BEFORE and AFTER spectral arrays "
            "have different shapes."
        )

    if before.ndim != 3:

        raise ValueError(
            "Expected spectral arrays with "
            "shape (bands, height, width)."
        )

    if before.shape[0] < 4:

        raise ValueError(
            "Expected B02, B03, B04 and B08."
        )

    # --------------------------------------------------------
    # Load validity masks
    # --------------------------------------------------------

    before_mask_path = (
        find_existing_mask(
            "before"
        )
    )

    after_mask_path = (
        find_existing_mask(
            "after"
        )
    )

    before_mask = load_array(
        before_mask_path
    ).astype(bool)

    after_mask = load_array(
        after_mask_path
    ).astype(bool)

    print()
    print(
        f"BEFORE mask: "
        f"{before_mask_path}"
    )

    print(
        f"AFTER mask : "
        f"{after_mask_path}"
    )

    # --------------------------------------------------------
    # Validate masks
    # --------------------------------------------------------

    if before_mask.shape != before.shape[1:]:

        raise ValueError(
            "BEFORE mask shape does not match "
            "BEFORE spectral image."
        )

    if after_mask.shape != after.shape[1:]:

        raise ValueError(
            "AFTER mask shape does not match "
            "AFTER spectral image."
        )

    # --------------------------------------------------------
    # Common ROI
    # --------------------------------------------------------

    roi_mask = common_valid_mask(
        before_mask,
        after_mask,
    )

    roi_stats = calculate_roi_statistics(
        roi_mask
    )

    print()
    print(
        f"ROI pixels: "
        f"{roi_stats['roi_pixels']}"
    )

    print(
        f"ROI coverage: "
        f"{roi_stats['coverage_percentage']:.2f}%"
    )

    # --------------------------------------------------------
    # Temporal harmonization
    # --------------------------------------------------------

    print()
    print(
        "Running temporal radiometric "
        "harmonization..."
    )

    harmonized_after, stable_mask = (
        harmonize_after_to_before(
            before,
            after,
            roi_mask,
        )
    )

    stable_pixels = int(
        np.count_nonzero(
            stable_mask
        )
    )

    stable_percentage = (
        stable_pixels
        / max(
            roi_stats["roi_pixels"],
            1,
        )
        * 100.0
    )

    print(
        f"Stable calibration pixels: "
        f"{stable_pixels}"
    )

    print(
        f"Stable calibration coverage: "
        f"{stable_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Physical NDVI
    # --------------------------------------------------------

    print()
    print(
        "Calculating physical NDVI..."
    )

    ndvi_before = calculate_ndvi(
        before
    )

    ndvi_after = calculate_ndvi(
        harmonized_after
    )

    ndvi_delta = ndvi_change(
        ndvi_before,
        ndvi_after,
        roi_mask,
    )

    # --------------------------------------------------------
    # Physical multispectral distance
    # --------------------------------------------------------

    print(
        "Calculating multispectral "
        "temporal distance..."
    )

    spectral_dist = spectral_distance(
        before,
        harmonized_after,
        roi_mask,
    )

    # --------------------------------------------------------
    # Robust scales for detection
    # --------------------------------------------------------

    absolute_ndvi_delta = np.abs(
        ndvi_delta
    )

    spectral_scale = percentile_scale(
        spectral_dist,
        roi_mask,
        percentile=95.0,
    )

    ndvi_scale = percentile_scale(
        absolute_ndvi_delta,
        roi_mask,
        percentile=95.0,
    )

    print()
    print(
        f"Spectral 95th percentile: "
        f"{spectral_scale:.6f}"
    )

    print(
        f"|NDVI change| 95th percentile: "
        f"{ndvi_scale:.6f}"
    )

    # --------------------------------------------------------
    # Detection components
    # --------------------------------------------------------

    spectral_component = (
        spectral_dist
        / spectral_scale
    )

    ndvi_component = (
        absolute_ndvi_delta
        / ndvi_scale
    )

    spectral_component = np.clip(
        spectral_component,
        0.0,
        1.0,
    )

    ndvi_component = np.clip(
        ndvi_component,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Combined change likelihood
    # --------------------------------------------------------

    change_score = (
        0.70
        * spectral_component
        +
        0.30
        * ndvi_component
    )

    change_score[
        ~roi_mask
    ] = 0.0

    # --------------------------------------------------------
    # Pixel-level detection
    # --------------------------------------------------------

    print()
    print(
        "Running adaptive change detection..."
    )

    detection = detect_change(
        change_score=change_score,
        valid_mask=roi_mask,
        percentile=THRESHOLD_PERCENTILE,
    )

    change_mask = detection[
        "change_mask"
    ]

    print(
        f"Threshold: "
        f"{detection['threshold']:.6f}"
    )

    print(
        f"Changed pixels: "
        f"{detection['changed_pixels']}"
    )

    print(
        f"Changed ROI area: "
        f"{detection['changed_percentage']:.2f}%"
    )

    # --------------------------------------------------------
    # Normalized score for zone evidence
    # --------------------------------------------------------

    normalized_change_score = (
        normalize_change_score(
            change_score,
            roi_mask,
        )
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    print()
    print(
        "Extracting connected regions..."
    )

    regions = connected_components(
        change_mask,
        min_pixels=MIN_REGION_PIXELS,
        connectivity=CONNECTIVITY,
    )

    raw_region_count = len(
        regions
    )

    print(
        f"Raw regions: "
        f"{raw_region_count}"
    )

    # --------------------------------------------------------
    # Spatial grouping
    # --------------------------------------------------------

    regions = merge_nearby_regions(
        regions,
        max_distance_pixels=(
            MERGE_DISTANCE_PIXELS
        ),
    )

    print(
        f"Grouped regions: "
        f"{len(regions)}"
    )

    # --------------------------------------------------------
    # Calculate zone metrics
    # --------------------------------------------------------

    zone_results = []

    for region in regions:

        region_mask = region_to_mask(
            region,
            change_mask.shape,
        )

        if region_mask.sum() == 0:
            continue

        metrics = calculate_region_metrics(
            region=region,
            change_score=normalized_change_score,
            ndvi_before=ndvi_before,
            ndvi_after=ndvi_after,
            spectral_distance=(
                spectral_dist
            ),
            region_mask=region_mask,
            resolution_m=resolution_m,
        )

        
        metrics["_pixels"] = region["pixels"]

        zone_results.append(
            metrics
        )

    # --------------------------------------------------------
    # Sort zones by physical area
    # --------------------------------------------------------

    zone_results.sort(
        key=lambda zone: zone["area_m2"],
        reverse=True,
    )

    # Re-number after sorting.

    for index, zone in enumerate(
        zone_results,
        start=1,
    ):

        zone["region_id"] = index

    # --------------------------------------------------------
    # Geographic zone polygons
    # --------------------------------------------------------

    spatial_reference = metadata["before"]["spatial_reference"]

    raster_transform = Affine(
        *spatial_reference["transform"]
    )

    source_crs = spatial_reference["crs"]

    polygon_regions = []

    for zone in zone_results:

        polygon_region = {
            "region_id": zone["region_id"],
            "pixels": zone["_pixels"],
            "pixel_count": zone["pixel_count"],
            "centroid_row": zone["centroid"]["row"],
            "centroid_col": zone["centroid"]["col"],
            "bbox": zone["bbox"],
        }

        polygon_regions.append(
            polygon_region
        )

    geojson = regions_to_feature_collection(
        regions=polygon_regions,
        raster_shape=before.shape[1:],
        transform=raster_transform,
        source_crs=source_crs,
    )

    print()
    print(
        f"GeoJSON zones: "
        f"{len(geojson['features'])}"
    )

    
    

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    for zone in zone_results:
            zone.pop("_pixels", None)

    report = {

        "pipeline": (
            "TERRAIN Sentinel-2 "
            "spectral zone pipeline"
        ),

        "version": "3.0",

        "methodology": {

            "change_detector": (
                "multispectral spectral-distance "
                "and NDVI evidence"
            ),

            "temporal_normalization": (
                "stable-pixel robust affine "
                "harmonization"
            ),

            "thresholding": (
                "adaptive percentile threshold"
            ),

            "region_extraction": (
                "connected components"
            ),

            "spatial_grouping": (
                "nearby-region grouping"
            ),

            "semantic_interpretation": (
                "not performed by this pipeline; "
                "reserved for Qwen semantic analysis"
            ),

        },

        "input": {

            "shape": list(
                before.shape
            ),

            "bands": BAND_NAMES,

            "resolution_m": (
                resolution_m
            ),

        },

        "roi": roi_stats,

        "harmonization": {

            "stable_pixels": (
                stable_pixels
            ),

            "stable_percentage": (
                stable_percentage
            ),

        },

        "detection": {

            "threshold_percentile": (
                THRESHOLD_PERCENTILE
            ),

            "threshold": (
                detection["threshold"]
            ),

            "changed_pixels": (
                detection["changed_pixels"]
            ),

            "changed_percentage": (
                detection[
                    "changed_percentage"
                ]
            ),

            "raw_regions": (
                raw_region_count
            ),

            "grouped_regions": (
                len(zone_results)
            ),

        },

        "zones": zone_results,

    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save geographic change zones
    # --------------------------------------------------------

    GEOJSON_PATH = (
        OUTPUT_DIR
        / "spectral_change_zones.geojson"
    )

    with open(
        GEOJSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            geojson,
            file,
            indent=2,
        )

    print(
        f"GeoJSON: {GEOJSON_PATH}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "SPECTRAL ZONE PIPELINE COMPLETE"
    )

    print("=" * 70)

    print(
        f"Final zones: "
        f"{len(zone_results)}"
    )

    for zone in zone_results:

        print()

        print(
            f"ZONE "
            f"{zone['region_id']}"
        )

        print(
            f"  Area: "
            f"{zone['area_m2']:.2f} m²"
        )

        print(
            f"  Change score: "
            f"{zone['change_evidence']['mean_change_score']:.3f}"
        )

        print(
            f"  NDVI change: "
            f"{zone['ndvi_evidence']['change']:.3f}"
        )

        print(
            f"  Spectral contrast: "
            f"{zone['change_evidence']['mean_spectral_distance']:.3f}"
        )

        print(
            f"  Confidence: "
            f"{zone['confidence']:.3f}"
        )

        print(
            f"  Severity: "
            f"{zone['severity']}"
        )

        print(
            f"  Direction: "
            f"{zone['direction']}"
        )

    print()

    print(
        f"Report: {REPORT_PATH}"
    )

    return report

    


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_pipeline()