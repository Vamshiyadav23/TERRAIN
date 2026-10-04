from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping


SENTINEL_STAC_URL = "https://earth-search.aws.element84.com/v1"

# Sentinel-2 L2A assets exposed by Earth Search.
BANDS = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "nir": "B08",
}

SCL_ASSET = "scl"

DEFAULT_OUTPUT_SIZE = (256, 256)

OUTPUT_ROOT = Path("backend") / "satellite_data"


# Sentinel-2 Scene Classification Layer values.
#
# 0  = No data
# 1  = Saturated / defective
# 2  = Dark features / shadows
# 3  = Cloud shadow
# 4  = Vegetation
# 5  = Bare soil
# 6  = Water
# 7  = Unclassified
# 8  = Cloud medium probability
# 9  = Cloud high probability
# 10 = Thin cirrus
# 11 = Snow / ice
#
# For change detection we retain:
# 4, 5, 6, 7
#
# and reject:
# 0, 1, 2, 3, 8, 9, 10, 11
VALID_SCL_CLASSES = {4, 5, 6, 7}


def connect_to_stac() -> Client:
    """Connect to the Earth Search STAC API."""
    return Client.open(SENTINEL_STAC_URL)


def _radius_to_bbox(
    latitude: float,
    longitude: float,
    radius_m: float,
) -> list[float]:
    """
    Convert a center point and radius into an approximate WGS84 bbox.

    This is appropriate for the relatively small analysis areas
    supported by TERRAIN.
    """

    lat_delta = radius_m / 111_000

    longitude_scale = max(
        np.cos(np.radians(latitude)),
        0.01,
    )

    lon_delta = radius_m / (
        111_000 * longitude_scale
    )

    return [
        longitude - lon_delta,
        latitude - lat_delta,
        longitude + lon_delta,
        latitude + lat_delta,
    ]


def search_sentinel2(
    latitude: float,
    longitude: float,
    radius_m: float,
    date_range: str,
    max_cloud_cover: float = 20.0,
) -> list[Any]:
    """
    Search Sentinel-2 L2A scenes around a location.

    Results are ordered from lowest to highest scene-level
    cloud cover.
    """

    client = connect_to_stac()

    bbox = _radius_to_bbox(
        latitude,
        longitude,
        radius_m,
    )

    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={
            "eo:cloud_cover": {
                "lt": max_cloud_cover,
            }
        },
        max_items=50,
    )

    items = list(search.items())

    items.sort(
        key=lambda item: (
            item.properties.get(
                "eo:cloud_cover",
                100.0,
            )
            or 100.0
        )
    )

    return items


def select_best_scene(
    latitude: float,
    longitude: float,
    radius_m: float,
    date_range: str,
    max_cloud_cover: float = 20.0,
) -> Any:
    """
    Select the Sentinel-2 scene with the lowest scene-level
    cloud cover in the requested time range.
    """

    scenes = search_sentinel2(
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        date_range=date_range,
        max_cloud_cover=max_cloud_cover,
    )

    if not scenes:
        raise RuntimeError(
            "No Sentinel-2 scenes found for "
            f"date range: {date_range}"
        )

    return scenes[0]


def _get_band_asset(
    scene: Any,
    band_name: str,
) -> Any:
    """Return a Sentinel-2 asset from a STAC scene."""

    if band_name not in scene.assets:
        raise RuntimeError(
            f"Asset '{band_name}' is not available "
            f"in scene '{scene.id}'."
        )

    return scene.assets[band_name]


def _read_asset_window(
    asset_href: str,
    bbox: list[float],
    output_size: tuple[int, int] | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read a raster asset inside a WGS84 bbox.

    The raster is transformed from the geographic WGS84
    request into the native CRS of the Sentinel-2 asset.
    """

    with rasterio.open(asset_href) as src:

        wgs84_geometry = box(
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
        )

        raster_geometry = transform_geom(
            "EPSG:4326",
            src.crs,
            mapping(wgs84_geometry),
        )

        data, transform = mask(
            src,
            [raster_geometry],
            crop=True,
            nodata=0,
        )

        data = data[0]

        original_height, original_width = data.shape

        if output_size is not None:

            target_height, target_width = output_size

            from rasterio.io import MemoryFile

            profile = src.profile.copy()

            profile.update(
                height=original_height,
                width=original_width,
                transform=transform,
                count=1,
            )

            with MemoryFile() as memfile:

                with memfile.open(**profile) as temp:

                    temp.write(data, 1)

                    data = temp.read(
                        1,
                        out_shape=(
                            target_height,
                            target_width,
                        ),
                        resampling=resampling,
                    )

            scale_x = (
                original_width / target_width
            )
            scale_y = (
                original_height / target_height
            )

            transform = (
                transform
                * transform.scale(
                    scale_x,
                    scale_y,
                )
            )

        metadata = {
            "transform": transform,
            "crs": src.crs,
            "width": data.shape[1],
            "height": data.shape[0],
        }

    return data, metadata


def read_spectral_scene(
    scene: Any,
    bbox: list[float],
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """
    Read Sentinel-2 B02/B03/B04/B08 and SCL.

    Returns:

        spectral:
            Shape (4, H, W)

        scl:
            Shape (H, W)

        metadata:
            CRS and raster information

    Spectral band order:

        [B02, B03, B04, B08]
    """

    band_arrays = []

    reference_metadata = None

    # ---------------------------------------------------------
    # Read spectral bands
    # ---------------------------------------------------------

    for asset_name in BANDS:

        asset = _get_band_asset(
            scene,
            asset_name,
        )

        data, band_metadata = _read_asset_window(
            asset.href,
            bbox,
            output_size=output_size,
            resampling=Resampling.bilinear,
        )

        data = data.astype(
            np.float32
        )

        # Sentinel-2 L2A reflectance scaling.
        data /= 10_000.0

        data = np.clip(
            data,
            0.0,
            1.0,
        )

        band_arrays.append(data)

        if reference_metadata is None:
            reference_metadata = band_metadata

    spectral = np.stack(
        band_arrays,
        axis=0,
    )

    # ---------------------------------------------------------
    # Read SCL
    # ---------------------------------------------------------

    scl_asset = _get_band_asset(
        scene,
        SCL_ASSET,
    )

    scl, scl_metadata = _read_asset_window(
        scl_asset.href,
        bbox,
        output_size=output_size,
        resampling=Resampling.nearest,
    )

    scl = scl.astype(
        np.uint8
    )

    # ---------------------------------------------------------
    # Validate dimensions
    # ---------------------------------------------------------

    if spectral.shape[1:] != scl.shape:
        raise RuntimeError(
            "Spectral and SCL dimensions do not match: "
            f"{spectral.shape[1:]} vs {scl.shape}"
        )

    metadata = {
        "crs": str(reference_metadata["crs"]),
        "transform": list(
            reference_metadata["transform"]
        ),
        "width": spectral.shape[2],
        "height": spectral.shape[1],
    }

    return (
        spectral,
        scl,
        metadata,
    )


def create_valid_pixel_mask(
    scl: np.ndarray,
) -> np.ndarray:
    """
    Create a boolean mask of pixels considered valid
    for spectral change analysis.
    """

    valid_mask = np.isin(
        scl,
        list(VALID_SCL_CLASSES),
    )

    return valid_mask.astype(
        bool
    )


def apply_valid_mask(
    spectral: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Replace invalid pixels with NaN.

    NaN is used internally so invalid pixels cannot
    silently contribute to statistics.
    """

    if spectral.shape[1:] != valid_mask.shape:
        raise ValueError(
            "Spectral array and valid mask dimensions "
            "do not match."
        )

    masked = spectral.copy()

    masked[
        :,
        ~valid_mask,
    ] = np.nan

    return masked


def save_spectral_array(
    array: np.ndarray,
    output_path: Path,
) -> None:
    """Save a spectral array as NumPy."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        array.astype(
            np.float32
        ),
    )


def save_mask(
    mask_array: np.ndarray,
    output_path: Path,
) -> None:
    """Save a boolean valid-pixel mask."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        mask_array.astype(
            bool
        ),
    )


def scene_metadata(
    scene: Any,
) -> dict[str, Any]:
    """Extract useful metadata from a STAC scene."""

    return {
        "scene_id": scene.id,
        "date": (
            scene.datetime.isoformat()
            if scene.datetime
            else None
        ),
        "cloud_cover": scene.properties.get(
            "eo:cloud_cover"
        ),
        "collection": (
            scene.collection_id
            if hasattr(scene, "collection_id")
            else "sentinel-2-l2a"
        ),
    }

def acquire_scene(
    scene,
    latitude: float,
    longitude: float,
    radius_m: float,
    output_directory: Path,
    prefix: str,
    output_size: tuple[int, int] | None = None,
):

    bbox = _radius_to_bbox(
        latitude,
        longitude,
        radius_m,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    band_arrays = []
    spatial_metadata = None

    for asset_key in BANDS:

        asset = scene.assets.get(asset_key)

        if asset is None:
            raise RuntimeError(
                f"Missing Sentinel-2 asset: {asset_key}"
            )

        data, metadata = _read_asset_window(
            asset.href,
            bbox,
            output_size=output_size or DEFAULT_OUTPUT_SIZE,
            resampling=Resampling.bilinear,
        )

        band_arrays.append(
            data.astype(np.float32) / 10000.0
        )

        if spatial_metadata is None:
            spatial_metadata = metadata

    spectral = np.stack(
        band_arrays,
        axis=0,
    )

    spectral = np.clip(
        spectral,
        0.0,
        1.0,
    )

    # ---------------------------------------------------------
    # Read Scene Classification Layer
    # ---------------------------------------------------------

    scl_asset = scene.assets.get(SCL_ASSET)

    if scl_asset is None:
        raise RuntimeError(
            "Sentinel-2 SCL asset is missing."
        )

    scl, _ = _read_asset_window(
        scl_asset.href,
        bbox,
        output_size=output_size or DEFAULT_OUTPUT_SIZE,
        resampling=Resampling.nearest,
    )

    scl = scl.astype(np.uint8)

    # ---------------------------------------------------------
    # Valid-pixel mask
    # ---------------------------------------------------------

    valid_mask = create_valid_pixel_mask(scl)

    valid_percentage = (
        float(np.mean(valid_mask) * 100.0)
    )

    # ---------------------------------------------------------
    # Save arrays
    # ---------------------------------------------------------

    spectral_path = (
        output_directory /
        f"{prefix}_spectral.npy"
    )

    mask_path = (
        output_directory /
        f"{prefix}_valid_mask.npy"
    )

    scl_path = (
        output_directory /
        f"{prefix}_scl.npy"
    )

    save_spectral_array(
        spectral,
        spectral_path,
    )

    save_mask(
        valid_mask,
        mask_path,
    )

    np.save(
        scl_path,
        scl,
    )

    # ---------------------------------------------------------
    # Spatial metadata
    # ---------------------------------------------------------

    transform = spatial_metadata["transform"]
    crs = spatial_metadata["crs"]

    bounds = rasterio.transform.array_bounds(
        spatial_metadata["height"],
        spatial_metadata["width"],
        transform,
    )

    resolution = (
        abs(transform.a),
        abs(transform.e),
    )

    spatial_reference = {
        "crs": str(crs),
        "width": spatial_metadata["width"],
        "height": spatial_metadata["height"],
        "resolution_m": [
            float(resolution[0]),
            float(resolution[1]),
        ],
        "transform": [
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],
        "bounds": [
            float(bounds[0]),
            float(bounds[1]),
            float(bounds[2]),
            float(bounds[3]),
        ],
    }

    return {
        "scene_id": scene.id,
        "date": scene.datetime.isoformat(),
        "cloud_cover": scene.properties.get(
            "eo:cloud_cover"
        ),
        "collection": "sentinel-2-l2a",

        "latitude": latitude,
        "longitude": longitude,
        "radius_m": radius_m,

        "bbox": bbox,

        "bands": list(BANDS.values()),

        "shape": list(spectral.shape),

        "valid_pixel_percentage": valid_percentage,

        "crs": str(crs),

        "spatial_reference": spatial_reference,

        "output": {
            "spectral": str(spectral_path),
            "valid_mask": str(mask_path),
            "scl": str(scl_path),
        },
    }


def acquire_before_after(
    latitude: float,
    longitude: float,
    radius_m: float,
    before_date_range: str,
    after_date_range: str,
    max_cloud_cover: float = 20.0,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
) -> dict[str, Any]:
    """
    Acquire the best Sentinel-2 scene for both the BEFORE
    and AFTER periods.

    Returns metadata for both scenes.
    """

    before_scene = select_best_scene(
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        date_range=before_date_range,
        max_cloud_cover=max_cloud_cover,
    )

    after_scene = select_best_scene(
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        date_range=after_date_range,
        max_cloud_cover=max_cloud_cover,
    )

    before_directory = (
        OUTPUT_ROOT / "before"
    )

    after_directory = (
        OUTPUT_ROOT / "after"
    )

    before_metadata = acquire_scene(
        scene=before_scene,
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        output_directory=before_directory,
        prefix="before",
        output_size=output_size,
    )

    after_metadata = acquire_scene(
        scene=after_scene,
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        output_directory=after_directory,
        prefix="after",
        output_size=output_size,
    )

    result = {
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "radius_m": radius_m,
        },
        "before": before_metadata,
        "after": after_metadata,
    }

    metadata_path = (
        OUTPUT_ROOT / "metadata.json"
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    return result