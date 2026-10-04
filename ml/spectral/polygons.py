from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
from pyproj import Transformer


def region_to_polygon(
    region: dict[str, Any],
    raster_shape: tuple[int, int],
    transform: Affine,
    source_crs: str,
) -> dict[str, Any] | None:
    """
    Convert an exact-pixel spectral change region into a geographic polygon.

    The region must contain:
        region["pixels"] -> Nx2 array/list of [row, col]

    The polygon is generated directly from the detected pixels using
    rasterio.features.shapes().
    """

    pixels = region.get("pixels")

    if pixels is None or len(pixels) == 0:
        return None

    mask = np.zeros(raster_shape, dtype=np.uint8)

    pixels = np.asarray(pixels, dtype=np.int32)

    rows = pixels[:, 0]
    cols = pixels[:, 1]

    valid = (
        (rows >= 0)
        & (rows < raster_shape[0])
        & (cols >= 0)
        & (cols < raster_shape[1])
    )

    mask[rows[valid], cols[valid]] = 1

    if not np.any(mask):
        return None

    # Convert the exact raster pixels into polygon geometry.
    geometries = []

    for geom, value in shapes(
        mask,
        mask=mask.astype(bool),
        transform=transform,
    ):
        if value == 1:
            geometries.append(shape(geom))

    if not geometries:
        return None

    # A zone may contain multiple disconnected pieces after pixel grouping.
    # Combine them into one geometry.
    from shapely.ops import unary_union

    geometry = unary_union(geometries)

    if geometry.is_empty:
        return None

    # Fix minor topology problems.
    geometry = geometry.buffer(0)

    # Convert from Sentinel-2 projected CRS to WGS84.
    transformer = Transformer.from_crs(
        source_crs,
        "EPSG:4326",
        always_xy=True,
    )

    geometry_wgs84 = shapely_transform(
        transformer.transform,
        geometry,
    )

    return mapping(geometry_wgs84)


def region_to_geojson_feature(
    region: dict[str, Any],
    raster_shape: tuple[int, int],
    transform: Affine,
    source_crs: str,
) -> dict[str, Any] | None:
    """
    Convert one spectral change region into a GeoJSON Feature.
    """

    geometry = region_to_polygon(
        region=region,
        raster_shape=raster_shape,
        transform=transform,
        source_crs=source_crs,
    )

    if geometry is None:
        return None

    properties = {
        key: value
        for key, value in region.items()
        if key != "pixels"
    }

    # Convert NumPy scalar values into normal Python values
    # so the result can be serialized to JSON.
    for key, value in list(properties.items()):
        if isinstance(value, np.generic):
            properties[key] = value.item()

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": properties,
    }


def regions_to_feature_collection(
    regions: list[dict[str, Any]],
    raster_shape: tuple[int, int],
    transform: Affine,
    source_crs: str,
) -> dict[str, Any]:
    """
    Convert all detected regions into a GeoJSON FeatureCollection.
    """

    features = []

    for region in regions:
        feature = region_to_geojson_feature(
            region=region,
            raster_shape=raster_shape,
            transform=transform,
            source_crs=source_crs,
        )

        if feature is not None:
            features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }