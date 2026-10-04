import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SATELLITE_DATA_DIR = (
    PROJECT_ROOT / "backend" / "satellite_data"
)

METADATA_PATH = SATELLITE_DATA_DIR / "metadata.json"

GEOJSON_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "spectral_change_zones.geojson"
)


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_zones():
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_zone(region_id: int):
    geojson = load_zones()

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})

        if int(properties.get("region_id", -1)) == int(
            region_id
        ):
            return feature

    return None


def load_spectral_data():
    before_path = (
        SATELLITE_DATA_DIR
        / "before"
        / "before_spectral.npy"
    )

    after_path = (
        SATELLITE_DATA_DIR
        / "after"
        / "after_spectral.npy"
    )

    before_mask_path = (
        SATELLITE_DATA_DIR
        / "before"
        / "before_valid_mask.npy"
    )

    after_mask_path = (
        SATELLITE_DATA_DIR
        / "after"
        / "after_valid_mask.npy"
    )

    before = np.load(before_path)
    after = np.load(after_path)

    before_mask = np.load(before_mask_path)
    after_mask = np.load(after_mask_path)

    return (
        before,
        after,
        before_mask,
        after_mask,
    )


def stretch_band(
    band,
    valid_mask,
    low=2,
    high=98,
):
    values = band[valid_mask]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return np.zeros_like(
            band,
            dtype=np.uint8,
        )

    low_value, high_value = np.percentile(
        values,
        [low, high],
    )

    if high_value <= low_value:
        high_value = low_value + 1e-6

    normalized = (
        band - low_value
    ) / (
        high_value - low_value
    )

    normalized = np.clip(
        normalized,
        0,
        1,
    )

    return (
        normalized * 255
    ).astype(np.uint8)


def make_rgb(
    spectral,
    valid_mask,
    common_percentiles=None,
):
    """
    Spectral order:
        0 = B02 Blue
        1 = B03 Green
        2 = B04 Red
        3 = B08 NIR

    RGB output uses:
        R = B04
        G = B03
        B = B02
    """

    if common_percentiles is None:
        common_percentiles = []

        for band_index in [2, 1, 0]:
            values = spectral[
                band_index
            ][valid_mask]

            values = values[
                np.isfinite(values)
            ]

            if values.size == 0:
                common_percentiles.append(
                    (0.0, 1.0)
                )
            else:
                common_percentiles.append(
                    tuple(
                        np.percentile(
                            values,
                            [2, 98],
                        )
                    )
                )

    channels = []

    for channel_index, band_index in enumerate(
        [2, 1, 0]
    ):
        band = spectral[band_index]

        low_value, high_value = (
            common_percentiles[channel_index]
        )

        if high_value <= low_value:
            high_value = low_value + 1e-6

        normalized = (
            band - low_value
        ) / (
            high_value - low_value
        )

        normalized = np.clip(
            normalized,
            0,
            1,
        )

        channels.append(
            (
                normalized * 255
            ).astype(np.uint8)
        )

    rgb = np.stack(
        channels,
        axis=-1,
    )

    rgb[~valid_mask] = 0

    return rgb


def encode_png(array):
    image = Image.fromarray(
        array,
        mode="RGB",
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return buffer.getvalue()


def encode_mask(mask):
    image = Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L",
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True,
    )

    return buffer.getvalue()


def crop_with_padding(
    image,
    min_row,
    min_col,
    max_row,
    max_col,
    padding=15,
):
    height, width = image.shape[:2]

    min_row = max(
        0,
        min_row - padding,
    )

    min_col = max(
        0,
        min_col - padding,
    )

    max_row = min(
        height - 1,
        max_row + padding,
    )

    max_col = min(
        width - 1,
        max_col + padding,
    )

    return (
        image[
            min_row : max_row + 1,
            min_col : max_col + 1,
        ],
        (
            min_row,
            min_col,
            max_row,
            max_col,
        ),
    )


def build_zone_mask(
    feature,
    shape,
    transform,
    source_crs,
):
    geometry = feature.get("geometry")

    if not geometry:
        return np.zeros(
            shape,
            dtype=bool,
        )

    from shapely.geometry import shape as shapely_shape
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer

    # GeoJSON polygons are stored in EPSG:4326.
    geojson_crs = "EPSG:4326"

    # Sentinel-2 raster uses its native projected CRS.
    transformer = Transformer.from_crs(
        geojson_crs,
        source_crs,
        always_xy=True,
    )

    shapely_geometry = shapely_shape(
        geometry
    )

    projected_geometry = shapely_transform(
        transformer.transform,
        shapely_geometry,
    )

    mask = rasterize(
        [(projected_geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    return mask.astype(bool)

def create_zone_mask_overlay(
    before_rgb,
    zone_mask,
):
    """
    Creates a visual evidence image.

    This is a visualization of the detected
    zone only. It is not a probability map.
    """

    output = before_rgb.copy()

    outside = ~zone_mask

    # Dim everything outside the detected zone.
    output[outside] = (
        output[outside] * 0.30
    ).astype(np.uint8)

    # Add a bright white boundary.
    padded = np.pad(
        zone_mask,
        1,
        mode="constant",
        constant_values=False,
    )

    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    boundary = zone_mask & (
        ~up
        | ~down
        | ~left
        | ~right
    )

    output[boundary] = [
        255,
        255,
        255,
    ]

    return output


def get_zone_evidence(region_id: int):
    feature = find_zone(region_id)

    if feature is None:
        raise ValueError(
            f"Zone {region_id} was not found."
        )

    metadata = load_metadata()

    (
        before,
        after,
        before_valid,
        after_valid,
    ) = load_spectral_data()

    common_valid = (
        before_valid.astype(bool)
        & after_valid.astype(bool)
    )

    spatial_reference = metadata[
        "before"
    ]["spatial_reference"]

    transform = Affine(
        *spatial_reference["transform"]
    )

    raster_shape = before.shape[1:]

    source_crs = spatial_reference["crs"]

    zone_mask = build_zone_mask(
        feature,
        raster_shape,
        transform,
        source_crs,
    )

    evidence_mask = (
        zone_mask
        & common_valid
    )

    rows, cols = np.where(
        evidence_mask
    )

    if len(rows) == 0:
        raise ValueError(
            f"Zone {region_id} has no valid pixels."
        )

    min_row = int(rows.min())
    max_row = int(rows.max())
    min_col = int(cols.min())
    max_col = int(cols.max())

    # Use combined physical percentiles so the
    # before/after visualizations remain comparable.
    combined_percentiles = []

    for band_index in [2, 1, 0]:
        before_values = before[
            band_index
        ][common_valid]

        after_values = after[
            band_index
        ][common_valid]

        values = np.concatenate(
            [
                before_values,
                after_values,
            ]
        )

        values = values[
            np.isfinite(values)
        ]

        if values.size == 0:
            combined_percentiles.append(
                (0.0, 1.0)
            )
        else:
            combined_percentiles.append(
                tuple(
                    np.percentile(
                        values,
                        [2, 98],
                    )
                )
            )

    before_rgb = make_rgb(
        before,
        common_valid,
        combined_percentiles,
    )

    after_rgb = make_rgb(
        after,
        common_valid,
        combined_percentiles,
    )

    zone_crop_mask, crop_bounds = (
        crop_with_padding(
            zone_mask,
            min_row,
            min_col,
            max_row,
            max_col,
            padding=15,
        )
    )

    before_crop, _ = crop_with_padding(
        before_rgb,
        min_row,
        min_col,
        max_row,
        max_col,
        padding=15,
    )

    after_crop, _ = crop_with_padding(
        after_rgb,
        min_row,
        min_col,
        max_row,
        max_col,
        padding=15,
    )

    # Zone-only mask.
    mask_crop = (
        zone_crop_mask.astype(
            np.uint8
        ) * 255
    )

    # White zone boundary over the before image.
    overlay = create_zone_mask_overlay(
        before_crop,
        zone_crop_mask,
    )
        # ---------------------------------------------------------
    # Multispectral zone evidence
    # ---------------------------------------------------------
    #
    # Spectral band order:
    #   0 = B02 Blue
    #   1 = B03 Green
    #   2 = B04 Red
    #   3 = B08 NIR
    #
    # Calculate statistics only from pixels belonging
    # to this detected zone and valid in both dates.
    # ---------------------------------------------------------

    zone_values_before = before[
        :, evidence_mask
    ]

    zone_values_after = after[
        :, evidence_mask
    ]

    band_names = [
        "B02",
        "B03",
        "B04",
        "B08",
    ]

    band_profile = {}

    for band_index, band_name in enumerate(
        band_names
    ):
        before_values = zone_values_before[
            band_index
        ]

        after_values = zone_values_after[
            band_index
        ]

        band_profile[band_name] = {
            "before": float(
                np.mean(before_values)
            ),
            "after": float(
                np.mean(after_values)
            ),
            "change": float(
                np.mean(after_values)
                - np.mean(before_values)
            ),
        }

    # Per-pixel NDVI.
    before_red = before[
        2
    ][evidence_mask]

    before_nir = before[
        3
    ][evidence_mask]

    after_red = after[
        2
    ][evidence_mask]

    after_nir = after[
        3
    ][evidence_mask]

    before_ndvi = (
        (before_nir - before_red)
        / (
            before_nir
            + before_red
            + 1e-8
        )
    )

    after_ndvi = (
        (after_nir - after_red)
        / (
            after_nir
            + after_red
            + 1e-8
        )
    )

    ndvi_change = (
        after_ndvi - before_ndvi
    )

    ndvi_profile = {
        "before": float(
            np.mean(before_ndvi)
        ),
        "after": float(
            np.mean(after_ndvi)
        ),
        "change": float(
            np.mean(ndvi_change)
        ),
        "absolute_change": float(
            np.mean(
                np.abs(ndvi_change)
            )
        ),
    }

    # Physical spectral distance across
    # B02/B03/B04/B08.
    spectral_distance = np.sqrt(
        np.sum(
            (
                zone_values_after
                - zone_values_before
            )
            ** 2,
            axis=0,
        )
    )

    spectral_profile = {
        "mean": float(
            np.mean(spectral_distance)
        ),
        "median": float(
            np.median(spectral_distance)
        ),
        "maximum": float(
            np.max(spectral_distance)
        ),
    }


    properties = feature.get(
        "properties",
        {},
    )

    return {
        "region_id": region_id,

        "bounds": {
            "min_row": crop_bounds[0],
            "min_col": crop_bounds[1],
            "max_row": crop_bounds[2],
            "max_col": crop_bounds[3],
        },

        "properties": properties,

        "band_profile": band_profile,

        "ndvi_profile": ndvi_profile,

        "spectral_profile": spectral_profile,

        "before_png": encode_png(
            before_crop
        ),

        "after_png": encode_png(
            after_crop
        ),

        "mask_png": encode_mask(
            zone_crop_mask
        ),

        "overlay_png": encode_png(
            overlay
        ),
    }