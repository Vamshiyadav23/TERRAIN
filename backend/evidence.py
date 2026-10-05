import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine
from ml.spectral.preprocessing import (
    harmonize_after_to_before,
)

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
    """
    Crop a region with padding while preserving the source raster resolution.
    """
    height, width = image.shape[:2]

    min_row = max(0, int(min_row) - int(padding))
    min_col = max(0, int(min_col) - int(padding))
    max_row = min(height - 1, int(max_row) + int(padding))
    max_col = min(width - 1, int(max_col) + int(padding))

    return (
        image[min_row:max_row + 1, min_col:max_col + 1],
        (min_row, min_col, max_row, max_col),
    )


def crop_fixed_context(
    image,
    center_row,
    center_col,
    context_size=96,
):
    """
    Create a fixed-size contextual crop around a zone.

    The numerical detector remains at its native 256x256 resolution.
    This function is visualization-only. It gives small zones enough
    surrounding context to be interpretable in the dashboard.
    """
    height, width = image.shape[:2]

    context_size = int(
        min(context_size, height, width)
    )

    half = context_size // 2

    center_row = int(center_row)
    center_col = int(center_col)

    min_row = center_row - half
    min_col = center_col - half
    max_row = min_row + context_size
    max_col = min_col + context_size

    if min_row < 0:
        max_row -= min_row
        min_row = 0

    if min_col < 0:
        max_col -= min_col
        min_col = 0

    if max_row > height:
        shift = max_row - height
        min_row = max(0, min_row - shift)
        max_row = height

    if max_col > width:
        shift = max_col - width
        min_col = max(0, min_col - shift)
        max_col = width

    # Final safety adjustment.
    max_row = min(height, min_row + context_size)
    max_col = min(width, min_col + context_size)

    return (
        image[min_row:max_row, min_col:max_col],
        (min_row, min_col, max_row - 1, max_col - 1),
    )


def resize_display_image(
    image,
    size=384,
):
    """
    Upscale a visualization for dashboard/PDF presentation.

    This does not change any underlying Sentinel-2 measurements.
    """
    pil_image = Image.fromarray(
        image,
        mode="RGB",
    )

    # High-quality interpolation for the display image.
    pil_image = pil_image.resize(
        (size, size),
        Image.Resampling.LANCZOS,
    )

    # Very light sharpening after upscaling.
    # This improves edge readability without attempting
    # to create artificial satellite detail.
    from PIL import ImageFilter

    pil_image = pil_image.filter(
        ImageFilter.UnsharpMask(
            radius=1.0,
            percent=55,
            threshold=3,
        )
    )

    return np.asarray(pil_image)


def resize_display_mask(
    mask,
    size=384,
):
    """
    Resize a binary zone mask with nearest-neighbour interpolation so
    boundaries remain categorical and are not blurred.
    """
    pil_mask = Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L",
    )

    return np.asarray(
        pil_mask.resize(
            (size, size),
            Image.Resampling.NEAREST,
        )
    ) > 127

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
    Create a contextual visualization of the detected change zone.

    The zone itself is highlighted with a subtle cyan fill and
    a strong cyan boundary. Detection metrics are unaffected.
    """

    output = before_rgb.copy()

    # Dim the surrounding context slightly.
    outside = ~zone_mask

    output[outside] = (
        output[outside].astype(np.float32) * 0.42
    ).astype(np.uint8)

    # ---------------------------------------------------------
    # Semi-transparent cyan fill inside detected zone
    # ---------------------------------------------------------

    zone_pixels = output[zone_mask].astype(np.float32)

    cyan = np.array(
        [0.0, 220.0, 255.0],
        dtype=np.float32,
    )

    output[zone_mask] = (
        zone_pixels * 0.82
        + cyan * 0.18
    ).astype(np.uint8)

    # ---------------------------------------------------------
    # Detect exact zone boundary
    # ---------------------------------------------------------

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

    # Strong cyan boundary.
    output[boundary] = [
        0,
        235,
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
    harmonized_after, stable_mask = (
        harmonize_after_to_before(
            before,
            after,
            common_valid,
        )
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

    # ------------------------------------------------------------
    # VISUAL CONTEXT
    # ------------------------------------------------------------
    # Small detected zones can be only a few dozen pixels wide.
    # Cropping tightly around them produces tiny images such as
    # 55x53, which become blurry when enlarged in the dashboard.
    #
    # Use a fixed 96x96 native-pixel context around the zone centroid.
    # This changes visualization only; all numerical calculations
    # below continue to use the exact evidence_mask.
    # ------------------------------------------------------------

    center_row = int(round(float(rows.mean())))
    center_col = int(round(float(cols.mean())))

        # Calculate the visual context from the 2D zone mask.
    # This gives us true spatial row/column bounds.
    _, visual_bounds = crop_fixed_context(
        zone_mask,
        center_row,
        center_col,
        context_size=160,
    )

    visual_min_row, visual_min_col, visual_max_row, visual_max_col = (
        visual_bounds
    )

    zone_visual_mask = zone_mask[
        visual_min_row:visual_max_row + 1,
        visual_min_col:visual_max_col + 1,
    ]

    # ------------------------------------------------------------
    # Use combined physical percentiles so the
    # before/after visualizations remain comparable.
    combined_percentiles = []

    for band_index in [2, 1, 0]:
        before_values = before[
            band_index
        ][common_valid]

        after_values = harmonized_after[
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
        harmonized_after,
        common_valid,
        combined_percentiles,
    )

    # ------------------------------------------------------------
    # Extract the same fixed visual context from the RGB images.
    # ------------------------------------------------------------

    before_crop = before_rgb[
        visual_min_row:visual_max_row + 1,
        visual_min_col:visual_max_col + 1,
    ]

    after_crop = after_rgb[
        visual_min_row:visual_max_row + 1,
        visual_min_col:visual_max_col + 1,
    ]

    zone_crop_mask = zone_visual_mask

    # ------------------------------------------------------------
    # DISPLAY ENHANCEMENT
    # ------------------------------------------------------------
    # Keep native 96x96 context internally, then upscale only for
    # visual presentation. No new satellite detail is introduced.
    # ------------------------------------------------------------

    before_crop = resize_display_image(
        before_crop,
        size=384,
    )

    after_crop = resize_display_image(
        after_crop,
        size=384,
    )

    zone_crop_mask = resize_display_mask(
        zone_crop_mask,
        size=384,
    )

    # Clean categorical mask for API/PDF.
    mask_crop = (
        zone_crop_mask.astype(np.uint8) * 255
    )

    # Zone boundary over the before image.
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

    zone_values_after = harmonized_after[
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

    after_red = harmonized_after[
        2
    ][evidence_mask]

    after_nir = harmonized_after[
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
            "min_row": visual_min_row,
            "min_col": visual_min_col,
            "max_row": visual_max_row,
            "max_col": visual_max_col,
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