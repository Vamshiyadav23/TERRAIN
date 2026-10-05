from __future__ import annotations

import argparse
import base64
import json

from backend.evidence import get_zone_evidence
from ml.semantic.qwen import analyze_change_zone


# ============================================================
# HELPERS
# ============================================================

def image_to_bytes(value) -> bytes:
    """
    Convert an evidence image into raw PNG bytes.

    evidence.py may return:
        - bytes
        - bytearray
        - base64 data URI
        - plain base64 string
    """

    if isinstance(value, bytes):
        return value

    if isinstance(value, bytearray):
        return bytes(value)

    if isinstance(value, str):

        if value.startswith("data:"):
            value = value.split(",", 1)[1]

        return base64.b64decode(value)

    raise TypeError(
        f"Unsupported image type: {type(value)}"
    )


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# NUMERICAL CONTEXT
# ============================================================

def build_numerical_context(
    evidence: dict,
    region_id: int,
) -> dict:
    """
    Extract authoritative numerical metrics from the
    production spectral-change GeoJSON.
    """

    properties = evidence.get(
        "properties",
        {},
    )

    change_evidence = properties.get(
        "change_evidence",
        {},
    )

    ndvi_evidence = properties.get(
        "ndvi_evidence",
        {},
    )

    return {
        "region_id": region_id,

        "area_m2": safe_float(
            properties.get("area_m2")
        ),

        "pixel_count": safe_int(
            properties.get("pixel_count")
        ),

        "confidence": safe_float(
            properties.get("confidence")
        ),

        "severity": properties.get(
            "severity",
            "unknown",
        ),

        "direction": properties.get(
            "direction",
            "mixed_or_uncertain",
        ),

        "ndvi_before": safe_float(
            ndvi_evidence.get("before")
        ),

        "ndvi_after": safe_float(
            ndvi_evidence.get("after")
        ),

        "ndvi_change": safe_float(
            ndvi_evidence.get("change")
        ),

        "ndvi_absolute_change": safe_float(
            ndvi_evidence.get(
                "absolute_change"
            )
        ),

        "spectral_mean": safe_float(
            change_evidence.get(
                "mean_spectral_distance"
            )
        ),

        "spectral_median": safe_float(
            change_evidence.get(
                "median_spectral_distance"
            )
        ),

        "mean_change_score": safe_float(
            change_evidence.get(
                "mean_change_score"
            )
        ),
    }


# ============================================================
# BAND CONTEXT
# ============================================================

def build_band_context(
    evidence: dict,
) -> dict:
    """
    Convert evidence.py band profiles into the flat
    structure expected by qwen.py.
    """

    band_profile = evidence.get(
        "band_profile",
        {},
    )

    spectral_profile = evidence.get(
        "spectral_profile",
        {},
    )

    context = {}

    for band in (
        "B02",
        "B03",
        "B04",
        "B08",
    ):

        band_data = band_profile.get(
            band,
            {},
        )

        if not isinstance(
            band_data,
            dict,
        ):
            band_data = {}

        context[
            f"{band}_before"
        ] = safe_float(
            band_data.get("before")
        )

        context[
            f"{band}_after"
        ] = safe_float(
            band_data.get("after")
        )

        context[
            f"{band}_change"
        ] = safe_float(
            band_data.get("change")
        )

    context["spectral_maximum"] = safe_float(
        spectral_profile.get(
            "maximum"
        )
    )

    return context


# ============================================================
# SEMANTIC VALIDATION
# ============================================================

def validate_semantic_interpretation(
    result: dict,
    numerical: dict,
) -> dict:
    """
    Validate Qwen's semantic interpretation against
    authoritative numerical remote-sensing evidence.

    The numerical pipeline determines the detected
    change direction.

    Qwen provides semantic interpretation.

    This validator prevents obvious contradictions.
    """

    direction = numerical.get(
        "direction",
        "mixed_or_uncertain",
    )

    ndvi_change = safe_float(
        numerical.get("ndvi_change")
    )

    spectral_consistency = result.get(
        "spectral_consistency",
        "weak",
    )

    if spectral_consistency not in {
        "strong",
        "moderate",
        "weak",
    }:
        spectral_consistency = "weak"

        result[
            "spectral_consistency"
        ] = spectral_consistency

    # --------------------------------------------------------
    # Strong vegetation loss
    # --------------------------------------------------------

    if (
        direction == "vegetation_loss"
        and ndvi_change <= -0.15
        and spectral_consistency
        in {"strong", "moderate"}
    ):
        result[
            "classification"
        ] = "vegetation_loss"

        result[
            "needs_review"
        ] = False

    # --------------------------------------------------------
    # Strong vegetation gain
    # --------------------------------------------------------

    elif (
        direction == "vegetation_gain"
        and ndvi_change >= 0.15
        and spectral_consistency
        in {"strong", "moderate"}
    ):
        result[
            "classification"
        ] = "vegetation_gain"

        result[
            "needs_review"
        ] = False

    # --------------------------------------------------------
    # Spectral change with approximately stable NDVI
    # --------------------------------------------------------

    elif (
        direction == "spectral_change"
        and abs(ndvi_change) < 0.05
    ):
        result[
            "classification"
        ] = "spectral_change"

        result[
            "needs_review"
        ] = True

    # --------------------------------------------------------
    # Ambiguous evidence
    # --------------------------------------------------------

    elif direction in {
        "mixed_or_uncertain",
        "spectral_change",
    }:

        if abs(ndvi_change) < 0.05:

            result[
                "classification"
            ] = "uncertain"

            result[
                "needs_review"
            ] = True

    # --------------------------------------------------------
    # Prevent small NDVI changes from being called
    # vegetation loss/gain
    # --------------------------------------------------------

    if (
        abs(ndvi_change) < 0.05
        and result.get(
            "classification"
        ) in {
            "vegetation_loss",
            "vegetation_gain",
        }
    ):
        result[
            "classification"
        ] = "spectral_change"

        result[
            "needs_review"
        ] = True

    # --------------------------------------------------------
    # Uncertain always requires review
    # --------------------------------------------------------

    if result.get(
        "classification"
    ) == "uncertain":

        result[
            "needs_review"
        ] = True

        confidence = safe_float(
            result.get(
                "confidence",
                0.5,
            ),
            0.5,
        )

        result[
            "confidence"
        ] = min(
            confidence,
            0.75,
        )

    return result


# ============================================================
# ANALYZE ZONE
# ============================================================

def analyze_zone(
    region_id: int,
) -> dict:
    """
    Run semantic interpretation for one TERRAIN
    change zone.
    """

    # --------------------------------------------------------
    # 1. Load evidence
    # --------------------------------------------------------

    evidence = get_zone_evidence(
        region_id
    )

    # --------------------------------------------------------
    # 2. Build authoritative numerical context
    # --------------------------------------------------------

    numerical = build_numerical_context(
        evidence,
        region_id,
    )

    # --------------------------------------------------------
    # 3. Build multispectral context
    # --------------------------------------------------------

    band_context = build_band_context(
        evidence
    )

    # --------------------------------------------------------
    # 4. Combine context
    # --------------------------------------------------------

    zone_context = {
        **numerical,
        **band_context,
    }

    # --------------------------------------------------------
    # 5. Get images from the ACTUAL evidence.py structure
    #
    # evidence.py returns:
    #
    # before_png
    # after_png
    # mask_png
    # overlay_png
    # --------------------------------------------------------

    before_png = evidence.get(
        "before_png"
    )

    after_png = evidence.get(
        "after_png"
    )

    overlay_png = evidence.get(
        "overlay_png"
    )

    if before_png is None:
        raise RuntimeError(
            "Zone evidence does not contain "
            "a BEFORE image."
        )

    if after_png is None:
        raise RuntimeError(
            "Zone evidence does not contain "
            "an AFTER image."
        )

    if overlay_png is None:
        raise RuntimeError(
            "Zone evidence does not contain "
            "an OVERLAY image."
        )

    # --------------------------------------------------------
    # 6. Convert evidence images to bytes
    # --------------------------------------------------------

    before_bytes = image_to_bytes(
        before_png
    )

    after_bytes = image_to_bytes(
        after_png
    )

    overlay_bytes = image_to_bytes(
        overlay_png
    )

    # --------------------------------------------------------
    # 7. Call Qwen
    # --------------------------------------------------------

    semantic_result = analyze_change_zone(
        before_image=before_bytes,
        after_image=after_bytes,
        overlay_image=overlay_bytes,
        zone_context=zone_context,
    )

    # --------------------------------------------------------
    # 8. Validate Qwen AFTER it returns
    # --------------------------------------------------------

    semantic_result = validate_semantic_interpretation(
        semantic_result,
        numerical,
    )

    # --------------------------------------------------------
    # 9. Return final structured result
    # --------------------------------------------------------

    return {
        "region_id": region_id,

        "numerical_evidence": {
            "area_m2": numerical[
                "area_m2"
            ],

            "pixel_count": numerical[
                "pixel_count"
            ],

            "detector_confidence": numerical[
                "confidence"
            ],

            "severity": numerical[
                "severity"
            ],

            "direction": numerical[
                "direction"
            ],

            "ndvi_before": numerical[
                "ndvi_before"
            ],

            "ndvi_after": numerical[
                "ndvi_after"
            ],

            "ndvi_change": numerical[
                "ndvi_change"
            ],

            "spectral_contrast": numerical[
                "spectral_mean"
            ],
        },

        "semantic_interpretation": semantic_result,
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen semantic interpretation "
            "for a TERRAIN change zone."
        )
    )

    parser.add_argument(
        "region_id",
        type=int,
        help="Change zone region ID",
    )

    args = parser.parse_args()

    result = analyze_zone(
        args.region_id
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )