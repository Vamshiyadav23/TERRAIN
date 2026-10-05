import base64
import json
import urllib.request
from typing import Any


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b"


ALLOWED_CLASSES = {
    "vegetation_loss",
    "vegetation_gain",
    "new_construction",
    "demolition",
    "road_infrastructure",
    "water_change",
    "land_clearing",
    "agricultural_change",
    "spectral_change",
    "other",
    "uncertain",
}


SYSTEM_PROMPT = """
You are the semantic interpretation component of TERRAIN,
an AI-driven satellite image change intelligence system.

Your task is NOT to detect whether pixels changed.

A numerical remote-sensing detector has already identified
a geographically localized change zone.

Your task is to interpret what the detected change most
likely represents using:

1. BEFORE satellite image
2. AFTER satellite image
3. CHANGE ZONE overlay
4. Multispectral evidence
5. NDVI evidence
6. Spectral contrast
7. Change-zone metadata

You MUST distinguish numerical detection from semantic
interpretation.

Do not claim that the change detector is wrong merely
because the RGB imagery is visually ambiguous.

Do not invent objects, buildings, roads, water bodies,
vegetation, or construction that cannot reasonably be
supported by the supplied evidence.

If the visual evidence is insufficient, use "uncertain".

Possible classifications:

vegetation_loss
vegetation_gain
new_construction
demolition
road_infrastructure
water_change
land_clearing
agricultural_change
spectral_change
other
uncertain

Important:
CONSISTENCY RULES:

CONSISTENCY RULES:

- If classification is "uncertain", needs_review MUST be true.
- If needs_review is true, confidence should generally be below 0.75.
- Do not describe vegetation loss when NDVI change is approximately zero
  unless strong visual evidence independently supports vegetation loss.
- Do not describe vegetation gain when NDVI change is approximately zero
  unless strong visual evidence independently supports vegetation gain.
- Do not classify vegetation loss solely because spectral bands changed.
- Do not classify vegetation gain solely because spectral bands changed.
- NDVI decline alone is not proof of deforestation.
- NDVI increase alone is not proof of vegetation recovery.
- Treat absolute NDVI changes smaller than 0.05 as approximately stable
  unless strong visual evidence supports a specific interpretation.
- If the numerical direction is "spectral_change" and absolute NDVI
  change is smaller than 0.05, prefer "spectral_change" or "uncertain".
- If NDVI change is strongly negative and visual evidence supports
  vegetation removal, vegetation_loss may be selected.
- If NDVI change is strongly positive and visual evidence supports
  increased vegetation, vegetation_gain may be selected.
- The semantic interpretation confidence must reflect ambiguity.
- Never return classification="uncertain" together with
  needs_review=false.
  - When the numerical direction is "vegetation_gain", the NDVI change
  must be interpreted as an increase when it is positive.
- When the numerical direction is "vegetation_loss", the NDVI change
  must be interpreted as a decrease when it is negative.
- Never write "decreased" or "decrease" when describing a positive
  NDVI change.
- Never write "increased" or "increase" when describing a negative
  NDVI change.
- The summary must agree with the supplied numerical NDVI before,
  NDVI after, and NDVI change values.
- The visual_evidence descriptions must not contradict the numerical
  evidence.

Do not equate NDVI decline automatically with deforestation.
Consider the complete evidence.

Return ONLY valid JSON.
"""


def _strip_data_uri(value: str) -> str:
    """
    Convert:
        data:image/png;base64,AAAA...
    into:
        AAAA...
    """

    if value.startswith("data:"):
        return value.split(",", 1)[1]

    return value


def _image_bytes_to_base64(
    image_bytes: bytes,
) -> str:
    return base64.b64encode(
        image_bytes
    ).decode("utf-8")


def _post_json(
    payload: dict[str, Any],
    timeout: int = 180,
) -> dict[str, Any]:

    body = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:

        response_body = response.read()

    return json.loads(
        response_body.decode("utf-8")
    )


def analyze_change_zone(
    *,
    before_image: bytes,
    after_image: bytes,
    overlay_image: bytes,
    zone_context: dict[str, Any],
) -> dict[str, Any]:

    before_b64 = _image_bytes_to_base64(
        before_image
    )

    after_b64 = _image_bytes_to_base64(
        after_image
    )

    overlay_b64 = _image_bytes_to_base64(
        overlay_image
    )

    user_prompt = f"""
Analyze the selected TERRAIN change zone.

ZONE INFORMATION
----------------
Region ID:
{zone_context.get("region_id")}

Area:
{zone_context.get("area_m2")} m²

Pixel count:
{zone_context.get("pixel_count")}

Numerical detector confidence:
{zone_context.get("confidence")}

Detector severity:
{zone_context.get("severity")}

Detector direction:
{zone_context.get("direction")}

MULTISPECTRAL EVIDENCE
----------------------
B02 Blue:
before={zone_context.get("B02_before")}
after={zone_context.get("B02_after")}
change={zone_context.get("B02_change")}

B03 Green:
before={zone_context.get("B03_before")}
after={zone_context.get("B03_after")}
change={zone_context.get("B03_change")}

B04 Red:
before={zone_context.get("B04_before")}
after={zone_context.get("B04_after")}
change={zone_context.get("B04_change")}

B08 NIR:
before={zone_context.get("B08_before")}
after={zone_context.get("B08_after")}
change={zone_context.get("B08_change")}

NDVI:
before={zone_context.get("ndvi_before")}
after={zone_context.get("ndvi_after")}
change={zone_context.get("ndvi_change")}
absolute_change={zone_context.get("ndvi_absolute_change")}

Spectral contrast:
mean={zone_context.get("spectral_mean")}
median={zone_context.get("spectral_median")}
maximum={zone_context.get("spectral_maximum")}

Interpret the visual and numerical evidence together.

IMPORTANT NUMERICAL DIRECTION CHECK:

NDVI change = after NDVI - before NDVI.

If NDVI change is positive, NDVI increased.
If NDVI change is negative, NDVI decreased.

For this zone, the numerical detector direction is authoritative.

Your summary and visual_evidence MUST agree with:
- detector direction
- NDVI before
- NDVI after
- NDVI change

Do not describe a positive NDVI change as a decrease.
Do not describe a negative NDVI change as an increase.

Do not invent visual observations that are not visible.

Return JSON using exactly this structure:

{{
  "classification": "one allowed classification",
  "confidence": 0.0,
  "summary": "short evidence-based explanation",
  "visual_evidence": [
    "observation from the before/after imagery",
    "observation from the change-zone overlay"
  ],
  "spectral_consistency": "strong | moderate | weak",
  "alternative_explanations": [
    "possible alternative explanation"
  ],
  "needs_review": false
}}

The confidence is the confidence of the semantic interpretation,
NOT the numerical detector confidence.

Use a value between 0 and 1.

If the evidence does not support a specific interpretation,
use classification "uncertain".
"""

    payload = {
        "model": MODEL_NAME,
        "stream": False,

        "system": SYSTEM_PROMPT,

        "messages": [
            {
                "role": "user",
                "content": user_prompt,
                "images": [
                    before_b64,
                    after_b64,
                    overlay_b64,
                ],
            }
        ],

        "format": "json",

        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    response = _post_json(
        payload
    )

    message = response.get(
        "message",
        {},
    )

    content = message.get(
        "content",
        "",
    )

    if not content:
        raise RuntimeError(
            "Qwen returned an empty response."
        )

    try:
        result = json.loads(content)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Qwen returned invalid JSON: "
            f"{content}"
        ) from exc

    return validate_semantic_result(
        result
    )


def validate_semantic_result(
    result: dict[str, Any],
) -> dict[str, Any]:

    classification = str(
        result.get(
            "classification",
            "uncertain",
        )
    ).strip().lower()

    if classification not in ALLOWED_CLASSES:
        classification = "uncertain"

    try:
        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        confidence = 0.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    summary = str(
        result.get(
            "summary",
            "",
        )
    ).strip()

    visual_evidence = result.get(
        "visual_evidence",
        [],
    )

    if not isinstance(
        visual_evidence,
        list,
    ):
        visual_evidence = []

    visual_evidence = [
        str(item).strip()
        for item in visual_evidence
        if str(item).strip()
    ]

    spectral_consistency = str(
        result.get(
            "spectral_consistency",
            "weak",
        )
    ).strip().lower()

    if spectral_consistency not in {
        "strong",
        "moderate",
        "weak",
    }:
        spectral_consistency = "weak"

    alternatives = result.get(
        "alternative_explanations",
        [],
    )

    if not isinstance(
        alternatives,
        list,
    ):
        alternatives = []

    alternatives = [
        str(item).strip()
        for item in alternatives
        if str(item).strip()
    ]

    needs_review = bool(
        result.get(
            "needs_review",
            False,
        )
    )

    return {
        "classification": classification,
        "confidence": confidence,
        "summary": summary,
        "visual_evidence": visual_evidence,
        "spectral_consistency": spectral_consistency,
        "alternative_explanations": alternatives,
        "needs_review": needs_review,
    }