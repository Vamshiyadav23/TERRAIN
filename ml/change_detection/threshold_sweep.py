from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch

# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.change_detection.dataset import LEVIRCDDataset
from ml.change_detection.model import SiameseUNet


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "LEVIR-CD+"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "change_detection"
    / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "change_detection"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "threshold_sweep.json"
)

CROP_SIZE = 256

SEED = 42

VALIDATION_RATIO = 0.20

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Recreate the exact validation split
# ============================================================

def get_validation_indices(
    total_samples: int,
) -> list[int]:

    validation_size = int(
        total_samples
        * VALIDATION_RATIO
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    permutation = torch.randperm(
        total_samples,
        generator=generator,
    ).tolist()

    validation_indices = permutation[
        :validation_size
    ]

    return validation_indices


# ============================================================
# Full-image probability prediction
# ============================================================

@torch.no_grad()
def predict_probability_map(
    model: torch.nn.Module,
    before: torch.Tensor,
    after: torch.Tensor,
) -> np.ndarray:

    _, height, width = before.shape

    padded_height = (
        (
            (height + CROP_SIZE - 1)
            // CROP_SIZE
        )
        * CROP_SIZE
    )

    padded_width = (
        (
            (width + CROP_SIZE - 1)
            // CROP_SIZE
        )
        * CROP_SIZE
    )

    pad_bottom = padded_height - height
    pad_right = padded_width - width

    if pad_bottom > 0 or pad_right > 0:

        before = torch.nn.functional.pad(
            before,
            (0, pad_right, 0, pad_bottom),
            mode="constant",
            value=0,
        )

        after = torch.nn.functional.pad(
            after,
            (0, pad_right, 0, pad_bottom),
            mode="constant",
            value=0,
        )

    probability_map = torch.zeros(
        padded_height,
        padded_width,
        dtype=torch.float32,
    )

    for top in range(
        0,
        padded_height,
        CROP_SIZE,
    ):

        for left in range(
            0,
            padded_width,
            CROP_SIZE,
        ):

            before_tile = before[
                :,
                top:top + CROP_SIZE,
                left:left + CROP_SIZE,
            ]

            after_tile = after[
                :,
                top:top + CROP_SIZE,
                left:left + CROP_SIZE,
            ]

            before_tile = (
                before_tile
                .unsqueeze(0)
                .to(DEVICE)
            )

            after_tile = (
                after_tile
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = model(
                before_tile,
                after_tile,
            )

            probabilities = torch.sigmoid(
                logits
            )[0, 0].cpu()

            probability_map[
                top:top + CROP_SIZE,
                left:left + CROP_SIZE,
            ] = probabilities

    return probability_map[
        :height,
        :width,
    ].numpy()


# ============================================================
# Calculate global metrics
# ============================================================

def calculate_metrics(
    probability_maps: list[np.ndarray],
    labels: list[np.ndarray],
    threshold: float,
) -> dict[str, float]:

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for probability, label in zip(
        probability_maps,
        labels,
    ):

        prediction = (
            probability >= threshold
        )

        target = (
            label >= 0.5
        )

        total_tp += int(
            np.logical_and(
                prediction,
                target,
            ).sum()
        )

        total_fp += int(
            np.logical_and(
                prediction,
                np.logical_not(target),
            ).sum()
        )

        total_fn += int(
            np.logical_and(
                np.logical_not(prediction),
                target,
            ).sum()
        )

        total_tn += int(
            np.logical_and(
                np.logical_not(prediction),
                np.logical_not(target),
            ).sum()
        )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    iou = (
        total_tp
        / (
            total_tp
            + total_fp
            + total_fn
        )
        if (
            total_tp
            + total_fp
            + total_fn
        ) > 0
        else 0.0
    )

    total_pixels = (
        total_tp
        + total_fp
        + total_fn
        + total_tn
    )

    predicted_change_percentage = (
        (
            total_tp
            + total_fp
        )
        / total_pixels
        * 100
        if total_pixels > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "predicted_change_percentage":
            predicted_change_percentage,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "true_negative": total_tn,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TERRAIN LEVIR-CD+ VALIDATION THRESHOLD SWEEP")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = LEVIRCDDataset(
        DATASET_ROOT,
        split="train",
    )

    validation_indices = get_validation_indices(
        len(dataset)
    )

    print(
        f"Total samples: {len(dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_indices)}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model = SiameseUNet().to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Checkpoint validation F1: "
        f"{checkpoint.get('best_f1', 0.0):.4f}"
    )

    print()
    print(
        "Generating validation probability maps..."
    )
    print()

    # --------------------------------------------------------
    # Generate probability maps ONCE
    # --------------------------------------------------------

    probability_maps = []
    labels = []

    for position, dataset_index in enumerate(
        validation_indices,
        start=1,
    ):

        sample = dataset[dataset_index]

        before = sample["before"]
        after = sample["after"]
        label = sample["label"]

        probability = predict_probability_map(
            model,
            before,
            after,
        )

        probability_maps.append(
            probability
        )

        labels.append(
            label[0].numpy()
        )

        if position % 25 == 0:
            print(
                f"Processed "
                f"{position}/{len(validation_indices)} "
                f"validation images"
            )

    print()
    print(
        "Probability maps generated."
    )
    print()

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------

    results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            probability_maps,
            labels,
            threshold,
        )

        results.append(
            metrics
        )

    # --------------------------------------------------------
    # Find best threshold
    # --------------------------------------------------------

    best_result = max(
        results,
        key=lambda item: item["f1"],
    )

    # --------------------------------------------------------
    # Print table
    # --------------------------------------------------------

    print("=" * 70)
    print("VALIDATION THRESHOLD RESULTS")
    print("=" * 70)

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>11} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'IoU':>10}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['threshold']:>10.2f} "
            f"{result['precision']:>11.4f} "
            f"{result['recall']:>10.4f} "
            f"{result['f1']:>10.4f} "
            f"{result['iou']:>10.4f}"
        )

    print()
    print("=" * 70)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_result['f1']:.4f}"
    )

    print(
        f"IoU: "
        f"{best_result['iou']:.4f}"
    )

    print(
        f"Predicted change: "
        f"{best_result['predicted_change_percentage']:.2f}%"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "dataset": "LEVIR-CD+",
        "split": "validation",
        "validation_samples":
            len(validation_indices),
        "checkpoint_epoch":
            checkpoint.get("epoch"),
        "checkpoint_validation_f1":
            checkpoint.get("best_f1"),
        "thresholds_tested":
            THRESHOLDS,
        "best_threshold":
            best_result["threshold"],
        "best_result":
            best_result,
        "all_results":
            results,
    }

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

    print()
    print(
        f"Report saved:\n{REPORT_PATH}"
    )

    print()
    print("=" * 70)
    print("THRESHOLD SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()