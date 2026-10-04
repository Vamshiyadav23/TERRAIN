from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.change_detection.dataset import LEVIRCDDataset
from ml.change_detection.model_v2 import SiameseUNetV2


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
    / "change_detection_v2"
    / "best_model.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "change_detection_v2"
    / "test_evaluation.json"
)

CROP_SIZE = 256

THRESHOLD = 0.50


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("TERRAIN SIAMESE U-NET V2 TEST EVALUATION")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = LEVIRCDDataset(
        DATASET_ROOT,
        split="test",
    )

    print(
        f"Test samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SiameseUNetV2().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
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
        f"Validation F1 at checkpoint: "
        f"{checkpoint.get('best_f1', 'unknown')}"
    )

    print(
        f"Test threshold: {THRESHOLD}"
    )

    # --------------------------------------------------------
    # Global counters
    # --------------------------------------------------------

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    total_pixels = 0
    total_actual_change = 0
    total_predicted_change = 0

    # --------------------------------------------------------
    # Full-image tiled inference
    # --------------------------------------------------------

    stride = CROP_SIZE

    for index in range(len(dataset)):

        sample = dataset[index]

        before = sample["before"]
        after = sample["after"]
        label = sample["label"]

        _, height, width = before.shape

        padded_height = (
            ((height + stride - 1) // stride)
            * stride
        )

        padded_width = (
            ((width + stride - 1) // stride)
            * stride
        )

        pad_bottom = padded_height - height
        pad_right = padded_width - width

        if pad_bottom > 0 or pad_right > 0:

            before = F.pad(
                before,
                (0, pad_right, 0, pad_bottom),
            )

            after = F.pad(
                after,
                (0, pad_right, 0, pad_bottom),
            )

            label = F.pad(
                label,
                (0, pad_right, 0, pad_bottom),
            )

        for top in range(
            0,
            padded_height,
            stride,
        ):

            for left in range(
                0,
                padded_width,
                stride,
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

                label_tile = label[
                    :,
                    top:top + CROP_SIZE,
                    left:left + CROP_SIZE,
                ]

                before_tile = (
                    before_tile
                    .unsqueeze(0)
                    .to(device)
                )

                after_tile = (
                    after_tile
                    .unsqueeze(0)
                    .to(device)
                )

                label_tile = (
                    label_tile
                    .unsqueeze(0)
                    .to(device)
                )

                logits = model(
                    before_tile,
                    after_tile,
                )

                probabilities = torch.sigmoid(
                    logits
                )

                predictions = (
                    probabilities >= THRESHOLD
                )

                targets = (
                    label_tile >= 0.5
                )

                tp = (
                    predictions
                    & targets
                ).sum().item()

                fp = (
                    predictions
                    & ~targets
                ).sum().item()

                fn = (
                    ~predictions
                    & targets
                ).sum().item()

                tn = (
                    ~predictions
                    & ~targets
                ).sum().item()

                total_tp += int(tp)
                total_fp += int(fp)
                total_fn += int(fn)
                total_tn += int(tn)

                total_pixels += (
                    label_tile.numel()
                )

                total_actual_change += int(
                    targets.sum().item()
                )

                total_predicted_change += int(
                    predictions.sum().item()
                )

        if (index + 1) % 25 == 0:
            print(
                f"Evaluated "
                f"{index + 1}/{len(dataset)}"
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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
        2.0 * precision * recall
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

    accuracy = (
        (total_tp + total_tn)
        / total_pixels
        if total_pixels > 0
        else 0.0
    )

    actual_change_percentage = (
        total_actual_change
        / total_pixels
        * 100.0
    )

    predicted_change_percentage = (
        total_predicted_change
        / total_pixels
        * 100.0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {
        "model": "SiameseUNetV2",
        "dataset": "LEVIR-CD+",
        "split": "test",
        "num_samples": len(dataset),
        "threshold": THRESHOLD,
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "validation_f1": checkpoint.get(
            "best_f1"
        ),
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
        },
        "pixel_statistics": {
            "total_pixels": total_pixels,
            "actual_change_pixels":
                total_actual_change,
            "predicted_change_pixels":
                total_predicted_change,
            "actual_change_percentage":
                actual_change_percentage,
            "predicted_change_percentage":
                predicted_change_percentage,
        },
        "confusion_matrix": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V2 TEST EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Actual change: "
        f"{actual_change_percentage:.2f}%"
    )

    print(
        f"Predicted change: "
        f"{predicted_change_percentage:.2f}%"
    )

    print()

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"IoU:       {iou:.4f}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print()

    print(
        f"TP: {total_tp:,}"
    )

    print(
        f"FP: {total_fp:,}"
    )

    print(
        f"FN: {total_fn:,}"
    )

    print(
        f"TN: {total_tn:,}"
    )

    print()

    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    evaluate()
    