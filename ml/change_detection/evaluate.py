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

DATASET_ROOT = PROJECT_ROOT / "datasets" / "LEVIR-CD+"

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

REPORT_PATH = OUTPUT_DIR / "test_evaluation.json"

CROP_SIZE = 256

THRESHOLD = 0.40


# ============================================================
# Full-image tiled prediction
# ============================================================

@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset: LEVIRCDDataset,
    device: torch.device,
) -> dict:

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    total_loss_pixels = 0

    samples = []

    for index in range(len(dataset)):

        sample = dataset[index]

        before = sample["before"]
        after = sample["after"]
        label = sample["label"]

        _, height, width = before.shape

        original_height = height
        original_width = width

        # ----------------------------------------------------
        # Pad to complete 256x256 tiles
        # ----------------------------------------------------

        padded_height = (
            ((height + CROP_SIZE - 1) // CROP_SIZE)
            * CROP_SIZE
        )

        padded_width = (
            ((width + CROP_SIZE - 1) // CROP_SIZE)
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

            label = torch.nn.functional.pad(
                label,
                (0, pad_right, 0, pad_bottom),
                mode="constant",
                value=0,
            )

        # ----------------------------------------------------
        # Prediction canvas
        # ----------------------------------------------------

        prediction_map = torch.zeros(
            (1, padded_height, padded_width),
            dtype=torch.float32,
        )

        # ----------------------------------------------------
        # Tile inference
        # ----------------------------------------------------

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

                before_tile = before_tile.unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                after_tile = after_tile.unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                logits = model(
                    before_tile,
                    after_tile,
                )

                probabilities = torch.sigmoid(
                    logits
                )[0, 0].cpu()

                prediction_map[
                    0,
                    top:top + CROP_SIZE,
                    left:left + CROP_SIZE,
                ] = probabilities

        # ----------------------------------------------------
        # Remove padding
        # ----------------------------------------------------

        prediction_map = prediction_map[
            :,
            :original_height,
            :original_width,
        ]

        target = label[
            :,
            :original_height,
            :original_width,
        ]

        predictions = (
            prediction_map >= THRESHOLD
        ).float()

        targets = (
            target >= 0.5
        ).float()

        # ----------------------------------------------------
        # Pixel statistics
        # ----------------------------------------------------

        tp = int(
            (
                predictions * targets
            ).sum().item()
        )

        fp = int(
            (
                predictions * (1.0 - targets)
            ).sum().item()
        )

        fn = int(
            (
                (1.0 - predictions) * targets
            ).sum().item()
        )

        tn = int(
            (
                (1.0 - predictions)
                * (1.0 - targets)
            ).sum().item()
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        # ----------------------------------------------------
        # Per-image statistics
        # ----------------------------------------------------

        actual_change_pixels = int(
            targets.sum().item()
        )

        predicted_change_pixels = int(
            predictions.sum().item()
        )

        samples.append(
            {
                "index": index,
                "actual_change_pixels":
                    actual_change_pixels,
                "predicted_change_pixels":
                    predicted_change_pixels,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        if (index + 1) % 25 == 0:
            print(
                f"Evaluated "
                f"{index + 1}/{len(dataset)} test images"
            )

    # ========================================================
    # Global metrics
    # ========================================================

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
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
        / (total_tp + total_fp + total_fn)
        if total_tp + total_fp + total_fn > 0
        else 0.0
    )

    accuracy = (
        (total_tp + total_tn)
        / (
            total_tp
            + total_tn
            + total_fp
            + total_fn
        )
        if (
            total_tp
            + total_tn
            + total_fp
            + total_fn
        ) > 0
        else 0.0
    )

    total_pixels = (
        total_tp
        + total_tn
        + total_fp
        + total_fn
    )

    actual_change_percentage = (
        (total_tp + total_fn)
        / total_pixels
        * 100
        if total_pixels > 0
        else 0.0
    )

    predicted_change_percentage = (
        (total_tp + total_fp)
        / total_pixels
        * 100
        if total_pixels > 0
        else 0.0
    )

    return {
        "dataset": "LEVIR-CD+",
        "split": "test",
        "test_samples": len(dataset),
        "threshold": THRESHOLD,
        "total_pixels": total_pixels,
        "actual_change_percentage":
            actual_change_percentage,
        "predicted_change_percentage":
            predicted_change_percentage,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "true_negative": total_tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "per_image": samples,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TERRAIN LEVIR-CD+ TEST SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(0)
                .total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    dataset = LEVIRCDDataset(
        DATASET_ROOT,
        split="test",
    )

    print(
        f"Test samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model = SiameseUNet().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    checkpoint_f1 = checkpoint.get(
        "best_f1",
        None,
    )

    print(
        f"Checkpoint epoch: {checkpoint_epoch}"
    )

    if checkpoint_f1 is not None:

        print(
            f"Validation F1 at checkpoint: "
            f"{checkpoint_f1:.4f}"
        )

    print(
        f"Test threshold: {THRESHOLD}"
    )

    print()
    print("Running full-image tiled evaluation...")
    print()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_dataset(
        model,
        dataset,
        device,
    )

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
            results,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print(
        f"Test samples: "
        f"{results['test_samples']}"
    )

    print(
        f"Actual change: "
        f"{results['actual_change_percentage']:.2f}%"
    )

    print(
        f"Predicted change: "
        f"{results['predicted_change_percentage']:.2f}%"
    )

    print()

    print(
        f"Precision: "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{results['f1']:.4f}"
    )

    print(
        f"IoU: "
        f"{results['iou']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{results['accuracy']:.4f}"
    )

    print()
    print(
        f"TP: {results['true_positive']}"
    )

    print(
        f"FP: {results['false_positive']}"
    )

    print(
        f"FN: {results['false_negative']}"
    )

    print(
        f"TN: {results['true_negative']}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()