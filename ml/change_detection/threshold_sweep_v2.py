from __future__ import annotations

import csv
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
    / "threshold_sweep.csv"
)

CROP_SIZE = 256

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
# Get predictions once
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    dataset,
    device,
):
    all_probabilities = []
    all_targets = []

    for index in range(len(dataset)):

        sample = dataset[index]

        before = sample["before"]
        after = sample["after"]
        label = sample["label"]

        _, height, width = before.shape

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

                logits = model(
                    before_tile,
                    after_tile,
                )

                probabilities = torch.sigmoid(
                    logits
                )

                all_probabilities.append(
                    probabilities
                    .squeeze()
                    .cpu()
                )

                all_targets.append(
                    label_tile
                    .squeeze()
                    .cpu()
                )

        if (index + 1) % 25 == 0:
            print(
                f"Processed "
                f"{index + 1}/{len(dataset)}"
            )

    probabilities = torch.cat(
        [
            p.flatten()
            for p in all_probabilities
        ]
    )

    targets = torch.cat(
        [
            t.flatten()
            for t in all_targets
        ]
    )

    return probabilities, targets


# ============================================================
# Calculate metrics
# ============================================================

def calculate_metrics(
    probabilities,
    targets,
    threshold,
):

    predictions = (
        probabilities >= threshold
    )

    targets = (
        targets >= 0.5
    )

    tp = (
        predictions & targets
    ).sum().item()

    fp = (
        predictions & ~targets
    ).sum().item()

    fn = (
        ~predictions & targets
    ).sum().item()

    tn = (
        ~predictions & ~targets
    ).sum().item()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    predicted_change_percentage = (
        predictions.float().mean().item()
        * 100.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "predicted_change_percentage":
            predicted_change_percentage,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("TERRAIN V2 THRESHOLD SWEEP")
    print("=" * 70)

    print(
        f"Device: {device}"
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
        f"{checkpoint.get('epoch')}"
    )

    print(
        f"Validation F1: "
        f"{checkpoint.get('best_f1'):.4f}"
    )

    # --------------------------------------------------------
    # Generate probabilities ONCE
    # --------------------------------------------------------

    print()
    print(
        "Generating probability maps..."
    )

    probabilities, targets = (
        collect_predictions(
            model,
            dataset,
            device,
        )
    )

    print(
        f"Total evaluated pixels: "
        f"{len(targets):,}"
    )

    actual_change_percentage = (
        targets.float().mean().item()
        * 100.0
    )

    print(
        f"Actual change: "
        f"{actual_change_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Sweep thresholds
    # --------------------------------------------------------

    results = []

    print()
    print(
        "-" * 70
    )

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'IoU':>10} "
        f"{'Pred %':>10}"
    )

    print(
        "-" * 70
    )

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            probabilities,
            targets,
            threshold,
        )

        results.append(metrics)

        print(
            f"{threshold:>10.2f} "
            f"{metrics['precision']:>12.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} "
            f"{metrics['iou']:>10.4f} "
            f"{metrics['predicted_change_percentage']:>9.2f}%"
        )

    # --------------------------------------------------------
    # Best threshold
    # --------------------------------------------------------

    best_f1_result = max(
        results,
        key=lambda x: x["f1"],
    )

    best_iou_result = max(
        results,
        key=lambda x: x["iou"],
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()

        writer.writerows(results)

    # --------------------------------------------------------
    # Print best
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V2 THRESHOLD SWEEP COMPLETE")
    print("=" * 70)

    print()
    print(
        "BEST F1 THRESHOLD"
    )

    print(
        f"Threshold: "
        f"{best_f1_result['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_f1_result['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_f1_result['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_f1_result['f1']:.4f}"
    )

    print(
        f"IoU: "
        f"{best_f1_result['iou']:.4f}"
    )

    print(
        f"Predicted change: "
        f"{best_f1_result['predicted_change_percentage']:.2f}%"
    )

    print()
    print(
        "BEST IoU THRESHOLD"
    )

    print(
        f"Threshold: "
        f"{best_iou_result['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_iou_result['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_iou_result['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_iou_result['f1']:.4f}"
    )

    print(
        f"IoU: "
        f"{best_iou_result['iou']:.4f}"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
    