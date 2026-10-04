from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

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
    / "visualizations"
)

THRESHOLD = 0.5
NUM_EXAMPLES = 6
CROP_SIZE = 256


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_full_image(
    model: torch.nn.Module,
    before: torch.Tensor,
    after: torch.Tensor,
) -> np.ndarray:

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
# Tensor → RGB image
# ============================================================

def tensor_to_rgb(
    image: torch.Tensor,
) -> np.ndarray:

    image = (
        image
        .detach()
        .cpu()
        .numpy()
    )

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    image = np.clip(
        image,
        0,
        1,
    )

    return image


# ============================================================
# Create visualization
# ============================================================

def create_visualization(
    before: np.ndarray,
    after: np.ndarray,
    ground_truth: np.ndarray,
    probability: np.ndarray,
    index: int,
) -> None:

    prediction = (
        probability >= THRESHOLD
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    actual_pixels = int(
        ground_truth.sum()
    )

    predicted_pixels = int(
        prediction.sum()
    )

    intersection = np.logical_and(
        prediction,
        ground_truth,
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth,
    ).sum()

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )

    # BEFORE
    axes[0, 0].imshow(before)
    axes[0, 0].set_title(
        "Before",
        fontsize=13,
    )
    axes[0, 0].axis("off")

    # AFTER
    axes[0, 1].imshow(after)
    axes[0, 1].set_title(
        "After",
        fontsize=13,
    )
    axes[0, 1].axis("off")

    # Ground truth
    axes[0, 2].imshow(
        ground_truth,
        cmap="gray",
    )
    axes[0, 2].set_title(
        "Ground Truth",
        fontsize=13,
    )
    axes[0, 2].axis("off")

    # Probability
    axes[1, 0].imshow(
        probability,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(
        "Model Change Probability",
        fontsize=13,
    )
    axes[1, 0].axis("off")

    # Prediction
    axes[1, 1].imshow(
        prediction,
        cmap="gray",
    )
    axes[1, 1].set_title(
        "Predicted Change",
        fontsize=13,
    )
    axes[1, 1].axis("off")

    # Overlay
    axes[1, 2].imshow(after)

    axes[1, 2].imshow(
        prediction,
        cmap="Reds",
        alpha=0.45,
    )

    axes[1, 2].set_title(
        "Prediction Overlay",
        fontsize=13,
    )

    axes[1, 2].axis("off")

    fig.suptitle(
        (
            f"LEVIR-CD+ Test Sample {index} | "
            f"Actual: {actual_pixels:,} px | "
            f"Predicted: {predicted_pixels:,} px | "
            f"IoU: {iou:.3f}"
        ),
        fontsize=15,
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"test_sample_{index:04d}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("TERRAIN LEVIR-CD+ TEST VISUALIZATION")
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
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
        f"Validation F1: "
        f"{checkpoint.get('best_f1', 0):.4f}"
    )

    print()

    # --------------------------------------------------------
    # Select examples
    # --------------------------------------------------------

    # Spread examples across the test set
    if len(dataset) <= NUM_EXAMPLES:

        indices = list(
            range(len(dataset))
        )

    else:

        indices = np.linspace(
            0,
            len(dataset) - 1,
            NUM_EXAMPLES,
            dtype=int,
        ).tolist()

    print(
        "Visualizing test samples:",
        indices,
    )

    print()

    # --------------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------------

    for index in indices:

        sample = dataset[index]

        before = sample["before"]
        after = sample["after"]
        label = sample["label"]

        before_rgb = tensor_to_rgb(
            before
        )

        after_rgb = tensor_to_rgb(
            after
        )

        ground_truth = (
            label[0]
            .numpy()
            > 0.5
        )

        probability = predict_full_image(
            model,
            before,
            after,
        )

        create_visualization(
            before_rgb,
            after_rgb,
            ground_truth,
            probability,
            index,
        )

    print()
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()