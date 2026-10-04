from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.change_detection.dataset import LEVIRCDDataset
from ml.change_detection.model import (
    ChangeDetectionLoss,
    SiameseUNet,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

DATASET_ROOT = Path(
    "datasets/LEVIR-CD+"
)

CHECKPOINT_DIR = Path(
    "outputs/change_detection"
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR /
    "best_model.pt"
)

HISTORY_PATH = (
    CHECKPOINT_DIR /
    "training_history.npy"
)

CROP_SIZE = 256

BATCH_SIZE = 4

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

VALIDATION_RATIO = 0.20

NUM_WORKERS = 0

PIN_MEMORY = True


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = SEED) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # Reproducible convolution behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Change-aware crop
# ============================================================

def random_crop(
    before: torch.Tensor,
    after: torch.Tensor,
    label: torch.Tensor,
    crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    _, height, width = before.shape

    if (
        height < crop_size
        or width < crop_size
    ):
        raise ValueError(
            "Image is smaller than crop size."
        )

    top = random.randint(
        0,
        height - crop_size,
    )

    left = random.randint(
        0,
        width - crop_size,
    )

    before = before[
        :,
        top:top + crop_size,
        left:left + crop_size,
    ]

    after = after[
        :,
        top:top + crop_size,
        left:left + crop_size,
    ]

    label = label[
        :,
        top:top + crop_size,
        left:left + crop_size,
    ]

    return before, after, label


def change_aware_crop(
    before: torch.Tensor,
    after: torch.Tensor,
    label: torch.Tensor,
    crop_size: int,
    positive_probability: float = 0.65,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    _, height, width = before.shape

    changed = torch.nonzero(
        label[0] > 0.5,
        as_tuple=False,
    )

    # No positive pixels: fall back to random crop.
    if (
        len(changed) == 0
        or random.random() > positive_probability
    ):
        return random_crop(
            before,
            after,
            label,
            crop_size,
        )

    # Pick a random changed pixel.
    index = random.randrange(
        len(changed)
    )

    center_y = int(
        changed[index, 0]
    )

    center_x = int(
        changed[index, 1]
    )

    half = crop_size // 2

    top_min = max(
        0,
        center_y - half,
    )

    top_max = min(
        height - crop_size,
        center_y,
    )

    left_min = max(
        0,
        center_x - half,
    )

    left_max = min(
        width - crop_size,
        center_x,
    )

    if top_max < top_min:
        top = max(
            0,
            min(
                center_y - half,
                height - crop_size,
            ),
        )
    else:
        top = random.randint(
            top_min,
            top_max,
        )

    if left_max < left_min:
        left = max(
            0,
            min(
                center_x - half,
                width - crop_size,
            ),
        )
    else:
        left = random.randint(
            left_min,
            left_max,
        )

    before = before[
        :,
        top:top + crop_size,
        left:left + crop_size,
    ]

    after = after[
        :,
        top:top + crop_size,
        left:left + crop_size,
    ]

    label = label[
        :,
        top:top + crop_size,
        left:left + crop_size,
    ]

    return before, after, label


# ============================================================
# Augmentation
# ============================================================

def augment_pair(
    before: torch.Tensor,
    after: torch.Tensor,
    label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if random.random() < 0.5:

        before = torch.flip(
            before,
            dims=[2],
        )

        after = torch.flip(
            after,
            dims=[2],
        )

        label = torch.flip(
            label,
            dims=[2],
        )

    if random.random() < 0.5:

        before = torch.flip(
            before,
            dims=[1],
        )

        after = torch.flip(
            after,
            dims=[1],
        )

        label = torch.flip(
            label,
            dims=[1],
        )

    rotations = random.randint(
        0,
        3,
    )

    if rotations > 0:

        before = torch.rot90(
            before,
            rotations,
            dims=[1, 2],
        )

        after = torch.rot90(
            after,
            rotations,
            dims=[1, 2],
        )

        label = torch.rot90(
            label,
            rotations,
            dims=[1, 2],
        )

    return before, after, label


# ============================================================
# Batch preparation
# ============================================================

def prepare_sample(
    sample: dict,
    training: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:

    before = sample["before"]
    after = sample["after"]
    label = sample["label"]

    if training:

        before, after, label = (
            change_aware_crop(
                before,
                after,
                label,
                CROP_SIZE,
            )
        )

        before, after, label = (
            augment_pair(
                before,
                after,
                label,
            )
        )

    else:

        # Center crop for deterministic validation.
        before = TF.center_crop(
            before,
            [CROP_SIZE, CROP_SIZE],
        )

        after = TF.center_crop(
            after,
            [CROP_SIZE, CROP_SIZE],
        )

        label = TF.center_crop(
            label,
            [CROP_SIZE, CROP_SIZE],
        )

    return before, after, label


# ============================================================
# Collate functions
# ============================================================

def train_collate(
    batch: list[dict],
):
    before_batch = []
    after_batch = []
    label_batch = []

    for sample in batch:

        before, after, label = (
            prepare_sample(
                sample,
                training=True,
            )
        )

        before_batch.append(before)
        after_batch.append(after)
        label_batch.append(label)

    return (
        torch.stack(before_batch),
        torch.stack(after_batch),
        torch.stack(label_batch),
    )


def validation_collate(
    batch: list[dict],
):
    before_batch = []
    after_batch = []
    label_batch = []

    for sample in batch:

        before, after, label = (
            prepare_sample(
                sample,
                training=False,
            )
        )

        before_batch.append(before)
        after_batch.append(after)
        label_batch.append(label)

    return (
        torch.stack(before_batch),
        torch.stack(after_batch),
        torch.stack(label_batch),
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    targets = (
        targets >= 0.5
    ).float()

    tp = (
        predictions * targets
    ).sum().item()

    fp = (
        predictions * (1.0 - targets)
    ).sum().item()

    fn = (
        (1.0 - predictions) * targets
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
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


# ============================================================
# Full-image tiled validation
# ============================================================

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataset,
    indices: list[int],
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    total_loss = 0.0
    total_tiles = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    stride = CROP_SIZE

    for dataset_index in indices:

        sample = dataset[dataset_index]

        before = sample["before"]
        after = sample["after"]
        label = sample["label"]

        _, height, width = before.shape

        # ----------------------------------------------------
        # Pad image if necessary
        # ----------------------------------------------------

        padded_height = (
            ((height + stride - 1) // stride) * stride
        )

        padded_width = (
            ((width + stride - 1) // stride) * stride
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

        padded_height = before.shape[1]
        padded_width = before.shape[2]

        # ----------------------------------------------------
        # Process complete image tile-by-tile
        # ----------------------------------------------------

        for top in range(0, padded_height, stride):

            for left in range(0, padded_width, stride):

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

                # Add batch dimension
                before_tile = before_tile.unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                after_tile = after_tile.unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                label_tile = label_tile.unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )

                # ------------------------------------------------
                # Model inference
                # ------------------------------------------------

                logits = model(
                    before_tile,
                    after_tile,
                )

                loss = criterion(
                    logits,
                    label_tile,
                )

                total_loss += loss.item()
                total_tiles += 1

                # ------------------------------------------------
                # Global pixel statistics
                # ------------------------------------------------

                probabilities = torch.sigmoid(logits)

                predictions = (
                    probabilities >= 0.5
                ).float()

                targets = (
                    label_tile >= 0.5
                ).float()

                total_tp += int(
                    (
                        predictions * targets
                    ).sum().item()
                )

                total_fp += int(
                    (
                        predictions * (1.0 - targets)
                    ).sum().item()
                )

                total_fn += int(
                    (
                        (1.0 - predictions) * targets
                    ).sum().item()
                )

    if total_tiles == 0:
        raise RuntimeError(
            "Validation produced no tiles."
        )

    # --------------------------------------------------------
    # Global metrics
    # --------------------------------------------------------

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

    return {
        "loss": total_loss / total_tiles,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


# ============================================================
# Training
# ============================================================

def main():

    set_seed()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("TERRAIN LEVIR-CD+ CHANGE DETECTION TRAINING")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = LEVIRCDDataset(
        DATASET_ROOT,
        split="train",
    )

    total_samples = len(dataset)

    validation_size = int(
        total_samples
        * VALIDATION_RATIO
    )

    indices = list(
        range(total_samples)
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

    train_indices = permutation[
        validation_size:
    ]

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    
    print(
        f"Total samples: {total_samples}"
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_indices)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            PIN_MEMORY
            and device.type == "cuda"
        ),
        collate_fn=train_collate,
    )

    

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SiameseUNet().to(
        device
    )

    pos_weight = 3.0

    criterion = ChangeDetectionLoss(
        dice_weight=0.5,
        pos_weight=pos_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # CUDA mixed precision.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_f1 = -1.0

    history = []

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        batches = 0

        for before, after, labels in train_loader:

            before = before.to(
                device,
                non_blocking=True,
            )

            after = after.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):

                logits = model(
                    before,
                    after,
                )

                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
            )

            batches += 1

        train_loss = (
            running_loss / batches
        )

        validation = validate(
            model,
            dataset,
            validation_indices,
            criterion,
            device,
        )

        scheduler.step(
            validation["f1"]
        )

        current_lr = optimizer.param_groups[
            0
        ]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **validation,
            "learning_rate": current_lr,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {validation['loss']:.4f} | "
            f"Precision: {validation['precision']:.4f} | "
            f"Recall: {validation['recall']:.4f} | "
            f"F1: {validation['f1']:.4f} | "
            f"IoU: {validation['iou']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if validation["f1"] > best_f1:

            best_f1 = validation[
                "f1"
            ]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "validation_metrics": validation,
                    "config": {
                        "crop_size": CROP_SIZE,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "pos_weight": pos_weight,
                        "seed": SEED,
                    },
                },
                BEST_MODEL_PATH,
            )

            print(
                f" Saved best model "
                f"(F1={best_f1:.4f})"
            )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    np.save(
        HISTORY_PATH,
        np.array(
            history,
            dtype=object,
        ),
        allow_pickle=True,
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Best validation F1: {best_f1:.4f}"
    )
    print(
        f"Checkpoint: {BEST_MODEL_PATH}"
    )
    print(
        f"History: {HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()