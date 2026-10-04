from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two convolution layers with BatchNorm and ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Convolutional encoder block followed by 2x downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            in_channels,
            out_channels,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.conv(x)
        pooled = self.pool(features)

        return features, pooled


class SiameseEncoder(nn.Module):
    """
    Shared-weight encoder.

    The exact same encoder processes the before
    and after images.
    """

    def __init__(
        self,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.block1 = EncoderBlock(
            in_channels,
            32,
        )

        self.block2 = EncoderBlock(
            32,
            64,
        )

        self.block3 = EncoderBlock(
            64,
            128,
        )

        self.bottleneck = ConvBlock(
            128,
            256,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        skip1, x = self.block1(x)
        skip2, x = self.block2(x)
        skip3, x = self.block3(x)

        x = self.bottleneck(x)

        return (
            skip1,
            skip2,
            skip3,
            x,
        )


class DecoderBlock(nn.Module):
    """
    Upsampling decoder block.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.conv(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for binary change detection.

    Inputs:
        before: [B, 3, H, W]
        after:  [B, 3, H, W]

    Output:
        logits: [B, 1, H, W]

    The encoder weights are shared between
    before and after images.
    """

    def __init__(
        self,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.encoder = SiameseEncoder(
            in_channels=in_channels,
        )

        # Feature fusion happens through absolute
        # differences between the two temporal states.
        self.decoder3 = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.decoder2 = DecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        self.decoder1 = DecoderBlock(
            in_channels=64,
            skip_channels=32,
            out_channels=32,
        )

        self.head = nn.Conv2d(
            32,
            1,
            kernel_size=1,
        )

    @staticmethod
    def difference(
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        """
        Absolute temporal feature difference.
        """

        return torch.abs(
            before - after
        )

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:

        (
            before_skip1,
            before_skip2,
            before_skip3,
            before_bottleneck,
        ) = self.encoder(before)

        (
            after_skip1,
            after_skip2,
            after_skip3,
            after_bottleneck,
        ) = self.encoder(after)

        # Temporal differences at multiple scales.
        skip1 = self.difference(
            before_skip1,
            after_skip1,
        )

        skip2 = self.difference(
            before_skip2,
            after_skip2,
        )

        skip3 = self.difference(
            before_skip3,
            after_skip3,
        )

        bottleneck = self.difference(
            before_bottleneck,
            after_bottleneck,
        )

        x = self.decoder3(
            bottleneck,
            skip3,
        )

        x = self.decoder2(
            x,
            skip2,
        )

        x = self.decoder1(
            x,
            skip1,
        )

        logits = self.head(x)

        # Ensure output exactly matches input spatial dimensions.
        logits = F.interpolate(
            logits,
            size=before.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return logits


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    """

    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.reshape(
            probabilities.shape[0],
            -1,
        )

        targets = targets.reshape(
            targets.shape[0],
            -1,
        )

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        return 1.0 - dice.mean()


class ChangeDetectionLoss(nn.Module):
    """
    Combined BCE + Dice loss.

    BCE provides stable pixel-level optimization.
    Dice addresses the strong foreground/background
    imbalance in change-detection masks.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()

        self.dice_weight = dice_weight

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                torch.tensor(
                    [pos_weight],
                    dtype=torch.float32,
                ),
            )
        else:
            self.pos_weight = None

        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=self.pos_weight,
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits,
                targets,
            )

        dice = self.dice(
            logits,
            targets,
        )

        return (
            (1.0 - self.dice_weight) * bce
            + self.dice_weight * dice
        )