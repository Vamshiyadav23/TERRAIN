from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic convolution block
# ============================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
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


# ============================================================
# Temporal fusion
# ============================================================

class TemporalFusion(nn.Module):

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        # BEFORE + AFTER + ABSOLUTE DIFFERENCE
        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:

        difference = torch.abs(
            before - after
        )

        fused = torch.cat(
            [
                before,
                after,
                difference,
            ],
            dim=1,
        )

        return self.fusion(fused)


# ============================================================
# Siamese temporal encoder
# ============================================================

class SiameseTemporalEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = ConvBlock(
            3,
            32,
        )

        self.block2 = ConvBlock(
            32,
            64,
        )

        self.block3 = ConvBlock(
            64,
            128,
        )

        self.block4 = ConvBlock(
            128,
            256,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fusion1 = TemporalFusion(32)
        self.fusion2 = TemporalFusion(64)
        self.fusion3 = TemporalFusion(128)
        self.fusion4 = TemporalFusion(256)

    def encode(
        self,
        x: torch.Tensor,
    ):

        f1 = self.block1(x)

        x = self.pool(f1)

        f2 = self.block2(x)

        x = self.pool(f2)

        f3 = self.block3(x)

        x = self.pool(f3)

        f4 = self.block4(x)

        return f1, f2, f3, f4

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ):

        before_f1, before_f2, before_f3, before_f4 = (
            self.encode(before)
        )

        after_f1, after_f2, after_f3, after_f4 = (
            self.encode(after)
        )

        fused1 = self.fusion1(
            before_f1,
            after_f1,
        )

        fused2 = self.fusion2(
            before_f2,
            after_f2,
        )

        fused3 = self.fusion3(
            before_f3,
            after_f3,
        )

        fused4 = self.fusion4(
            before_f4,
            after_f4,
        )

        return (
            fused1,
            fused2,
            fused3,
            fused4,
        )


# ============================================================
# Decoder
# ============================================================

class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
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


# ============================================================
# V2 model
# ============================================================

class SiameseUNetV2(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError(
                "LEVIR-CD+ V2 expects RGB input."
            )

        self.encoder = SiameseTemporalEncoder()

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

    def forward(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:

        fused1, fused2, fused3, fused4 = (
            self.encoder(
                before,
                after,
            )
        )

        x = self.decoder3(
            fused4,
            fused3,
        )

        x = self.decoder2(
            x,
            fused2,
        )

        x = self.decoder1(
            x,
            fused1,
        )

        return self.head(x)


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SiameseUNetV2().to(device)

    before = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    after = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():

        output = model(
            before,
            after,
        )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 70)
    print("TERRAIN SIAMESE U-NET V2 SMOKE TEST")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Input: {before.shape}")
    print(f"Output: {output.shape}")
    print(f"Parameters: {parameters:,}")