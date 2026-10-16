"""
modules.py -- Model components for HipMRI 2D prostate segmentation (PyTorch only).

Two networks are defined so they can be compared under identical conditions:

    BaselineUNet   A standard 2D U-Net (Ronneberger et al., MICCAI 2015) with
                   batch norm. This is the baseline the specification requires.

    ImprovedUNet   The 2D adaptation of the "Improved U-Net" of Isensee et al.
                   (BraTS 2017, arXiv:1802.10508). Differences from the baseline:
                     * context modules: pre-activation residual blocks with dropout
                     * strided 3x3 convolutions for downsampling (instead of max pool)
                     * instance normalisation and leaky ReLU (slope 0.01)
                     * localisation modules (3x3 conv then a 1x1 conv that halves channels)
                     * deep supervision: segmentation maps from three decoder levels are
                       upsampled and summed, so coarse levels contribute directly to the
                       output and receive gradient directly from the loss

Both networks return raw logits of shape (N, num_classes, H, W); softmax is applied
inside the loss / at inference.

Also defined here: DiceCELoss, the training objective shared by both models.

As required by the specification, this file depends only on PyTorch (no NumPy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Baseline: standard 2D U-Net                                                  #
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    """(3x3 conv -> BatchNorm -> ReLU) x 2, the basic U-Net building block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BaselineUNet(nn.Module):
    """
    Standard U-Net: 4 max-pool downsamplings, a bottleneck, 4 transposed-conv
    upsamplings, and a skip connection (channel concatenation) at every level.
    """

    def __init__(self, in_channels=1, num_classes=6, base=32):
        super().__init__()
        chs = [base * 2 ** i for i in range(5)]  # 32, 64, 128, 256, 512
        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.encoders.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.ups.append(nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(c * 2, c))  # c from upsampling + c from skip
        self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


# --------------------------------------------------------------------------- #
# Improved U-Net (Isensee et al. 2017), adapted to 2D                          #
# --------------------------------------------------------------------------- #
def norm_act(channels):
    """Instance norm + leaky ReLU, as used throughout the Improved U-Net."""
    return nn.Sequential(nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(0.01, inplace=True))


class ContextModule(nn.Module):
    """
    Pre-activation residual block: (norm -> lrelu -> 3x3 conv) -> dropout ->
    (norm -> lrelu -> 3x3 conv). Its output is ADDED to its input by the caller,
    so the module only has to learn a residual correction.
    """

    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            norm_act(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Dropout2d(dropout),
            norm_act(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class EncoderLevel(nn.Module):
    """3x3 conv (stride 2 except at the first level) followed by a residual context module."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.context = ContextModule(out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x + self.context(x)  # residual connection


class UpsamplingModule(nn.Module):
    """Upsample x2 then a 3x3 conv that halves the number of feature maps."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm_act(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class LocalisationModule(nn.Module):
    """3x3 conv (recombines upsampled + skip features) then a 1x1 conv that halves channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            norm_act(in_ch),
            nn.Conv2d(in_ch, out_ch, 1),
            norm_act(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ImprovedUNet(nn.Module):
    """
    Five-level encoder (16 -> 256 channels) and a decoder with deep supervision.

    With a 256x128 input the bottleneck is 16x8. Segmentation maps taken from the
    decoder at 1/4 and 1/2 resolution are upsampled and summed into the
    full-resolution map, as in the original architecture.
    """

    def __init__(self, in_channels=1, num_classes=6, base=16, dropout=0.3):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]  # 16, 32, 64, 128, 256
        self.enc1 = EncoderLevel(in_channels, c[0], stride=1)
        self.enc2 = EncoderLevel(c[0], c[1], stride=2)
        self.enc3 = EncoderLevel(c[1], c[2], stride=2)
        self.enc4 = EncoderLevel(c[2], c[3], stride=2)
        self.enc5 = EncoderLevel(c[3], c[4], stride=2)  # bottleneck

        self.up4 = UpsamplingModule(c[4], c[3])
        self.loc4 = LocalisationModule(c[3] * 2, c[3])
        self.up3 = UpsamplingModule(c[3], c[2])
        self.loc3 = LocalisationModule(c[2] * 2, c[2])
        self.up2 = UpsamplingModule(c[2], c[1])
        self.loc2 = LocalisationModule(c[1] * 2, c[1])
        self.up1 = UpsamplingModule(c[1], c[0])
        self.final_conv = nn.Sequential(nn.Conv2d(c[0] * 2, c[1], 3, padding=1), norm_act(c[1]))

        # Deep supervision: 1x1 segmentation layers at three decoder resolutions.
        self.seg3 = nn.Conv2d(c[2], num_classes, 1)  # 1/4 resolution
        self.seg2 = nn.Conv2d(c[1], num_classes, 1)  # 1/2 resolution
        self.seg1 = nn.Conv2d(c[1], num_classes, 1)  # full resolution

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d4 = self.loc4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.loc3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.loc2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.final_conv(torch.cat([self.up1(d2), e1], dim=1))

        # Sum segmentation maps from coarse to fine (deep supervision).
        s = self.seg3(d3)
        s = F.interpolate(s, scale_factor=2, mode="nearest") + self.seg2(d2)
        s = F.interpolate(s, scale_factor=2, mode="nearest") + self.seg1(d1)
        return s


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #
class DiceCELoss(nn.Module):
    """
    Cross-entropy + soft Dice loss.

    Cross-entropy gives stable per-pixel gradients, but is dominated by the large
    body/background classes. The prostate is only ~0.4% of pixels (data audit), so
    a soft Dice term computed per class -- which weights every class equally
    regardless of its size -- is added to stop the small organs being ignored.
    Background is excluded from the Dice term.
    """

    def __init__(self, num_classes=6, dice_weight=1.0, ce_weight=1.0, include_background=False, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.first_class = 0 if include_background else 1
        self.eps = eps

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target)
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)  # sum over batch and pixels -> one Dice score per class
        intersection = (probs * onehot).sum(dims)
        cardinality = (probs + onehot).sum(dims)
        dice = (2 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1 - dice[self.first_class:].mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
MODELS = {"baseline": BaselineUNet, "improved": ImprovedUNet}


def build_model(name, in_channels=1, num_classes=6):
    """Factory used by train.py / predict.py: name is 'baseline' or 'improved'."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name](in_channels=in_channels, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
