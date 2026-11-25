"""Networks: the generator UNet shared by all methods, and the two segmenters."""
import torch
from torch import nn


class Generator(nn.Module):
    """diffusers UNet2DModel on cat(x_t, cond); time t in [0, 1] is passed as 1000 * t."""

    def __init__(self, cfg_model, cond_channels=3, out_channels=1, sample_size=192):
        super().__init__()
        from diffusers import UNet2DModel
        base, mult = cfg_model["base_channels"], cfg_model["channel_mult"]
        attn = set(cfg_model["attn_levels"])
        levels = range(len(mult))
        self.net = UNet2DModel(
            sample_size=sample_size,
            in_channels=out_channels + cond_channels,
            out_channels=out_channels,
            layers_per_block=cfg_model["layers_per_block"],
            block_out_channels=tuple(base * m for m in mult),
            down_block_types=tuple("AttnDownBlock2D" if i in attn else "DownBlock2D" for i in levels),
            up_block_types=tuple("AttnUpBlock2D" if i in attn else "UpBlock2D" for i in reversed(levels)),
            norm_num_groups=min(32, base),
            # channels per attention head; 8 is the diffusers default the approved trunk was smoked with
            attention_head_dim=cfg_model.get("attn_head_dim", 8),
        )

    def forward(self, x_t, t, cond):
        return self.net(torch.cat([x_t, cond], dim=1), t * 1000.0).sample


def build_segmenter(arch):
    """1-channel FLAIR slice -> 1-channel whole-tumor logit. S_train and S_eval differ in arch and seed."""
    if arch == "basicunet":
        from monai.networks.nets import BasicUNet
        return BasicUNet(spatial_dims=2, in_channels=1, out_channels=1, features=(32, 32, 64, 128, 256, 32))
    if arch == "segresnet":
        from monai.networks.nets import SegResNet
        return SegResNet(spatial_dims=2, in_channels=1, out_channels=1, init_filters=32,
                         blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1))
    if arch == "tiny":  # tests and smoke runs only
        return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 3, padding=1))
    raise ValueError(arch)


def count_params(module):
    return sum(p.numel() for p in module.parameters())
