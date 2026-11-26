"""In-house baselines on the trunk network, and the sampler shared by train and infer.

method.name (configs/exp/<id>.yaml):
  bridge      F1, Brownian-bridge FM from T2 (trunk; bridge.py)
  regression  UNet-L1: the same Generator fed x_t = T2 and t = 0, one forward pass, L1 loss
  pix2pix     regression generator + conditional 70x70 PatchGAN (vanilla GAN loss + l1_weight * L1)
The generator is identical in all three, so differences come from the training objective only.
"""
import torch
import torch.nn.functional as F
from torch import nn

from . import bridge

METHODS = ("bridge", "regression", "pix2pix")


def regress(model, source, cond):
    return model(source, torch.zeros(source.shape[0], device=source.device), cond)


@torch.no_grad()
def generate(model, source, cond, cfg, steps, generator=None):
    """FLAIR in [-1, 1] for one batch, for any method (steps is ignored by one-pass methods)."""
    name = cfg["method"]["name"]
    if name == "bridge":
        return bridge.sample(model, source, cond, steps, cfg["method"]["sigma"], generator)
    if name in ("regression", "pix2pix"):
        return regress(model, source, cond).float().clamp(-1, 1)
    raise NotImplementedError("method.name=%s" % name)


class PatchDiscriminator(nn.Module):
    """pix2pix 70x70 PatchGAN (C64-C128-C256-C512, stride 2/2/2/1) on cat(T2, FLAIR); one logit per patch."""

    def __init__(self, in_channels=2, base=64):
        super().__init__()
        layers, ch = [nn.Conv2d(in_channels, base, 4, 2, 1), nn.LeakyReLU(0.2, True)], base
        for mult, stride in ((2, 2), (4, 2), (8, 1)):
            layers += [nn.Conv2d(ch, base * mult, 4, stride, 1, bias=False), nn.BatchNorm2d(base * mult),
                       nn.LeakyReLU(0.2, True)]
            ch = base * mult
        layers.append(nn.Conv2d(ch, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, source, image):
        return self.net(torch.cat([source, image], dim=1))


def _gan(logits, real):
    return F.binary_cross_entropy_with_logits(logits.float(), torch.full_like(logits.float(), float(real)))


def regression_loss(model, batch):
    pred = regress(model, batch["source"], batch["cond"])
    return F.l1_loss(pred.float(), batch["target"]), pred


def pix2pix_step(model, disc, opt_d, batch, l1_weight):
    """One discriminator update; returns the generator loss (adversarial + l1_weight * L1) and parts."""
    l1, pred = regression_loss(model, batch)
    y, x = batch["source"], batch["target"]
    d_loss = 0.5 * (_gan(disc(y, x), True) + _gan(disc(y, pred.detach()), False))
    opt_d.zero_grad(set_to_none=True)
    d_loss.backward()
    opt_d.step()
    g_adv = _gan(disc(y, pred), True)
    return g_adv + l1_weight * l1, {"l1": float(l1), "g_adv": float(g_adv), "d": float(d_loss)}
