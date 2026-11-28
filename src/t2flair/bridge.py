"""F1: Brownian-bridge flow matching from the T2 slice y to the FLAIR slice x (D-002).

Marginal:  x_t = (1 - t) y + t x + sigma * sqrt(t (1 - t)) * eps,  t ~ U(0, 1).
The network predicts the endpoint x_hat = f(x_t, t, cond) and is trained with MSE to x.
Sampling starts at x_0 = y and follows the exact Brownian-bridge transition towards x_hat:
  x_s | x_t ~ N(x_t + (s - t)/(1 - t) (x_hat - x_t),  sigma^2 (s - t)(1 - s)/(1 - t)),
whose marginals match the training interpolation when x_hat is exact. The last step returns x_hat.

F0 (noise-start control, method.name noise_fm; card 2026-11-28-f0-noise-start): the same network,
condition and x-prediction MSE loss, only the path changes. x_t = (1 - t) eps + t x, so t=0 is pure
noise and the T2 slices enter through cond only. Sampling starts at x_0 = eps and takes deterministic
Euler steps x_s = x_t + (s - t)/(1 - t) (x_hat - x_t), i.e. velocity (x_hat - x_t)/(1 - t); the last
step returns x_hat. This is the bridge mean update with sigma = 0 and a noise start.
"""
import torch
import torch.nn.functional as F


def interpolate(y, x, t, sigma, noise):
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * y + t * x + sigma * torch.sqrt(t * (1 - t)) * noise


def training_loss(model, batch, sigma, generator=None):
    y, x, cond = batch["source"], batch["target"], batch["cond"]
    t = torch.rand(x.shape[0], device=x.device, generator=generator)
    noise = torch.randn(x.shape, device=x.device, generator=generator)
    x_t = interpolate(y, x, t, sigma, noise)
    x_hat = model(x_t, t, cond)
    return F.mse_loss(x_hat.float(), x)


@torch.no_grad()
def sample(model, y, cond, steps, sigma, generator=None, clamp=True):
    """Integrate from t=0 (x=y) to t=1 in `steps` uniform steps; returns x_hat at t=1."""
    x = y.clone()
    for k in range(steps):
        t, s = k / steps, (k + 1) / steps
        x_hat = model(x, torch.full((x.shape[0],), t, device=x.device), cond).float()
        if clamp:
            x_hat = x_hat.clamp(-1, 1)
        if k == steps - 1:
            return x_hat
        mean = x + (s - t) / (1 - t) * (x_hat - x)
        std = sigma * ((s - t) * (1 - s) / (1 - t)) ** 0.5
        noise = torch.randn(x.shape, device=x.device, generator=generator)
        x = mean + std * noise
    return x


def noise_interpolate(x, t, noise):
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * noise + t * x


def noise_training_loss(model, batch, generator=None):
    """F0: same x-prediction MSE as F1, on the straight path from Gaussian noise to the FLAIR slice."""
    x, cond = batch["target"], batch["cond"]
    t = torch.rand(x.shape[0], device=x.device, generator=generator)
    noise = torch.randn(x.shape, device=x.device, generator=generator)
    x_hat = model(noise_interpolate(x, t, noise), t, cond)
    return F.mse_loss(x_hat.float(), x)


@torch.no_grad()
def noise_sample(model, shape_like, cond, steps, generator=None, clamp=True):
    """F0: Euler from x_0 ~ N(0, I) (shape of shape_like) to t=1 in `steps` uniform steps; returns x_hat at t=1."""
    x = torch.randn(shape_like.shape, device=shape_like.device, generator=generator)
    for k in range(steps):
        t, s = k / steps, (k + 1) / steps
        x_hat = model(x, torch.full((x.shape[0],), t, device=x.device), cond).float()
        if clamp:
            x_hat = x_hat.clamp(-1, 1)
        if k == steps - 1:
            return x_hat
        x = x + (s - t) / (1 - t) * (x_hat - x)
    return x
