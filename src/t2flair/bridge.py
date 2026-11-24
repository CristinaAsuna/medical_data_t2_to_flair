"""F1: Brownian-bridge flow matching from the T2 slice y to the FLAIR slice x (D-002).

Marginal:  x_t = (1 - t) y + t x + sigma * sqrt(t (1 - t)) * eps,  t ~ U(0, 1).
The network predicts the endpoint x_hat = f(x_t, t, cond) and is trained with MSE to x.
Sampling starts at x_0 = y and follows the exact Brownian-bridge transition towards x_hat:
  x_s | x_t ~ N(x_t + (s - t)/(1 - t) (x_hat - x_t),  sigma^2 (s - t)(1 - s)/(1 - t)),
whose marginals match the training interpolation when x_hat is exact. The last step returns x_hat.
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
