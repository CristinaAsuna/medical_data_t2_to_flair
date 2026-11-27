"""Exercise the official SelfRDB updates and sampler on project train/val slices."""
import json
import sys
from pathlib import Path
from time import perf_counter

import lightning as L
import torch
import yaml
from torch.utils.data import DataLoader

from adapter import BridgeSlices, project_prediction
from t2flair.config import ROOT, load
from t2flair.splits import load_split

UPSTREAM = ROOT / "third_party" / "SelfRDB"
sys.path.insert(0, str(UPSTREAM))
from main import BridgeRunner


class Timing(L.Callback):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        for key in ("d_loss", "g_loss/total"):
            value = trainer.callback_metrics[key]
            if not torch.isfinite(value):
                raise RuntimeError(f"Non-finite {key} at batch {batch_idx + 1}")
        if batch_idx == 9:
            torch.cuda.synchronize()
            self.start = perf_counter()
        if batch_idx == 59:
            torch.cuda.synchronize()
            self.it_s = 50 / (perf_counter() - self.start)


def main():
    config = load()
    with (UPSTREAM / "config.yaml").open() as handle:
        official = yaml.safe_load(handle)
    params = official["model"]
    params["generator_params"]["image_size"] = official["data"]["image_size"]
    L.seed_everything(official["seed_everything"], workers=True)
    model = BridgeRunner(**params)
    data = config["data"]
    train = BridgeSlices(data["processed"], load_split(data["splits"], "train")[:8])
    val = BridgeSlices(data["processed"], load_split(data["splits"], "val")[:1])
    size = official["data"]["train_batch_size"]
    loader = DataLoader(train, batch_size=size, shuffle=True, drop_last=True, num_workers=4)
    timing = Timing()
    # Each batch updates two optimizers. Keep the official cosine scheduler's T_max=50.
    trainer = L.Trainer(accelerator="gpu", devices=1, precision=32, max_epochs=50,
                        max_steps=120, callbacks=[timing], logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        limit_val_batches=0, num_sanity_val_steps=0, deterministic=True,
                        default_root_dir=str(ROOT / "scratch" / "selfrdb-smoke"))
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, train_dataloaders=loader)
    if trainer.global_step != 120:
        raise RuntimeError(f"Expected 60 generator and 60 discriminator updates, got {trainer.global_step}")
    model.eval().cuda()
    # Use interior slices and verify the full pad/crop round trip against the source volume.
    middle = len(val) // 2
    inputs = []
    for index in range(middle, middle + size):
        target, source, _ = val[index]
        volume_index, z = val.index[index]
        for tensor, key in ((target, "flair"), (source, "t2")):
            original = torch.from_numpy(val.vols[volume_index][key][z]).float()
            if not torch.equal(project_prediction(tensor)[0], original):
                raise RuntimeError(f"Misaligned {key} at slice {z}")
        inputs.append(source)
    batch = torch.stack(inputs).cuda()
    L.seed_everything(0)
    torch.cuda.synchronize()
    start = perf_counter()
    prediction = project_prediction(model.diffusion.sample_x0(batch, model.generator))
    torch.cuda.synchronize()
    infer_sec = perf_counter() - start
    if prediction.shape != (size, 1, 192, 192) or not torch.isfinite(prediction).all():
        raise RuntimeError("Invalid prediction shape or non-finite sampled output")
    print(json.dumps(dict(generator_updates=60, batch=size, it_s=timing.it_s,
                          train_hours_50k=50000 / timing.it_s / 3600,
                          infer_sec_per_slice=infer_sec / size,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          output_range=[prediction.min().item(), prediction.max().item()])))


if __name__ == "__main__":
    main()
