"""Small helpers shared by the entry points: run directories, devices, seeds, logging."""
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import artifacts
from .config import ROOT


def run_path(cfg, role, smoke=False):
    """runs/<exp>/<role>/ (smoke: scratch/smoke/<exp>/<role>/). Roles: model, s_train, s_eval."""
    path = artifacts.run_dir(cfg["exp"], smoke=smoke) / role
    path.mkdir(parents=True, exist_ok=True)
    return path


def seg_run(cfg, role, smoke=False):
    """Segmenters are trained once (trunk card) and shared; seg.runs_exp names where they live."""
    exp = cfg["seg"].get("runs_exp", cfg["exp"])
    path = artifacts.run_dir(exp, smoke=smoke) / role
    return path


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def machine():
    path = Path.home() / ".config" / "lab" / "machine"
    return path.read_text().strip() if path.exists() else "unknown"


class JsonlLog:
    def __init__(self, path):
        self.path = Path(path)
        self.start = time.time()

    def write(self, **row):
        row["time"] = round(time.time() - self.start, 1)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def to_device(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
