"""Load configs/trunk.yaml, overlay an experiment file and command-line overrides.

Experiments only list differences from the trunk (AGENTS.md §4). `${VAR}` in strings is
expanded from the environment when the config is loaded.
"""
import copy
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _merge(base, extra):
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand(node):
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    if isinstance(node, str):
        def sub(match):
            name = match.group(1)
            if name not in os.environ:
                raise KeyError("config uses ${%s} but it is not set; source tools/env.sh and ~/.bashrc" % name)
            return os.environ[name]
        return _VAR.sub(sub, node)
    return node


def _set(cfg, dotted, raw):
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 reads "2e-4" as a string; accept plain numbers on the command line
        try:
            value = float(value)
        except ValueError:
            pass
    node[keys[-1]] = value


def load(path="configs/trunk.yaml", exp=None, overrides=()):
    """Return the merged, expanded config. `overrides` are 'a.b=value' strings (YAML values)."""
    with open(_resolve(path)) as handle:
        cfg = yaml.safe_load(handle)
    if exp:
        with open(_resolve(exp)) as handle:
            cfg = _merge(cfg, yaml.safe_load(handle))
    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError("override must look like key.sub=value: %r" % item)
        _set(cfg, key, value)
    return _expand(cfg)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def add_args(parser):
    parser.add_argument("--config", default="configs/trunk.yaml")
    parser.add_argument("--exp", default=None, help="configs/exp/<id>.yaml, differences only")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override, e.g. train.steps=20")


def from_args(args):
    return load(args.config, args.exp, args.set)
