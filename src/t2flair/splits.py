"""Subject-level splits (docs/DATA.md §划分).

main: seed 0, 430 of the 1251 subjects -> train 300 / val 30 / test 100.
lit:  as in the public D3Bridge script: sort IDs, python random.seed(42), shuffle,
      take 25 / 5 / 10 (slices [27, 127) are selected at preprocessing time).

    python -m t2flair.splits --subjects <file with one ID per line> --out configs/splits
"""
import argparse
import random
import re
from pathlib import Path

import numpy as np

ID = re.compile(r"BraTS2021_\d{5}")
MAIN = {"train": 300, "val": 30, "test": 100}
LIT = {"train": 25, "val": 5, "test": 10}


def read_ids(path):
    ids = sorted({line.strip() for line in Path(path).read_text().splitlines() if ID.fullmatch(line.strip())})
    if not ids:
        raise ValueError("no BraTS2021_<5 digits> IDs in %s" % path)
    return ids


def main_split(ids, seed=0, sizes=MAIN):
    order = np.random.default_rng(seed).permutation(len(ids))
    picked = [ids[i] for i in order[: sum(sizes.values())]]
    out, start = {}, 0
    for name, size in sizes.items():
        out[name] = sorted(picked[start:start + size])
        start += size
    return out


def lit_split(ids, seed=42, sizes=LIT):
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    out, start = {}, 0
    for name, size in sizes.items():
        out[name] = sorted(shuffled[start:start + size])
        start += size
    return out


def load_split(prefix, name):
    """prefix like 'configs/splits/main' -> IDs from configs/splits/main_<name>.txt."""
    from .config import _resolve
    return read_ids(_resolve(str(prefix) + "_" + name + ".txt"))


def write(out_dir, protocol, split):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in split.items():
        (out_dir / "{}_{}.txt".format(protocol, name)).write_text("\n".join(ids) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", required=True)
    parser.add_argument("--out", default="configs/splits")
    args = parser.parse_args()
    ids = read_ids(args.subjects)
    print("subjects:", len(ids))
    for protocol, split in (("main", main_split(ids)), ("lit", lit_split(ids))):
        names = list(split)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not set(split[a]) & set(split[b]), (protocol, a, b)
        write(args.out, protocol, split)
        print(protocol, {k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()
