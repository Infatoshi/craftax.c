"""Single-axis sweeps over training config. Reports SPS per config."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = os.environ.get("TRAIN_PY",
    str(Path.home() / "PufferLib" / ".venv" / "bin" / "python"))

ENV = {
    **os.environ,
    "CRAFTAX_LIB": str(REPO / "libcraftax.so"),
    "PYTHONPATH": str(REPO / "python"),
    "OMP_NUM_THREADS": "16",
    "OMP_PROC_BIND": "close",
    "OMP_PLACES": "cores",
}


def run(**kwargs):
    cmd = [PY, str(REPO / "python" / "train_graph.py"),
           "--graph", "--graph-rollout"]
    for k, v in kwargs.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    out = subprocess.check_output(cmd, env=ENV, text=True, stderr=subprocess.STDOUT)
    for line in out.splitlines():
        m = re.search(r"SPS:\s*([\d,]+)", line)
        if m:
            return int(m.group(1).replace(",", ""))
    raise RuntimeError(f"couldn't parse SPS from:\n{out[-500:]}")


def header(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def sweep_axis(name, values, base, fmt=lambda v: str(v)):
    header(f"Sweep: {name}")
    print(f"  base: {base}")
    results = []
    for v in values:
        cfg = dict(base); cfg[name] = v
        try:
            sps = run(**cfg)
            results.append((v, sps))
            print(f"    {name}={fmt(v):>10}  SPS={sps:>12,}")
        except Exception as e:
            print(f"    {name}={fmt(v):>10}  FAILED  {e}")
            results.append((v, 0))
    return results


def main():
    # Base config -- our current best known.
    base = dict(
        num_envs=8192,
        horizon=64,
        iters=15,
        epochs=4,
        minibatch=4096,
        hidden=128,
        lr=3e-4,
    )

    print(f"Baseline run of base config:")
    base_sps = run(**base)
    print(f"  baseline SPS = {base_sps:,}")

    r_mb    = sweep_axis("minibatch", [2048, 4096, 8192, 16384, 32768, 65536], base)
    r_ep    = sweep_axis("epochs",    [1, 2, 4, 8],                             base)
    r_h     = sweep_axis("horizon",   [32, 64, 128, 256],                       base)
    r_ne    = sweep_axis("num_envs",  [2048, 4096, 8192, 16384],                base)
    r_hid   = sweep_axis("hidden",    [64, 128, 256, 512],                      base)

    # Combine winners from each axis.
    def best(r): return max(r, key=lambda kv: kv[1])[0]
    combined = dict(base,
        minibatch = best(r_mb),
        epochs    = best(r_ep),
        horizon   = best(r_h),
        num_envs  = best(r_ne),
        hidden    = best(r_hid),
    )
    header("Combined best-of-each-axis")
    print(f"  config: {combined}")
    try:
        combined_sps = run(**combined)
        print(f"  combined SPS = {combined_sps:,}  ({combined_sps/base_sps:.2f}x over base)")
    except Exception as e:
        print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
