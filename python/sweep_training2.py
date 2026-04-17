"""Phase 2: pairwise / combined configs that balance SPS with training quality.

The phase-1 sweep flagged three axes that move SPS noticeably:
  - epochs    (K=1 is ~2x K=4 but changes PPO sample efficiency)
  - minibatch (bigger => fewer launches; compute-bound regime at 16k+)
  - hidden    (64 vs 128 is ~3% SPS but halves model capacity)

This sweep holds "total gradient steps per iter" roughly constant or picks
configurations commonly used in published PPO work, so we find the config
that maximizes SPS without mangling training.
"""
import os, re, subprocess, sys
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
        if m: return int(m.group(1).replace(",", ""))
    raise RuntimeError("no SPS")


def desc(c):
    # total minibatch updates per PPO iter
    total = c["num_envs"] * c["horizon"]
    mbs_per_epoch = total // c["minibatch"]
    total_updates = mbs_per_epoch * c["epochs"]
    return f"{total_updates:>3} updates/iter ({mbs_per_epoch} mb x {c['epochs']} ep)"


def main():
    base = dict(num_envs=8192, horizon=64, iters=15,
                epochs=4, minibatch=4096, hidden=128, lr=3e-4)
    print("Configurations:")
    print(f"  {'config':70s} {'SPS':>10s}  {'description':>28s}")
    print("-" * 112)

    configs = [
        # Baseline
        ("baseline",                                  dict(base)),
        # Vary epochs only
        ("K=2",                                       dict(base, epochs=2)),
        ("K=1",                                       dict(base, epochs=1)),
        # MB sweep at K=4 (keep training dynamics)
        ("K=4 MB=8k",                                 dict(base, minibatch=8192)),
        ("K=4 MB=16k",                                dict(base, minibatch=16384)),
        ("K=4 MB=32k",                                dict(base, minibatch=32768)),
        # Balanced: K=2 + bigger MB
        ("K=2 MB=8k",                                 dict(base, epochs=2, minibatch=8192)),
        ("K=2 MB=16k",                                dict(base, epochs=2, minibatch=16384)),
        ("K=2 MB=32k",                                dict(base, epochs=2, minibatch=32768)),
        # "full-batch PPO" (a single update per epoch)
        ("K=4 MB=full (524k)",                        dict(base, epochs=4, minibatch=524288)),
        ("K=8 MB=full",                               dict(base, epochs=8, minibatch=524288)),
        # Bigger NE + same total updates
        ("NE=16k K=2 MB=16k",                         dict(base, num_envs=16384, epochs=2, minibatch=16384)),
        ("NE=16k K=4 MB=32k",                         dict(base, num_envs=16384, epochs=4, minibatch=32768)),
        # Policy size tradeoffs
        ("H=64 K=4 MB=4k",                            dict(base, hidden=64)),
        ("H=256 K=4 MB=16k",                          dict(base, hidden=256, minibatch=16384)),
        # Max SPS (quality-compromised)
        ("K=1 MB=16k",                                dict(base, epochs=1, minibatch=16384)),
        ("K=1 MB=full",                               dict(base, epochs=1, minibatch=524288)),
        # Final candidate for quality-preserving best
        ("K=4 MB=16k H=128",                          dict(base, minibatch=16384)),
    ]

    best_quality_preserving = None  # K>=3, H>=128
    best_any = None
    for name, cfg in configs:
        try:
            sps = run(**cfg)
        except Exception as e:
            sps = 0
            note = f" FAILED: {str(e)[:40]}"
        else:
            note = ""
        print(f"  {name:70s} {sps:>10,}  {desc(cfg):>28s}{note}")
        if sps > 0:
            if (best_any is None) or sps > best_any[1]: best_any = (name, sps, cfg)
            if cfg["epochs"] >= 3 and cfg["hidden"] >= 128:
                if (best_quality_preserving is None) or sps > best_quality_preserving[1]:
                    best_quality_preserving = (name, sps, cfg)

    print()
    if best_any:
        print(f"  BEST OVERALL           : {best_any[0]}  {best_any[1]:,} SPS")
    if best_quality_preserving:
        print(f"  BEST (K>=3, H>=128)    : {best_quality_preserving[0]}  {best_quality_preserving[1]:,} SPS")


if __name__ == "__main__":
    main()
