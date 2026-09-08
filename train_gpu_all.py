"""Train all four ankle-mode variants on GPU, one after another (NOT in parallel like
train_all.py's CPU runs -- all four would otherwise compete for the same single GPU/8GB VRAM).
Checkpoints/logs land under runs_gpu/<mode>/<run-id>/ and tb_logs_gpu/<mode>/<run-id>/
(train_gpu.py's own default -- this script doesn't override it, just passes --ankle and lets each
run pick its own timestamped run-id), so `tensorboard --logdir tb_logs_gpu` picks up all of them
for comparison once each has run.

    python train_gpu_all.py                              # all four, sequentially
    python train_gpu_all.py --single --dual               # just a subset
    python train_gpu_all.py --timesteps 20000000           # override any train_gpu.py flag

Any flag train_gpu.py accepts is forwarded unchanged to each run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODES = {"baseline": None, "single": "single", "dual": "dual", "detailed": "detailed"}


def _run(mode_name: str, ankle_mode: str | None, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "train_gpu.py")]
    if ankle_mode is not None:
        cmd += ["--ankle", ankle_mode]
    cmd += extra_args
    print(f"\n=== {mode_name} (ankle={ankle_mode!r}) ===")
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all (or a chosen subset of) ankle-mode variants on GPU, sequentially.")
    parser.add_argument("--baseline", action="store_true", help="no ankle (original rigid foot)")
    parser.add_argument("--single", action="store_true", help="single-hinge ankle")
    parser.add_argument("--dual", action="store_true", help="dual-hinge (pitch+roll) ankle")
    parser.add_argument("--detailed", action="store_true", help="dual-hinge ankle + split heel/toe foot")
    args, extra_args = parser.parse_known_args()

    selected = [name for name in ("baseline", "single", "dual", "detailed") if getattr(args, name)]
    if not selected:
        selected = list(MODES.keys())  # nothing specified -> all four

    results = {}
    for mode_name in selected:
        results[mode_name] = _run(mode_name, MODES[mode_name], extra_args)

    print()
    failed = {name: code for name, code in results.items() if code != 0}
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print(f"ALL DONE: {list(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
