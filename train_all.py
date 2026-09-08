"""Launch parallel CPU training runs for all four ankle-mode variants (or a chosen subset), each
as its own train.py subprocess. Checkpoints/logs land under runs/<mode>/<run-id>/ and
tb_logs/<mode>/<run-id>/ (train.py's own default -- this script doesn't override it, just passes
--ankle and lets each subprocess pick its own timestamped run-id), so `tensorboard --logdir
tb_logs` picks up all of them at once for side-by-side comparison.

    python train_all.py                              # all four, in parallel
    python train_all.py --single --dual               # just these two, in parallel
    python train_all.py --detailed --timesteps 500000  # just detailed, with an override

Any flag train.py itself accepts (--timesteps, --save-freq, --resume, ...) can be passed here too
and is forwarded unchanged to every subprocess launched -- don't use --resume across different
modes in one invocation though, since each mode needs its own checkpoint lineage.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# name -> ankle_mode value passed to train.py's --ankle (None omits the flag entirely)
MODES = {"baseline": None, "single": "single", "dual": "dual", "detailed": "detailed"}


def _stream(process: subprocess.Popen, label: str) -> None:
    for line in process.stdout:
        print(f"[{label}] {line}", end="")


def _launch(mode_name: str, ankle_mode: str | None, extra_args: list[str]) -> subprocess.Popen:
    cmd = [sys.executable, str(PROJECT_ROOT / "train.py")]
    if ankle_mode is not None:
        cmd += ["--ankle", ankle_mode]
    cmd += extra_args

    # Running 4 single-threaded envs at once on the same machine -- cap each subprocess's own
    # intra-op thread pool so they don't all try to claim every core and fight each other.
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all (or a chosen subset of) ankle-mode variants in parallel.")
    parser.add_argument("--baseline", action="store_true", help="no ankle (original rigid foot)")
    parser.add_argument("--single", action="store_true", help="single-hinge ankle")
    parser.add_argument("--dual", action="store_true", help="dual-hinge (pitch+roll) ankle")
    parser.add_argument("--detailed", action="store_true", help="dual-hinge ankle + split heel/toe foot")
    args, extra_args = parser.parse_known_args()

    selected = [name for name in ("baseline", "single", "dual", "detailed") if getattr(args, name)]
    if not selected:
        selected = list(MODES.keys())  # nothing specified -> run all four

    processes: dict[str, subprocess.Popen] = {}
    for mode_name in selected:
        ankle_mode = MODES[mode_name]
        print(f"Launching {mode_name} (ankle={ankle_mode!r})...")
        processes[mode_name] = _launch(mode_name, ankle_mode, extra_args)

    threads = [
        threading.Thread(target=_stream, args=(process, mode_name), daemon=True)
        for mode_name, process in processes.items()
    ]
    for t in threads:
        t.start()

    try:
        exit_codes = {mode_name: process.wait() for mode_name, process in processes.items()}
    except KeyboardInterrupt:
        print("\nInterrupted -- stopping all runs...")
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            process.wait()
        return 130

    for t in threads:
        t.join(timeout=2)

    print()
    failed = {name: code for name, code in exit_codes.items() if code != 0}
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print(f"ALL DONE: {list(exit_codes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
