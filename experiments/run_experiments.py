"""
Convenience launcher. Run one mode at a time to control the time budget:
python experiments/run_experiments.py --mode hover --timesteps 100000
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "hover", "shaped"], required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    command = [
        sys.executable, str(ROOT / "training" / "train.py"),
        "--mode", args.mode, "--timesteps", str(args.timesteps)
    ]
    if args.tag:
        command += ["--tag", args.tag]
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
