# experiments/exp_full_suite.py
from __future__ import annotations

import subprocess
import sys


def run_script(script: str):
    print(f"\n=== Running {script} ===")
    result = subprocess.run([sys.executable, script], check=True)
    return result.returncode


def main():
    run_script("experiments/exp_batch_and_offload.py")
    run_script("experiments/exp_threshold_unlinkability.py")
    print("\nAll experiment suites finished successfully.")


if __name__ == "__main__":
    main()
