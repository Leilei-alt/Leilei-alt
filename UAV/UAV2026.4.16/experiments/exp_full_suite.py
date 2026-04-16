from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def run_script(script_name: str):
    script = BASE / script_name
    print(f"\n=== Running {script.name} ===")
    result = subprocess.run([sys.executable, str(script)], check=True)
    return result.returncode


def main():
    run_script("exp_threshold_unlinkability.py")
    run_script("exp_batch_and_offload.py")
    run_script("exp_computation_cost_breakdown.py")
    print("\nAll ECC+MEC experiment suites finished successfully.")


if __name__ == "__main__":
    main()
