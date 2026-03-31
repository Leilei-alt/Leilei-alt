# experiments/exp_computation_cost_breakdown.py
"""
Computation-cost breakdown experiment for Fusion-BLS.

Goal
----
Produce four runtime metrics that can directly map to Fig.10-style bars:
    (a) Vehicle/Device
    (b) RSU/MEC Server
    (c) TA/DCA
    (d) Total

Measurement principle
---------------------
This script uses implementation-level runtime measurement (wall-clock timing)
with coarse-grained per-entity accounting:

- vehicle:
    UAV-side operations only
- server:
    AAS + ECS operations only
- ta:
    DCA-side operations only
- total:
    vehicle + server + ta

Notes
-----
1. This is NOT the same as the paper's symbolic formula style
   (e.g. T_bp + T_mp + T_h). Instead, it is an implementation-runtime version.
2. By default, this script measures "Auth-only" cost.
   Edge task processing can be optionally included with INCLUDE_EDGE_PROCESSING=True.
3. Local UAV task processing is NOT included in Vehicle by default, because
   Fig.10 is closer to authentication / key-establishment cost than full task execution.

Adjust imports and method signatures if your local code differs slightly.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict





# -----------------------------------------------------------------------------
# Project path bootstrap
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
RESULTS_DIR = THIS_FILE.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Project imports
# Adjust these if your actual class/function names differ.
# -----------------------------------------------------------------------------
from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS
from entities.ecs import ECS

# Optional plotting: keep this local so the experiment still runs without it.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TRIALS = 30

# Use the BLS path only.
USE_BLS = True

PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481

# If True, include ECS.process_task(...) into "server" and "total".
# If False, only authentication + key establishment + offload verification are measured.
INCLUDE_EDGE_PROCESSING = False

# If True, also measure UAV local processing separately (not counted in main Fig.10 bars by default).
MEASURE_LOCAL_TASK_PROCESSING = False

# Default experiment settings.
TASK_TYPE = "VIDEO_ANALYSIS"
TASK_SIZE = 512

# DCA threshold configuration.
NUM_DCA_MEMBERS = 5
THRESHOLD = 3

# If your domain IDs or ECS IDs differ, edit here.
SOURCE_DOMAIN = "B"
TARGET_DOMAIN = "A"
AAS_ID = "AAS-A-01"
ECS_ID = "ECS-A-01"

# Output filenames
RAW_CSV = RESULTS_DIR / "comp_breakdown_raw.csv"
SUMMARY_CSV = RESULTS_DIR / "comp_breakdown_summary.csv"
FIG_PNG = RESULTS_DIR / "comp_breakdown_fig10_style.png"


# -----------------------------------------------------------------------------
# Runtime profiler
# -----------------------------------------------------------------------------
@dataclass
class RuntimeProfiler:
    """
    Coarse-grained profiler that accumulates runtime per (entity, operation).
    entity in {"vehicle", "server", "ta", "other"}
    """
    stats_ms: Dict[Tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))

    def measure_call(self, entity: str, op: str, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stats_ms[(entity, op)] += elapsed_ms
        return result, elapsed_ms

    def add_time_ms(self, entity: str, op: str, elapsed_ms: float) -> None:
        self.stats_ms[(entity, op)] += elapsed_ms

    def get_entity_total(self, entity: str) -> float:
        return sum(v for (e, _), v in self.stats_ms.items() if e == entity)

    def flatten(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for (entity, op), val in sorted(self.stats_ms.items()):
            out[f"{entity}::{op}"] = val
        return out


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def now_ts() -> int:
    """Return a simple integer timestamp."""
    return int(time.time())


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def safe_std(values: List[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_trials(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Summarize the main bar values across all trials.
    """
    vehicle_vals = [r["vehicle_ms"] for r in rows]
    server_vals = [r["server_ms"] for r in rows]
    ta_vals = [r["ta_ms"] for r in rows]
    total_vals = [r["total_ms"] for r in rows]

    summary = [
        {
            "metric": "vehicle_ms",
            "count": len(vehicle_vals),
            "mean": safe_mean(vehicle_vals),
            "std": safe_std(vehicle_vals),
            "min": min(vehicle_vals) if vehicle_vals else 0.0,
            "max": max(vehicle_vals) if vehicle_vals else 0.0,
        },
        {
            "metric": "server_ms",
            "count": len(server_vals),
            "mean": safe_mean(server_vals),
            "std": safe_std(server_vals),
            "min": min(server_vals) if server_vals else 0.0,
            "max": max(server_vals) if server_vals else 0.0,
        },
        {
            "metric": "ta_ms",
            "count": len(ta_vals),
            "mean": safe_mean(ta_vals),
            "std": safe_std(ta_vals),
            "min": min(ta_vals) if ta_vals else 0.0,
            "max": max(ta_vals) if ta_vals else 0.0,
        },
        {
            "metric": "total_ms",
            "count": len(total_vals),
            "mean": safe_mean(total_vals),
            "std": safe_std(total_vals),
            "min": min(total_vals) if total_vals else 0.0,
            "max": max(total_vals) if total_vals else 0.0,
        },
    ]
    return summary


# -----------------------------------------------------------------------------
# Environment / entity setup
# -----------------------------------------------------------------------------
def build_entities() -> Tuple[DCACluster, DCACluster, AAS, ECS, UAV]:
    dca_a = DCACluster(domain_id=TARGET_DOMAIN, n=NUM_DCA_MEMBERS, t=THRESHOLD, prime=PRIME)
    dca_b = DCACluster(domain_id=SOURCE_DOMAIN, n=NUM_DCA_MEMBERS, t=THRESHOLD, prime=PRIME)

    # AAS 属于目标域 A，但内部持有目标域自己的 dca_cluster，用于验证 token.target_domain == self.domain_id
    aas_a = AAS(domain_id=TARGET_DOMAIN, dca_cluster=dca_a)

    ecs_a = ECS(ecs_id=ECS_ID, domain_id=TARGET_DOMAIN, prime=PRIME)
    uav = UAV(uav_id="UAV-001", prime=PRIME)
    return dca_a, dca_b, aas_a, ecs_a, uav





# -----------------------------------------------------------------------------
# Main single-trial measurement logic
# -----------------------------------------------------------------------------
def run_single_trial(
    trial_idx: int,
    task_type: str = TASK_TYPE,
    task_size: int = TASK_SIZE,
    include_edge_processing: bool = INCLUDE_EDGE_PROCESSING,
    measure_local_task_processing: bool = MEASURE_LOCAL_TASK_PROCESSING,
) -> Dict[str, Any]:
    profiler = RuntimeProfiler()

    dca_a, dca_b, aas_a, ecs_a, uav = build_entities()

    # 1) TA/DCA: issue registration shares
    partial_shares, _ = profiler.measure_call(
        "ta",
        "issue_registration_shares",
        dca_b.issue_registration_shares,
        uav.seed,
        THRESHOLD,
    )

    # 2) Vehicle: reconstruct registration credential
    reg_ok, _ = profiler.measure_call(
        "vehicle",
        "request_registration",
        uav.request_registration,
        SOURCE_DOMAIN,
        partial_shares,
        THRESHOLD,
    )
    if not reg_ok:
        raise RuntimeError(f"Trial {trial_idx}: registration failed.")

    # 3) TA/DCA: generate cross-domain token
    token, _ = profiler.measure_call(
        "ta",
        "generate_cross_domain_token",
        dca_b.generate_cross_domain_token,
        TARGET_DOMAIN,
        THRESHOLD,
    )
    if token is None:
        raise RuntimeError(f"Trial {trial_idx}: cross-domain token generation failed.")

    # 4) Server: install token at AAS
    _, _ = profiler.measure_call(
        "server",
        "install_cross_domain_token",
        aas_a.install_cross_domain_token,
        SOURCE_DOMAIN,
        token,
    )

    # 5) Vehicle: build BLS access request
    timestamp_req = now_ts()
    access_req, _ = profiler.measure_call(
        "vehicle",
        "generate_bls_access_request",
        uav.generate_bls_access_request,
        SOURCE_DOMAIN,
        TARGET_DOMAIN,
        "EDGE_SERVICE",
        timestamp_req,
    )

    # 6) Server: verify BLS access request
    access_result, _ = profiler.measure_call(
        "server",
        "verify_bls_access_request",
        aas_a.verify_bls_access_request,
        access_req,
        dca_b,
        uav.bls_public_key,
    )
    if not access_result.success:
        raise RuntimeError(f"Trial {trial_idx}: BLS access verification failed: {access_result.reason}")

    # 7) Server: ECS issues service challenge
    timestamp_challenge = now_ts()
    challenge, _ = profiler.measure_call(
        "server",
        "issue_service_challenge",
        ecs_a.issue_service_challenge,
        timestamp_challenge,
    )

    # 8) Vehicle: UAV responds to challenge
    # signature: respond_service_challenge(spid, timestamp) -> (nonce_u, timestamp)
    (nonce_u, resp_ts), _ = profiler.measure_call(
        "vehicle",
        "respond_service_challenge",
        uav.respond_service_challenge,
        access_req.spid,
        challenge.timestamp,
    )

    # 9) Vehicle: derive service key
    _, _ = profiler.measure_call(
        "vehicle",
        "derive_service_key",
        uav.derive_service_key,
        access_req.spid,
        challenge.ecs_id,
        TARGET_DOMAIN,
        challenge.nonce_e,
        nonce_u,
        challenge.timestamp,
        "EDGE_SERVICE",
    )

    # 10) Server: derive service key
    _, _ = profiler.measure_call(
        "server",
        "derive_service_key",
        ecs_a.derive_service_key,
        access_req.spid,
        nonce_u,
        challenge,
        "EDGE_SERVICE",
    )

    # Optional consistency check
    if uav.current_service_key != ecs_a.current_service_key:
        raise RuntimeError(f"Trial {trial_idx}: service key mismatch.")

    # 11) Vehicle: build offload packet
    offload_packet, _ = profiler.measure_call(
        "vehicle",
        "build_offload_packet",
        uav.build_offload_packet,
        task_type,
        task_size,
    )

    # 12) Server: verify offload packet
    verify_packet_ok, _ = profiler.measure_call(
        "server",
        "verify_offload_packet",
        ecs_a.verify_offload_packet,
        offload_packet,
        ecs_a.current_service_key,
    )
    if not verify_packet_ok:
        raise RuntimeError(f"Trial {trial_idx}: offload packet verification failed.")

    # 13) Optional: edge task processing
    if include_edge_processing:
        _, _ = profiler.measure_call(
            "server",
            "process_task",
            ecs_a.process_task,
            task_type,
            task_size,
        )

    # 14) Optional: local UAV task processing
    local_task_ms = None
    if measure_local_task_processing:
        _, local_task_ms = profiler.measure_call(
            "vehicle",
            "local_task_processing",
            uav.process_task_locally,
            task_type,
            task_size,
        )

    vehicle_ms = profiler.get_entity_total("vehicle")
    server_ms = profiler.get_entity_total("server")
    ta_ms = profiler.get_entity_total("ta")
    total_ms = vehicle_ms + server_ms + ta_ms

    row: Dict[str, Any] = {
        "trial": trial_idx,
        "task_type": task_type,
        "task_size": task_size,
        "use_bls": True,
        "include_edge_processing": include_edge_processing,
        "measure_local_task_processing": measure_local_task_processing,
        "vehicle_ms": vehicle_ms,
        "server_ms": server_ms,
        "ta_ms": ta_ms,
        "total_ms": total_ms,
        "local_task_ms": local_task_ms if local_task_ms is not None else "",
    }
    row.update(profiler.flatten())
    return row


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_fig10_style(summary_rows: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Plot only your scheme's 4 bars in a Fig.10-style layout:
        (a) Vehicle/Device
        (b) RSU/MEC Server
        (c) TA/DCA
        (d) Total
    """
    if plt is None:
        print("[WARN] matplotlib not available; skip plotting.")
        return

    metric_to_summary = {row["metric"]: row for row in summary_rows}

    vehicle_mean = metric_to_summary["vehicle_ms"]["mean"]
    vehicle_std = metric_to_summary["vehicle_ms"]["std"]

    server_mean = metric_to_summary["server_ms"]["mean"]
    server_std = metric_to_summary["server_ms"]["std"]

    ta_mean = metric_to_summary["ta_ms"]["mean"]
    ta_std = metric_to_summary["ta_ms"]["std"]

    total_mean = metric_to_summary["total_ms"]["mean"]
    total_std = metric_to_summary["total_ms"]["std"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    labels = ["Fusion-BLS"]

    # (a) Vehicle
    axes[0].bar(labels, [vehicle_mean], yerr=[vehicle_std], capsize=5)
    axes[0].set_title("(a)")
    axes[0].set_ylabel("Computation Cost (ms)")
    axes[0].set_xlabel("Vehicle/Device")

    # (b) Server
    axes[1].bar(labels, [server_mean], yerr=[server_std], capsize=5)
    axes[1].set_title("(b)")
    axes[1].set_ylabel("Computation Cost (ms)")
    axes[1].set_xlabel("RSU/MEC Server")

    # (c) TA/DCA
    axes[2].bar(labels, [ta_mean], yerr=[ta_std], capsize=5)
    axes[2].set_title("(c)")
    axes[2].set_ylabel("Computation Cost (ms)")
    axes[2].set_xlabel("TA/DCA")

    # (d) Total
    axes[3].bar(labels, [total_mean], yerr=[total_std], capsize=5)
    axes[3].set_title("(d)")
    axes[3].set_ylabel("Computation Cost (ms)")
    axes[3].set_xlabel("Total Entities")

    fig.suptitle("Fusion-BLS Computation Cost Breakdown", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment driver
# -----------------------------------------------------------------------------
def run_experiment(
    trials: int = TRIALS,
    task_type: str = TASK_TYPE,
    task_size: int = TASK_SIZE,
    include_edge_processing: bool = INCLUDE_EDGE_PROCESSING,
    measure_local_task_processing: bool = MEASURE_LOCAL_TASK_PROCESSING,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []

    for trial_idx in range(1, trials + 1):
        row = run_single_trial(
            trial_idx=trial_idx,
            task_type=task_type,
            task_size=task_size,
            include_edge_processing=include_edge_processing,
            measure_local_task_processing=measure_local_task_processing,
        )
        raw_rows.append(row)

    summary_rows = summarize_trials(raw_rows)
    return raw_rows, summary_rows


def print_summary(summary_rows: List[Dict[str, Any]]) -> None:
    print("\n=== Fusion-BLS Computation Cost Breakdown Summary ===")
    for row in summary_rows:
        print(
            f"{row['metric']:>10s} | "
            f"mean={row['mean']:.4f} ms | "
            f"std={row['std']:.4f} ms | "
            f"min={row['min']:.4f} ms | "
            f"max={row['max']:.4f} ms"
        )


def main() -> None:
    raw_rows, summary_rows = run_experiment(
        trials=TRIALS,
        task_type=TASK_TYPE,
        task_size=TASK_SIZE,
        include_edge_processing=INCLUDE_EDGE_PROCESSING,
        measure_local_task_processing=MEASURE_LOCAL_TASK_PROCESSING,
    )

    write_csv(RAW_CSV, raw_rows)
    write_csv(SUMMARY_CSV, summary_rows)
    plot_fig10_style(summary_rows, FIG_PNG)
    print_summary(summary_rows)

    print("\nArtifacts:")
    print(f"  Raw CSV     : {RAW_CSV}")
    print(f"  Summary CSV : {SUMMARY_CSV}")
    print(f"  Figure PNG  : {FIG_PNG}")


if __name__ == "__main__":
    main()
