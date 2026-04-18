from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from entities.aas import AAS
from entities.dca import DCACluster
from entities.ecs import ECS
from entities.uav import UAV

PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RuntimeProfiler:
    stats_ms: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))

    def measure_call(self, entity: str, op: str, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.stats_ms[(entity, op)] += elapsed_ms
        return result, elapsed_ms

    def get_entity_total(self, entity: str) -> float:
        return sum(v for (e, _), v in self.stats_ms.items() if e == entity)

    def flatten(self) -> dict[str, float]:
        return {f"{e}::{op}": v for (e, op), v in sorted(self.stats_ms.items())}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_aas_and_token(dca_a: DCACluster, dca_b: DCACluster, threshold: int):
    aas_a = AAS(domain_id="A", dca_cluster=dca_a, replay_window=30)
    token = dca_b.generate_cross_domain_token(target_domain="A", available_count=threshold, validity_period=600)
    if token is None:
        raise RuntimeError("Token generation failed")
    aas_a.install_cross_domain_token(source_domain="B", token=token)
    return aas_a


def run_one_trial(task_type: str = "VIDEO_ANALYSIS", task_size: int = 64, n: int = 5, t: int = 3) -> dict[str, Any]:
    profiler = RuntimeProfiler()

    dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
    dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)
    aas_a, _ = profiler.measure_call("server", "prepare_aas_and_token", prepare_aas_and_token, dca_a, dca_b, t)
    ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=PRIME)
    uav = UAV(uav_id="UAV-COST-01", prime=PRIME)

    shares, _ = profiler.measure_call("ta", "issue_registration_shares", dca_b.issue_registration_shares, uav.seed, t)
    ok, _ = profiler.measure_call("vehicle", "request_registration", uav.request_registration, "B", shares, t)
    if not ok:
        raise RuntimeError("Registration failed")

    req, _ = profiler.measure_call("vehicle", "generate_access_request", uav.generate_access_request, "B", "A", "EDGE_SERVICE")
    result, _ = profiler.measure_call("server", "verify_access_request", aas_a.verify_access_request, req, dca_b, None, req.timestamp)
    if not result.success:
        raise RuntimeError(result.reason)

    challenge, _ = profiler.measure_call("server", "issue_service_challenge", ecs.issue_service_challenge, req.timestamp)
    _, _ = profiler.measure_call("server", "establish_internal_channel_with_ecs", aas_a.establish_internal_channel_with_ecs, ecs.ecs_id, challenge.nonce_e)
    _, _ = profiler.measure_call("server", "establish_internal_channel", ecs.establish_internal_channel, aas_a.aas_id, 1, challenge.nonce_e)
    (nonce_u, _), _ = profiler.measure_call("vehicle", "respond_service_challenge", uav.respond_service_challenge, req.spid, challenge.timestamp)
    sk_u, _ = profiler.measure_call(
        "vehicle", "derive_service_key", uav.derive_service_key,
        req.spid, ecs.ecs_id, "A", challenge.nonce_e, nonce_u, challenge.timestamp, task_type,
    )
    sk_e, _ = profiler.measure_call(
        "server", "derive_service_key", ecs.derive_service_key,
        req.spid, nonce_u, challenge, req.credential_public_key_b64, task_type,
    )
    if sk_u != sk_e:
        raise RuntimeError("Service keys mismatch")

    packet, _ = profiler.measure_call("vehicle", "build_offload_packet", uav.build_offload_packet, task_type, task_size)
    verified, _ = profiler.measure_call("server", "verify_offload_packet", ecs.verify_offload_packet, packet, sk_e)
    if not verified:
        raise RuntimeError("Offload verification failed")
    _, _ = profiler.measure_call("server", "process_task", ecs.process_task, task_type, task_size)

    vehicle_ms = profiler.get_entity_total("vehicle")
    server_ms = profiler.get_entity_total("server")
    ta_ms = profiler.get_entity_total("ta")
    total_ms = vehicle_ms + server_ms + ta_ms

    row = {
        "vehicle_ms": vehicle_ms,
        "server_ms": server_ms,
        "ta_ms": ta_ms,
        "total_ms": total_ms,
    }
    row.update(profiler.flatten())
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["vehicle_ms", "server_ms", "ta_ms", "total_ms"]
    out = []
    for m in metrics:
        vals = [r[m] for r in rows]
        out.append({
            "metric": m,
            "count": len(vals),
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) >= 2 else 0.0,
            "min": min(vals),
            "max": max(vals),
        })
    return out

def run_trials(
    num_trials: int = 30,
    warmup_runs: int = 3,
    task_type: str = "VIDEO_ANALYSIS",
    task_size: int = 64,
    n: int = 5,
    t: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []

    total_runs = warmup_runs + num_trials
    for trial in range(total_runs):
        row = run_one_trial(task_type=task_type, task_size=task_size, n=n, t=t)

        if trial < warmup_runs:
            continue

        row["trial"] = trial - warmup_runs
        row["task_type"] = task_type
        row["task_size"] = task_size
        raw_rows.append(row)

        print(
            f"[CompBreakdown] trial={trial - warmup_runs} "
            f"vehicle={row['vehicle_ms']:.3f}ms "
            f"server={row['server_ms']:.3f}ms "
            f"ta={row['ta_ms']:.3f}ms total={row['total_ms']:.3f}ms"
        )

    entity_summary = summarize(raw_rows)
    op_summary = summarize_operations(raw_rows)
    return raw_rows, entity_summary, op_summary

def summarize_operations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    op_keys = [
        k for k in rows[0].keys()
        if "::" in k
    ]

    out = []
    for k in sorted(op_keys):
        vals = [r[k] for r in rows]
        out.append({
            "operation": k,
            "count": len(vals),
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) >= 2 else 0.0,
            "min": min(vals),
            "max": max(vals),
        })
    return out

def plot_entity_summary(summary_rows: list[dict[str, Any]]) -> None:
    if plt is None or not summary_rows:
        return

    labels = [r["metric"] for r in summary_rows]
    vals = [r["mean"] for r in summary_rows]
    errs = [r["std"] for r in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals, yerr=errs, capsize=4)
    plt.ylabel("Time (ms)")
    plt.title("Computation-cost breakdown of the proposed scheme (mean ± std)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comp_breakdown_mean_std.png", dpi=200)
    plt.close()

def plot_operation_summary(op_summary_rows: list[dict[str, Any]]) -> None:
    if plt is None or not op_summary_rows:
        return

    labels = [r["operation"] for r in op_summary_rows]
    vals = [r["mean"] for r in op_summary_rows]
    errs = [r["std"] for r in op_summary_rows]

    plt.figure(figsize=(11, 5))
    plt.bar(labels, vals, yerr=errs, capsize=3)
    plt.ylabel("Time (ms)")
    plt.title("Operation-level computation cost (mean ± std)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comp_breakdown_operations_mean_std.png", dpi=200)
    plt.close()


def plot_summary(summary_rows: list[dict[str, Any]]) -> None:
    if plt is None or not summary_rows:
        return
    labels = [r["metric"] for r in summary_rows]
    vals = [r["mean"] for r in summary_rows]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.ylabel("Time (ms)")
    plt.title("Computation-cost breakdown under ECC anonymous auth + MEC")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comp_breakdown.png", dpi=200)
    plt.close()


def main():
    raw_rows, entity_summary, op_summary = run_trials(
        num_trials=30,
        warmup_runs=3,
        task_type="VIDEO_ANALYSIS",
        task_size=64,
        n=5,
        t=3,
    )

    write_csv(RESULTS_DIR / "comp_breakdown_raw.csv", raw_rows)
    write_csv(RESULTS_DIR / "comp_breakdown_entity_summary.csv", entity_summary)
    write_csv(RESULTS_DIR / "comp_breakdown_operation_summary.csv", op_summary)

    plot_entity_summary(entity_summary)
    plot_operation_summary(op_summary)

    print(f"[Saved] raw -> {RESULTS_DIR / 'comp_breakdown_raw.csv'}")
    print(f"[Saved] entity summary -> {RESULTS_DIR / 'comp_breakdown_entity_summary.csv'}")
    print(f"[Saved] operation summary -> {RESULTS_DIR / 'comp_breakdown_operation_summary.csv'}")

if __name__ == "__main__":
    main()
