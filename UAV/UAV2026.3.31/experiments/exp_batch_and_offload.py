# experiments/exp_batch_and_offload.py
from __future__ import annotations

import time
from pathlib import Path

from analysis.stats_utils import write_dict_rows_to_csv, group_and_summarize
from analysis.plot_utils import plot_line_with_errorbars, plot_multi_line
from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS
from entities.ecs import ECS


PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481
RAW_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")
FIG_DIR = Path("results/figures")


def prepare_uav_and_aas(dca_a: DCACluster, dca_b: DCACluster, threshold: int):
    aas_a = AAS(domain_id="A", dca_cluster=dca_a, replay_window=10)
    token = dca_b.generate_cross_domain_token(target_domain="A", available_count=threshold, validity_period=300)
    if token is None:
        raise RuntimeError("Failed to generate cross-domain token.")
    aas_a.install_cross_domain_token(source_domain="B", token=token)
    return aas_a


def register_uav_in_b(dca_b: DCACluster, uav: UAV, threshold: int):
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=threshold)
    ok = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)
    if not ok:
        raise RuntimeError(f"Failed to register {uav.uav_id} in domain B.")


def run_batch_auth_trials(num_trials: int, batch_sizes: list[int], n: int = 5, t: int = 3) -> tuple[list[dict], list[dict]]:
    raw_rows = []

    for trial in range(num_trials):
        dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
        dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)

        for batch_size in batch_sizes:
            aas_a = prepare_uav_and_aas(dca_a, dca_b, t)

            uavs = []
            requests = []
            for i in range(batch_size):
                uav = UAV(uav_id=f"UAV-BATCH-{trial}-{batch_size}-{i}", prime=PRIME)
                register_uav_in_b(dca_b, uav, t)
                req = uav.generate_access_request(
                    source_domain="B",
                    target_domain="A",
                    req_type="EDGE_SERVICE",
                )
                uavs.append(uav)
                requests.append(req)

            start = time.perf_counter()
            success_count = 0
            for uav, req in zip(uavs, requests):
                result = aas_a.verify_access_request(
                    request=req,
                    source_dca=dca_b,
                    uav_public_key=uav.public_key,
                )
                if result.success:
                    success_count += 1
            end = time.perf_counter()

            total_time = end - start
            avg_time = total_time / batch_size
            throughput = success_count / total_time if total_time > 0 else 0.0

            row = {
                "trial": trial,
                "batch_size": batch_size,
                "success_count": success_count,
                "total_time": total_time,
                "avg_time": avg_time,
                "throughput": throughput,
            }
            raw_rows.append(row)

            print(
                f"[BatchAuth][trial={trial}] batch_size={batch_size}, "
                f"success={success_count}/{batch_size}, total_time={total_time:.6f}s, "
                f"avg_time={avg_time:.6f}s, throughput={throughput:.2f} req/s"
            )

    summary_rows = group_and_summarize(
        raw_rows,
        group_key="batch_size",
        value_keys=["total_time", "avg_time", "throughput"],
    )
    return raw_rows, summary_rows


def run_task_scaling_trials(
    num_trials: int,
    task_type: str,
    task_sizes: list[int],
    n: int = 5,
    t: int = 3,
) -> tuple[list[dict], list[dict]]:
    raw_rows = []

    for trial in range(num_trials):
        dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
        dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)

        for task_size in task_sizes:
            # Local
            local_uav = UAV(uav_id=f"UAV-LOCAL-{trial}-{task_size}", prime=PRIME)
            local_result = local_uav.process_task_locally(task_type=task_type, task_size=task_size)
            local_time = local_result["measured_processing_time"]

            # Edge offloading
            aas_a = prepare_uav_and_aas(dca_a, dca_b, t)
            ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=PRIME)
            uav = UAV(uav_id=f"UAV-OFFLOAD-{trial}-{task_size}", prime=PRIME)
            register_uav_in_b(dca_b, uav, t)

            t0 = time.perf_counter()
            req = uav.generate_access_request(source_domain="B", target_domain="A", req_type="EDGE_SERVICE")
            access_result = aas_a.verify_access_request(
                request=req,
                source_dca=dca_b,
                uav_public_key=uav.public_key,
            )
            t1 = time.perf_counter()

            if not access_result.success:
                raise RuntimeError(f"Access failed for task_size={task_size}: {access_result.reason}")

            challenge = ecs.issue_service_challenge(timestamp=int(time.time()))
            nonce_u, _ = uav.respond_service_challenge(spid=req.spid, timestamp=challenge.timestamp)

            sk_uav = uav.derive_service_key(
                spid=req.spid,
                ecs_id=ecs.ecs_id,
                domain_id="A",
                nonce_e=challenge.nonce_e,
                nonce_u=nonce_u,
                challenge_timestamp=challenge.timestamp,
                context="EDGE_SERVICE",
            )
            sk_ecs = ecs.derive_service_key(
                uav_spid=req.spid,
                nonce_u=nonce_u,
                challenge=challenge,
                context="EDGE_SERVICE",
            )
            t2 = time.perf_counter()

            if sk_uav != sk_ecs:
                raise RuntimeError(f"Service key mismatch for task_size={task_size}")

            packet = uav.build_offload_packet(task_type=task_type, task_size=task_size)
            verified = ecs.verify_offload_packet(packet=packet, service_key=sk_ecs)
            if not verified:
                raise RuntimeError(f"Packet verification failed for task_size={task_size}")

            ecs.process_task(task_type=task_type, task_size=task_size)
            t3 = time.perf_counter()

            access_t = t1 - t0
            key_t = t2 - t1
            task_t = t3 - t2
            edge_total_t = t3 - t0

            row = {
                "trial": trial,
                "task_size": task_size,
                "local_time": local_time,
                "edge_total_time": edge_total_t,
                "access_time": access_t,
                "key_time": key_t,
                "task_time": task_t,
            }
            raw_rows.append(row)

            print(
                f"[TaskScaling][trial={trial}] size={task_size}, local={local_time:.6f}s, "
                f"edge_total={edge_total_t:.3f}s, access={access_t:.6f}s, "
                f"key={key_t:.6f}s, task={task_t:.6f}s"
            )

    summary_rows = group_and_summarize(
        raw_rows,
        group_key="task_size",
        value_keys=["local_time", "edge_total_time", "access_time", "key_time", "task_time"],
    )
    return raw_rows, summary_rows


def save_and_plot_batch(raw_rows: list[dict], summary_rows: list[dict]):
    write_dict_rows_to_csv(RAW_DIR / "batch_auth_raw.csv", raw_rows)
    write_dict_rows_to_csv(SUMMARY_DIR / "batch_auth_summary.csv", summary_rows)

    x = [int(r["batch_size"]) for r in summary_rows]

    plot_line_with_errorbars(
        x_vals=x,
        y_means=[r["total_time_mean"] for r in summary_rows],
        y_stds=[r["total_time_std"] for r in summary_rows],
        xlabel="Number of UAVs",
        ylabel="Total authentication time (s)",
        title="UAV count vs total authentication time",
        out_path=FIG_DIR / "batch_total_time_mean_std.png",
    )

    plot_line_with_errorbars(
        x_vals=x,
        y_means=[r["avg_time_mean"] for r in summary_rows],
        y_stds=[r["avg_time_std"] for r in summary_rows],
        xlabel="Number of UAVs",
        ylabel="Average authentication time per UAV (s)",
        title="UAV count vs average authentication time",
        out_path=FIG_DIR / "batch_avg_time_mean_std.png",
    )

    plot_line_with_errorbars(
        x_vals=x,
        y_means=[r["throughput_mean"] for r in summary_rows],
        y_stds=[r["throughput_std"] for r in summary_rows],
        xlabel="Number of UAVs",
        ylabel="Throughput (requests/s)",
        title="UAV count vs authentication throughput",
        out_path=FIG_DIR / "batch_throughput_mean_std.png",
    )


def save_and_plot_task_scaling(raw_rows: list[dict], summary_rows: list[dict]):
    write_dict_rows_to_csv(RAW_DIR / "task_scaling_raw.csv", raw_rows)
    write_dict_rows_to_csv(SUMMARY_DIR / "task_scaling_summary.csv", summary_rows)

    x = [int(r["task_size"]) for r in summary_rows]

    plot_multi_line(
        x_vals=x,
        series=[
            ("Local execution", [r["local_time_mean"] for r in summary_rows]),
            ("Edge offloading total", [r["edge_total_time_mean"] for r in summary_rows]),
        ],
        xlabel="Task size",
        ylabel="Total time (s)",
        title="Task size vs local execution / edge offloading",
        out_path=FIG_DIR / "task_local_vs_edge_mean.png",
    )

    plot_multi_line(
        x_vals=x,
        series=[
            ("Access", [r["access_time_mean"] for r in summary_rows]),
            ("Key agreement", [r["key_time_mean"] for r in summary_rows]),
            ("Edge task processing", [r["task_time_mean"] for r in summary_rows]),
        ],
        xlabel="Task size",
        ylabel="Time (s)",
        title="Task size vs offloading time breakdown",
        out_path=FIG_DIR / "task_offloading_breakdown_mean.png",
    )


def main():
    num_trials = 20

    print("=== Running repeated batch authentication experiment ===")
    batch_raw, batch_summary = run_batch_auth_trials(
        num_trials=num_trials,
        batch_sizes=[1, 5, 10, 20, 50, 100],
    )
    save_and_plot_batch(batch_raw, batch_summary)

    print("\n=== Running repeated task scaling experiment ===")
    task_raw, task_summary = run_task_scaling_trials(
        num_trials=num_trials,
        task_type="VIDEO_ANALYSIS",
        task_sizes=[2, 4, 8, 16, 32],
    )
    save_and_plot_task_scaling(task_raw, task_summary)

    print("\nDone. Raw CSV saved under ./results/raw/")
    print("Summary CSV saved under ./results/summary/")
    print("Figures saved under ./results/figures/")


if __name__ == "__main__":
    main()
