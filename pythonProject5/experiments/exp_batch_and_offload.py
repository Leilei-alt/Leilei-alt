# experiments/exp_batch_and_offload.py
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt

from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS
from entities.ecs import ECS


PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481


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


def experiment_batch_authentication(
    dca_a: DCACluster,
    dca_b: DCACluster,
    threshold: int,
    batch_sizes: list[int],
):
    """
    Compare sequential verification vs 'batch-style' grouped verification.
    Current prototype does not use true BLS aggregation yet.
    This experiment mainly builds the timing framework.
    """
    sequential_total_times = []
    sequential_avg_times = []
    throughput_values = []

    for batch_size in batch_sizes:
        aas_a = prepare_uav_and_aas(dca_a, dca_b, threshold)

        uavs = []
        requests = []
        for i in range(batch_size):
            uav = UAV(uav_id=f"UAV-BATCH-{batch_size}-{i}", prime=PRIME)
            register_uav_in_b(dca_b, uav, threshold)
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
        throughput = success_count / total_time if total_time > 0 else 0

        sequential_total_times.append(total_time)
        sequential_avg_times.append(avg_time)
        throughput_values.append(throughput)

        print(
            f"[BatchAuth] batch_size={batch_size}, success={success_count}/{batch_size}, "
            f"total_time={total_time:.6f}s, avg_time={avg_time:.6f}s, throughput={throughput:.2f} req/s"
        )

    return {
        "batch_sizes": batch_sizes,
        "sequential_total_times": sequential_total_times,
        "sequential_avg_times": sequential_avg_times,
        "throughput_values": throughput_values,
    }


def experiment_task_size_scaling(
    dca_a: DCACluster,
    dca_b: DCACluster,
    threshold: int,
    task_type: str,
    task_sizes: list[int],
):
    local_times = []
    edge_total_times = []
    access_times = []
    key_times = []
    task_times = []

    for task_size in task_sizes:
        # Local baseline
        local_uav = UAV(uav_id=f"UAV-LOCAL-{task_size}", prime=PRIME)
        local_result = local_uav.process_task_locally(task_type=task_type, task_size=task_size)
        local_time = local_result["measured_processing_time"]

        # Offloading path
        aas_a = prepare_uav_and_aas(dca_a, dca_b, threshold)
        ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=PRIME)
        uav = UAV(uav_id=f"UAV-OFFLOAD-{task_size}", prime=PRIME)
        register_uav_in_b(dca_b, uav, threshold)

        # Access
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

        # Key agreement
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

        # Offload task
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

        local_times.append(local_time)
        edge_total_times.append(edge_total_t)
        access_times.append(access_t)
        key_times.append(key_t)
        task_times.append(task_t)

        print(
            f"[TaskScaling] size={task_size}, local={local_time:.6f}s, edge_total={edge_total_t:.6f}s, "
            f"access={access_t:.6f}s, key={key_t:.6f}s, task={task_t:.6f}s"
        )

    return {
        "task_sizes": task_sizes,
        "local_times": local_times,
        "edge_total_times": edge_total_times,
        "access_times": access_times,
        "key_times": key_times,
        "task_times": task_times,
    }


def plot_batch_auth_results(results: dict, out_dir: Path):
    batch_sizes = results["batch_sizes"]
    total_times = results["sequential_total_times"]
    avg_times = results["sequential_avg_times"]
    throughput = results["throughput_values"]

    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, total_times, marker="o")
    plt.xlabel("Number of UAVs")
    plt.ylabel("Total authentication time (s)")
    plt.title("UAV count vs total authentication time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "batch_total_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, avg_times, marker="o")
    plt.xlabel("Number of UAVs")
    plt.ylabel("Average authentication time per UAV (s)")
    plt.title("UAV count vs average authentication time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "batch_avg_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, throughput, marker="o")
    plt.xlabel("Number of UAVs")
    plt.ylabel("Throughput (requests/s)")
    plt.title("UAV count vs authentication throughput")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "batch_throughput.png", dpi=200)
    plt.close()


def plot_task_scaling_results(results: dict, out_dir: Path):
    task_sizes = results["task_sizes"]
    local_times = results["local_times"]
    edge_total_times = results["edge_total_times"]
    access_times = results["access_times"]
    key_times = results["key_times"]
    task_times = results["task_times"]

    plt.figure(figsize=(8, 5))
    plt.plot(task_sizes, local_times, marker="o", label="Local execution")
    plt.plot(task_sizes, edge_total_times, marker="o", label="Edge offloading total")
    plt.xlabel("Task size")
    plt.ylabel("Total time (s)")
    plt.title("Task size vs local execution / edge offloading")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "task_local_vs_edge.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(task_sizes, access_times, marker="o", label="Access")
    plt.plot(task_sizes, key_times, marker="o", label="Key agreement")
    plt.plot(task_sizes, task_times, marker="o", label="Edge task processing")
    plt.xlabel("Task size")
    plt.ylabel("Time (s)")
    plt.title("Task size vs offloading time breakdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "task_offloading_breakdown.png", dpi=200)
    plt.close()


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 5
    t = 3
    dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
    dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)

    print("=== Running batch authentication experiment ===")
    batch_results = experiment_batch_authentication(
        dca_a=dca_a,
        dca_b=dca_b,
        threshold=t,
        batch_sizes=[1, 5, 10, 20, 50, 100],
    )
    plot_batch_auth_results(batch_results, out_dir)

    print("\n=== Running task scaling experiment ===")
    task_results = experiment_task_size_scaling(
        dca_a=dca_a,
        dca_b=dca_b,
        threshold=t,
        task_type="VIDEO_ANALYSIS",
        task_sizes=[2, 4, 8, 16, 32],
    )
    plot_task_scaling_results(task_results, out_dir)

    print("\nExperiment finished. Figures saved under ./results/")


if __name__ == "__main__":
    main()
