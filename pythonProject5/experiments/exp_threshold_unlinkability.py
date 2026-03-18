# experiments/exp_threshold_unlinkability.py
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt

from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS
from entities.ecs import ECS


PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481


def prepare_token_and_aas(dca_a: DCACluster, dca_b: DCACluster, threshold: int) -> AAS:
    aas_a = AAS(domain_id="A", dca_cluster=dca_a, replay_window=10)
    token = dca_b.generate_cross_domain_token(
        target_domain="A",
        available_count=threshold,
        validity_period=300,
    )
    if token is None:
        raise RuntimeError("Failed to generate cross-domain token.")
    aas_a.install_cross_domain_token(source_domain="B", token=token)
    return aas_a


def register_uav(dca_b: DCACluster, uav: UAV, threshold: int) -> None:
    shares = dca_b.issue_registration_shares(
        uav_seed=uav.seed,
        available_count=threshold,
    )
    ok = uav.request_registration(
        domain_id="B",
        partial_shares=shares,
        threshold=threshold,
    )
    if not ok:
        raise RuntimeError("UAV registration failed.")


# ============================================================
# Experiment 1: (n,t) parameter scaling
# ============================================================

def experiment_threshold_scaling(configs: list[tuple[int, int]]):
    registration_times = []
    token_times = []
    labels = []

    for n, t in configs:
        dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
        dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)
        uav = UAV(uav_id=f"UAV-TH-{n}-{t}", prime=PRIME)

        # Registration timing
        start_reg = time.perf_counter()
        shares = dca_b.issue_registration_shares(
            uav_seed=uav.seed,
            available_count=t,
        )
        ok = uav.request_registration(
            domain_id="B",
            partial_shares=shares,
            threshold=t,
        )
        end_reg = time.perf_counter()
        if not ok:
            raise RuntimeError(f"Registration failed for config {(n, t)}")

        # Cross-domain token generation timing
        start_tok = time.perf_counter()
        token = dca_b.generate_cross_domain_token(
            target_domain="A",
            available_count=t,
            validity_period=300,
        )
        end_tok = time.perf_counter()
        if token is None:
            raise RuntimeError(f"Token generation failed for config {(n, t)}")

        reg_t = end_reg - start_reg
        tok_t = end_tok - start_tok

        registration_times.append(reg_t)
        token_times.append(tok_t)
        labels.append(f"({n},{t})")

        print(
            f"[ThresholdScaling] config=({n},{t}), "
            f"registration={reg_t:.6f}s, token={tok_t:.6f}s"
        )

    return {
        "labels": labels,
        "registration_times": registration_times,
        "token_times": token_times,
    }


def plot_threshold_scaling(results: dict, out_dir: Path):
    labels = results["labels"]
    reg_times = results["registration_times"]
    tok_times = results["token_times"]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, reg_times)
    plt.xlabel("(n,t) configuration")
    plt.ylabel("Registration time (s)")
    plt.title("Threshold configuration vs UAV registration time")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_registration_time.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(labels, tok_times)
    plt.xlabel("(n,t) configuration")
    plt.ylabel("Cross-domain token generation time (s)")
    plt.title("Threshold configuration vs token generation time")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_token_time.png", dpi=200)
    plt.close()


# ============================================================
# Experiment 2: unlinkability / anonymity
# ============================================================

def experiment_unlinkability(num_sessions: int = 100):
    dca_a = DCACluster(domain_id="A", n=5, t=3, prime=PRIME)
    dca_b = DCACluster(domain_id="B", n=5, t=3, prime=PRIME)

    aas_a = prepare_token_and_aas(dca_a, dca_b, threshold=3)
    ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=PRIME)

    uav = UAV(uav_id="UAV-UNLINK-01", prime=PRIME)
    register_uav(dca_b, uav, threshold=3)

    spids = []
    service_keys = []

    for i in range(num_sessions):
        req = uav.generate_access_request(
            source_domain="B",
            target_domain="A",
            req_type="EDGE_SERVICE",
            timestamp=int(time.time()) + i,  # avoid replay rejection
        )

        result = aas_a.verify_access_request(
            request=req,
            source_dca=dca_b,
            uav_public_key=uav.public_key,
            now_ts=req.timestamp,
        )
        if not result.success:
            raise RuntimeError(f"Access failed at session {i}: {result.reason}")

        challenge = ecs.issue_service_challenge(timestamp=req.timestamp)
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

        if sk_uav != sk_ecs:
            raise RuntimeError(f"Service key mismatch at session {i}")

        spids.append(req.spid)
        service_keys.append(sk_uav)

    unique_spids = len(set(spids))
    unique_keys = len(set(service_keys))

    spid_repeat_rate = 1 - unique_spids / len(spids)
    key_repeat_rate = 1 - unique_keys / len(service_keys)

    print(
        f"[Unlinkability] sessions={num_sessions}, "
        f"unique_spids={unique_spids}, spid_repeat_rate={spid_repeat_rate:.6f}, "
        f"unique_keys={unique_keys}, key_repeat_rate={key_repeat_rate:.6f}"
    )

    return {
        "num_sessions": num_sessions,
        "unique_spids": unique_spids,
        "unique_keys": unique_keys,
        "spid_repeat_rate": spid_repeat_rate,
        "key_repeat_rate": key_repeat_rate,
    }


def plot_unlinkability(results: dict, out_dir: Path):
    metrics = ["SPID repeat rate", "Service-key repeat rate"]
    values = [results["spid_repeat_rate"], results["key_repeat_rate"]]

    plt.figure(figsize=(7, 5))
    plt.bar(metrics, values)
    plt.ylabel("Repeat rate")
    plt.title("Unlinkability evaluation")
    plt.tight_layout()
    plt.savefig(out_dir / "unlinkability_spid_repeat.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(["Unique SPIDs", "Unique keys"], [results["unique_spids"], results["unique_keys"]])
    plt.ylabel("Count")
    plt.title("Unique session identifiers and keys")
    plt.tight_layout()
    plt.savefig(out_dir / "unlinkability_key_repeat.png", dpi=200)
    plt.close()


# ============================================================
# Experiment 3: authentication overhead ratio
# ============================================================

def experiment_auth_overhead_ratio(task_sizes: list[int], task_type: str = "VIDEO_ANALYSIS"):
    dca_a = DCACluster(domain_id="A", n=5, t=3, prime=PRIME)
    dca_b = DCACluster(domain_id="B", n=5, t=3, prime=PRIME)

    ratios = []
    access_times = []
    key_times = []
    task_times = []
    total_times = []

    for task_size in task_sizes:
        aas_a = prepare_token_and_aas(dca_a, dca_b, threshold=3)
        ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=PRIME)
        uav = UAV(uav_id=f"UAV-RATIO-{task_size}", prime=PRIME)
        register_uav(dca_b, uav, threshold=3)

        # Access
        t0 = time.perf_counter()
        req = uav.generate_access_request(
            source_domain="B",
            target_domain="A",
            req_type="EDGE_SERVICE",
        )
        result = aas_a.verify_access_request(
            request=req,
            source_dca=dca_b,
            uav_public_key=uav.public_key,
        )
        t1 = time.perf_counter()

        if not result.success:
            raise RuntimeError(f"Access failed for size={task_size}: {result.reason}")

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
            raise RuntimeError(f"Service key mismatch for size={task_size}")

        # Task processing
        packet = uav.build_offload_packet(task_type=task_type, task_size=task_size)
        verified = ecs.verify_offload_packet(packet=packet, service_key=sk_ecs)
        if not verified:
            raise RuntimeError(f"Packet verification failed for size={task_size}")

        ecs.process_task(task_type=task_type, task_size=task_size)
        t3 = time.perf_counter()

        access_t = t1 - t0
        key_t = t2 - t1
        task_t = t3 - t2
        total_t = t3 - t0

        ratio = (access_t + key_t) / total_t if total_t > 0 else 0.0

        access_times.append(access_t)
        key_times.append(key_t)
        task_times.append(task_t)
        total_times.append(total_t)
        ratios.append(ratio)

        print(
            f"[AuthRatio] size={task_size}, access={access_t:.6f}s, key={key_t:.6f}s, "
            f"task={task_t:.6f}s, total={total_t:.6f}s, ratio={ratio:.6f}"
        )

    return {
        "task_sizes": task_sizes,
        "access_times": access_times,
        "key_times": key_times,
        "task_times": task_times,
        "total_times": total_times,
        "ratios": ratios,
    }


def plot_auth_overhead_ratio(results: dict, out_dir: Path):
    task_sizes = results["task_sizes"]
    ratios = results["ratios"]

    plt.figure(figsize=(8, 5))
    plt.plot(task_sizes, ratios, marker="o")
    plt.xlabel("Task size")
    plt.ylabel("Authentication overhead ratio")
    plt.title("Task size vs authentication overhead ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "auth_overhead_ratio.png", dpi=200)
    plt.close()


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Running threshold scaling experiment ===")
    threshold_results = experiment_threshold_scaling(
        configs=[(5, 3), (7, 3), (10, 4), (15, 5)]
    )
    plot_threshold_scaling(threshold_results, out_dir)

    print("\n=== Running unlinkability experiment ===")
    unlink_results = experiment_unlinkability(num_sessions=100)
    plot_unlinkability(unlink_results, out_dir)

    print("\n=== Running authentication overhead ratio experiment ===")
    ratio_results = experiment_auth_overhead_ratio(
        task_sizes=[2, 4, 8, 16, 32],
        task_type="VIDEO_ANALYSIS",
    )
    plot_auth_overhead_ratio(ratio_results, out_dir)

    print("\nExperiments finished. Figures saved under ./results/")


if __name__ == "__main__":
    main()
