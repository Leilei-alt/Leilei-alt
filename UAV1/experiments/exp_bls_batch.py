# experiments/exp_bls_batch.py
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt

from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS


PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481


def prepare_aas_and_token(dca_a: DCACluster, dca_b: DCACluster, threshold: int) -> AAS:
    aas_a = AAS(domain_id="A", dca_cluster=dca_a, replay_window=10)
    token = dca_b.generate_cross_domain_token(
        target_domain="A",
        available_count=threshold,
        validity_period=300,
    )
    if token is None:
        raise RuntimeError("Failed to generate token")
    aas_a.install_cross_domain_token(source_domain="B", token=token)
    return aas_a


def register_uav(dca_b: DCACluster, uav: UAV, threshold: int):
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=threshold)
    ok = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)
    if not ok:
        raise RuntimeError("UAV registration failed")


def run_bls_batch_experiment(batch_sizes: list[int], n: int = 5, t: int = 3):
    ecdsa_seq_times = []
    bls_seq_times = []
    bls_agg_times = []

    for batch_size in batch_sizes:
        dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
        dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)
        aas_a = prepare_aas_and_token(dca_a, dca_b, t)

        uavs = []
        ecdsa_requests = []
        bls_requests = []
        bls_public_keys = []

        base_ts = int(time.time())

        for i in range(batch_size):
            uav = UAV(uav_id=f"UAV-BLS-{batch_size}-{i}", prime=PRIME)
            register_uav(dca_b, uav, t)

            e_req = uav.generate_access_request(
                source_domain="B",
                target_domain="A",
                req_type="EDGE_SERVICE",
                timestamp=base_ts,
            )
            b_req = uav.generate_bls_access_request(
                source_domain="B",
                target_domain="A",
                req_type="EDGE_SERVICE",
                timestamp=base_ts,
            )

            uavs.append(uav)
            ecdsa_requests.append(e_req)
            bls_requests.append(b_req)
            bls_public_keys.append(uav.bls_public_key)

        # ECDSA sequential verify
        t0 = time.perf_counter()
        for uav, req in zip(uavs, ecdsa_requests):
            result = aas_a.verify_access_request(
                request=req,
                source_dca=dca_b,
                uav_public_key=uav.public_key,
                now_ts=base_ts,
            )
            if not result.success:
                raise RuntimeError(f"ECDSA verify failed: {result.reason}")
        t1 = time.perf_counter()

        # BLS sequential verify
        for uav, req in zip(uavs, bls_requests):
            result = aas_a.verify_bls_access_request(
                request=req,
                source_dca=dca_b,
                uav_bls_public_key=uav.bls_public_key,
                now_ts=base_ts,
            )
            if not result.success:
                raise RuntimeError(f"BLS single verify failed: {result.reason}")
        t2 = time.perf_counter()

        # BLS aggregate verify
        result = aas_a.batch_verify_bls_access_requests(
            requests=bls_requests,
            source_dca=dca_b,
            public_keys=bls_public_keys,
            now_ts=base_ts,
        )
        t3 = time.perf_counter()

        if not result.success:
            raise RuntimeError(f"BLS aggregate verify failed: {result.reason}")

        ecdsa_time = t1 - t0
        bls_seq_time = t2 - t1
        bls_agg_time = t3 - t2

        ecdsa_seq_times.append(ecdsa_time)
        bls_seq_times.append(bls_seq_time)
        bls_agg_times.append(bls_agg_time)

        print(
            f"[BLSBatch] batch={batch_size}, "
            f"ECDSA_seq={ecdsa_time:.6f}s, "
            f"BLS_seq={bls_seq_time:.6f}s, "
            f"BLS_agg={bls_agg_time:.6f}s"
        )

    return {
        "batch_sizes": batch_sizes,
        "ecdsa_seq_times": ecdsa_seq_times,
        "bls_seq_times": bls_seq_times,
        "bls_agg_times": bls_agg_times,
    }


def plot_bls_results(results: dict):
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    x = results["batch_sizes"]

    plt.figure(figsize=(8, 5))
    plt.plot(x, results["ecdsa_seq_times"], marker="o", label="ECDSA sequential")
    plt.plot(x, results["bls_seq_times"], marker="o", label="BLS sequential")
    plt.plot(x, results["bls_agg_times"], marker="o", label="BLS aggregate")
    plt.xlabel("Number of UAVs")
    plt.ylabel("Verification time (s)")
    plt.title("ECDSA vs BLS batch verification")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "bls_batch_compare.png", dpi=200)
    plt.close()


def main():
    results = run_bls_batch_experiment(batch_sizes=[1, 5, 10, 20, 50, 100])
    plot_bls_results(results)
    print("BLS batch experiment finished. Figure saved to results/figures/bls_batch_compare.png")


if __name__ == "__main__":
    main()
