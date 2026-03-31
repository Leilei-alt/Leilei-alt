# experiments/exp_bls_batch.py
from __future__ import annotations

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from analysis.stats_utils import write_dict_rows_to_csv, group_and_summarize
from analysis.plot_utils import plot_line_with_errorbars, plot_multi_line
from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS


PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481
RAW_DIR = Path("results/raw")
SUMMARY_DIR = Path("results/summary")
FIG_DIR = Path("results/figures")


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
        raise RuntimeError("UAV registration failed")


def run_bls_batch_trials(
    num_trials: int,
    batch_sizes: list[int],
    n: int = 5,
    t: int = 3,
) -> tuple[list[dict], list[dict]]:
    raw_rows = []

    for trial in range(num_trials):
        dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
        dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)
        aas_a = prepare_aas_and_token(dca_a, dca_b, t)

        for batch_size in batch_sizes:
            uavs = []
            ecdsa_requests = []
            bls_requests = []
            bls_public_keys = []

            base_ts = int(time.time()) + trial  # avoid replay edge cases across trials

            for i in range(batch_size):
                uav = UAV(uav_id=f"UAV-BLS-{trial}-{batch_size}-{i}", prime=PRIME)
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
            success_ecdsa = 0
            for uav, req in zip(uavs, ecdsa_requests):
                result = aas_a.verify_access_request(
                    request=req,
                    source_dca=dca_b,
                    uav_public_key=uav.public_key,
                    now_ts=base_ts,
                )
                if result.success:
                    success_ecdsa += 1
            t1 = time.perf_counter()

            # BLS sequential verify
            success_bls_seq = 0
            for uav, req in zip(uavs, bls_requests):
                result = aas_a.verify_bls_access_request(
                    request=req,
                    source_dca=dca_b,
                    uav_bls_public_key=uav.bls_public_key,
                    now_ts=base_ts,
                )
                if result.success:
                    success_bls_seq += 1
            t2 = time.perf_counter()

            # BLS aggregate verify
            result = aas_a.batch_verify_bls_access_requests(
                requests=bls_requests,
                source_dca=dca_b,
                public_keys=bls_public_keys,
                now_ts=base_ts,
            )
            t3 = time.perf_counter()

            if success_ecdsa != batch_size:
                raise RuntimeError(f"ECDSA verification failed in trial={trial}, batch={batch_size}")
            if success_bls_seq != batch_size:
                raise RuntimeError(f"BLS sequential verification failed in trial={trial}, batch={batch_size}")
            if not result.success:
                raise RuntimeError(f"BLS aggregate verification failed in trial={trial}, batch={batch_size}: {result.reason}")

            ecdsa_time = t1 - t0
            bls_seq_time = t2 - t1
            bls_agg_time = t3 - t2
            speedup = bls_seq_time / bls_agg_time if bls_agg_time > 0 else 0.0

            row = {
                "trial": trial,
                "batch_size": batch_size,
                "ecdsa_seq_time": ecdsa_time,
                "bls_seq_time": bls_seq_time,
                "bls_agg_time": bls_agg_time,
                "bls_speedup": speedup,
            }
            raw_rows.append(row)

            print(
                f"[BLSBatch][trial={trial}] batch={batch_size}, "
                f"ECDSA_seq={ecdsa_time:.6f}s, "
                f"BLS_seq={bls_seq_time:.6f}s, "
                f"BLS_agg={bls_agg_time:.6f}s, "
                f"speedup={speedup:.3f}x"
            )

    summary_rows = group_and_summarize(
        raw_rows,
        group_key="batch_size",
        value_keys=["ecdsa_seq_time", "bls_seq_time", "bls_agg_time", "bls_speedup"],
    )
    return raw_rows, summary_rows


def save_and_plot_bls_results(raw_rows: list[dict], summary_rows: list[dict]):
    write_dict_rows_to_csv(RAW_DIR / "bls_batch_raw.csv", raw_rows)
    write_dict_rows_to_csv(SUMMARY_DIR / "bls_batch_summary.csv", summary_rows)

    x = [int(r["batch_size"]) for r in summary_rows]

    # Main comparison figure with error bars
    plot_multi_line(
        x_vals=x,
        series=[
            ("ECDSA sequential", [r["ecdsa_seq_time_mean"] for r in summary_rows]),
            ("BLS sequential", [r["bls_seq_time_mean"] for r in summary_rows]),
            ("BLS aggregate", [r["bls_agg_time_mean"] for r in summary_rows]),
        ],
        xlabel="Number of UAVs",
        ylabel="Verification time (s)",
        title="ECDSA vs BLS batch verification (mean)",
        out_path=FIG_DIR / "bls_batch_compare_mean.png",
    )

    # Add an error-bar version for aggregate only
    plot_line_with_errorbars(
        x_vals=x,
        y_means=[r["bls_agg_time_mean"] for r in summary_rows],
        y_stds=[r["bls_agg_time_std"] for r in summary_rows],
        xlabel="Number of UAVs",
        ylabel="BLS aggregate verification time (s)",
        title="BLS aggregate verification time (mean ± std)",
        out_path=FIG_DIR / "bls_batch_aggregate_mean_std.png",
    )

    # Speedup figure
    plot_line_with_errorbars(
        x_vals=x,
        y_means=[r["bls_speedup_mean"] for r in summary_rows],
        y_stds=[r["bls_speedup_std"] for r in summary_rows],
        xlabel="Number of UAVs",
        ylabel="Speedup (BLS seq / BLS agg)",
        title="BLS aggregate verification speedup (mean ± std)",
        out_path=FIG_DIR / "bls_batch_speedup.png",
    )


def main():
    num_trials = 20
    batch_sizes = [1, 5, 10, 20, 50, 100]

    raw_rows, summary_rows = run_bls_batch_trials(
        num_trials=num_trials,
        batch_sizes=batch_sizes,
        n=5,
        t=3,
    )
    save_and_plot_bls_results(raw_rows, summary_rows)

    print("\nDone.")
    print("Raw CSV saved under ./results/raw/bls_batch_raw.csv")
    print("Summary CSV saved under ./results/summary/bls_batch_summary.csv")
    print("Figures saved under ./results/figures/")


if __name__ == "__main__":
    main()
