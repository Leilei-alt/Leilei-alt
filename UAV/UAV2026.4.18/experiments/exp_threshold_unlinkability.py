from __future__ import annotations

import csv
import time
from pathlib import Path
from statistics import mean, stdev

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS

PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_aas_and_token(dca_a: DCACluster, dca_b: DCACluster, threshold: int) -> AAS:
    aas_a = AAS(domain_id="A", dca_cluster=dca_a, replay_window=10)
    token = dca_b.generate_cross_domain_token(target_domain="A", available_count=threshold, validity_period=300)
    if token is None:
        raise RuntimeError("Failed to generate cross-domain token.")
    aas_a.install_cross_domain_token(source_domain="B", token=token)
    return aas_a


def register_uav(dca_b: DCACluster, uav: UAV, threshold: int) -> None:
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=threshold)
    ok = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)
    if not ok:
        raise RuntimeError("UAV registration failed.")


def experiment_threshold_scaling(
    configs: list[tuple[int, int]],
    num_trials: int = 30,
    warmup_runs: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Return:
        raw_rows: every valid trial after warm-up
        summary_rows: mean/std/min/max for each (n,t)
    """
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    for n, t in configs:
        reg_vals: list[float] = []
        tok_vals: list[float] = []

        total_runs = warmup_runs + num_trials
        for trial in range(total_runs):
            dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)
            uav = UAV(uav_id=f"UAV-TH-{n}-{t}-{trial}", prime=PRIME)

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
            reg_time = time.perf_counter() - start_reg
            if not ok:
                raise RuntimeError(f"Registration failed for config {(n, t)} at trial {trial}")

            start_tok = time.perf_counter()
            token = dca_b.generate_cross_domain_token(
                target_domain="A",
                available_count=t,
                validity_period=300,
            )
            tok_time = time.perf_counter() - start_tok
            if token is None:
                raise RuntimeError(f"Token generation failed for config {(n, t)} at trial {trial}")

            # Skip warm-up runs
            if trial < warmup_runs:
                continue

            reg_vals.append(reg_time)
            tok_vals.append(tok_time)

            row = {
                "n": n,
                "t": t,
                "trial": trial - warmup_runs,
                "registration_time_s": reg_time,
                "token_time_s": tok_time,
            }
            raw_rows.append(row)

        reg_mean = mean(reg_vals)
        tok_mean = mean(tok_vals)
        reg_std = stdev(reg_vals) if len(reg_vals) >= 2 else 0.0
        tok_std = stdev(tok_vals) if len(tok_vals) >= 2 else 0.0

        summary = {
            "n": n,
            "t": t,
            "count": len(reg_vals),
            "registration_time_mean_s": reg_mean,
            "registration_time_std_s": reg_std,
            "registration_time_min_s": min(reg_vals),
            "registration_time_max_s": max(reg_vals),
            "token_time_mean_s": tok_mean,
            "token_time_std_s": tok_std,
            "token_time_min_s": min(tok_vals),
            "token_time_max_s": max(tok_vals),
        }
        summary_rows.append(summary)

        print(
            f"[ThresholdScaling] (n,t)=({n},{t}) "
            f"reg_mean={reg_mean:.6f}s reg_std={reg_std:.6f}s "
            f"tok_mean={tok_mean:.6f}s tok_std={tok_std:.6f}s "
            f"(count={len(reg_vals)})"
        )

    return raw_rows, summary_rows


def experiment_unlinkability(num_sessions: int = 100) -> tuple[list[dict], dict]:
    dca_a = DCACluster(domain_id="A", n=5, t=3, prime=PRIME)
    dca_b = DCACluster(domain_id="B", n=5, t=3, prime=PRIME)
    aas_a = prepare_aas_and_token(dca_a, dca_b, threshold=3)

    uav = UAV(uav_id="UAV-UNLINK-01", prime=PRIME)
    register_uav(dca_b, uav, threshold=3)

    rows: list[dict] = []
    spids: list[str] = []
    eph_keys: list[str] = []

    base_ts = int(time.time())
    for i in range(num_sessions):
        req = uav.generate_access_request(
            source_domain="B",
            target_domain="A",
            req_type="EDGE_SERVICE",
            timestamp=base_ts + i,
        )
        result = aas_a.verify_access_request(request=req, source_dca=dca_b, now_ts=base_ts + i)
        if not result.success:
            raise RuntimeError(f"Unlinkability run failed at session {i}: {result.reason}")

        spids.append(req.spid)
        eph_keys.append(req.ephemeral_public_key_b64)
        rows.append(
            {
                "session": i,
                "pseudo_id": req.pseudo_id,
                "spid": req.spid,
                "ephemeral_public_key_prefix": req.ephemeral_public_key_b64[:32],
                "timestamp": req.timestamp,
            }
        )

    summary = {
        "sessions": num_sessions,
        "stable_pseudo_id": len({r['pseudo_id'] for r in rows}) == 1,
        "unique_spid_count": len(set(spids)),
        "unique_ephemeral_key_count": len(set(eph_keys)),
        "spid_uniqueness_ratio": len(set(spids)) / max(1, num_sessions),
        "ephemeral_uniqueness_ratio": len(set(eph_keys)) / max(1, num_sessions),
    }
    print(f"[Unlinkability] unique_spid={summary['unique_spid_count']}/{num_sessions}, unique_eph={summary['unique_ephemeral_key_count']}/{num_sessions}")
    return rows, summary


def plot_threshold_scaling(summary_rows: list[dict]) -> None:
    if plt is None or not summary_rows:
        return

    labels = [f"({r['n']},{r['t']})" for r in summary_rows]

    reg_means = [r["registration_time_mean_s"] for r in summary_rows]
    reg_stds = [r["registration_time_std_s"] for r in summary_rows]

    tok_means = [r["token_time_mean_s"] for r in summary_rows]
    tok_stds = [r["token_time_std_s"] for r in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, reg_means, yerr=reg_stds, capsize=4)
    plt.xlabel("(n,t) configuration")
    plt.ylabel("Registration time (s)")
    plt.title("ECC-anonymous registration under threshold DCA (mean ± std)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "threshold_registration_time_mean_std.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(labels, tok_means, yerr=tok_stds, capsize=4)
    plt.xlabel("(n,t) configuration")
    plt.ylabel("Cross-domain token generation time (s)")
    plt.title("Cross-domain token issuance under threshold DCA (mean ± std)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "threshold_token_time_mean_std.png", dpi=200)
    plt.close()


def main() -> None:
    configs = [(3, 2), (5, 3), (7, 3), (7, 5), (9, 5)]

    threshold_raw_rows, threshold_summary_rows = experiment_threshold_scaling(
        configs=configs,
        num_trials=30,
        warmup_runs=3,
    )
    write_csv(RESULTS_DIR / "threshold_scaling_raw.csv", threshold_raw_rows)
    write_csv(RESULTS_DIR / "threshold_scaling_summary.csv", threshold_summary_rows)
    plot_threshold_scaling(threshold_summary_rows)

    unlink_rows, unlink_summary = experiment_unlinkability(num_sessions=100)
    write_csv(RESULTS_DIR / "unlinkability_sessions.csv", unlink_rows)
    write_csv(RESULTS_DIR / "unlinkability_summary.csv", [unlink_summary])

    avg_reg = mean([r["registration_time_mean_s"] for r in threshold_summary_rows])
    avg_tok = mean([r["token_time_mean_s"] for r in threshold_summary_rows])

    print(f"[Summary] avg_registration_mean={avg_reg:.6f}s avg_token_mean={avg_tok:.6f}s")
    print(f"[Summary] unlinkability_summary={unlink_summary}")
    print(f"[Saved] raw threshold rows -> {RESULTS_DIR / 'threshold_scaling_raw.csv'}")
    print(f"[Saved] summary threshold rows -> {RESULTS_DIR / 'threshold_scaling_summary.csv'}")

if __name__ == "__main__":
    main()
