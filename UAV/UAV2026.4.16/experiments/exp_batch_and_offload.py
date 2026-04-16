from __future__ import annotations

import csv
import time
from pathlib import Path
from statistics import mean, stdev

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from core.bls_utils import BLS_AVAILABLE, verify_signature as bls_verify_signature
from entities.aas import AAS
from entities.dca import DCACluster
from entities.ecs import ECS
from entities.uav import UAV

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


def summarize(rows: list[dict], group_key: str, value_keys: list[str], extra_group_key: str | None = None) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if extra_group_key is None:
            k = (row[group_key],)
        else:
            k = (row[group_key], row[extra_group_key])
        groups.setdefault(k, []).append(row)

    out: list[dict] = []
    for key, sub in sorted(groups.items(), key=lambda kv: kv[0]):
        item = {group_key: key[0], "count": len(sub)}
        if extra_group_key is not None:
            item[extra_group_key] = key[1]
        for vk in value_keys:
            vals = [r[vk] for r in sub]
            item[f"{vk}_mean"] = mean(vals)
            item[f"{vk}_std"] = stdev(vals) if len(vals) >= 2 else 0.0
        out.append(item)
    return out


def prepare_aas(dca_a: DCACluster, dca_b: DCACluster, threshold: int) -> AAS:
    aas = AAS(domain_id="A", dca_cluster=dca_a, replay_window=60)
    token = dca_b.generate_cross_domain_token(
        target_domain="A",
        available_count=threshold,
        validity_period=1800,
    )
    if token is None:
        raise RuntimeError("Failed to generate token")
    aas.install_cross_domain_token(source_domain="B", token=token)
    return aas


def register_uav(dca_b: DCACluster, uav: UAV, threshold: int) -> None:
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=threshold)
    ok = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)
    if not ok:
        raise RuntimeError("Registration failed")


def _verify_bls_individual_requests(
    aas_a: AAS,
    dca_b: DCACluster,
    bls_requests: list,
    bls_public_keys: list,
) -> tuple[int, dict[str, int]]:
    """
    BLS 逐个验签：
    1) 先复用 AAS 的跨域 token 校验
    2) 再对每条请求单独做 BLS verify
    """
    success = 0
    fail_reasons: dict[str, int] = {}

    token = aas_a.cross_domain_tokens.get("B")
    if token is None:
        return 0, {"No cross-domain token installed": len(bls_requests)}

    now_ts = int(time.time())
    valid_token = aas_a.dca_cluster.verify_cross_domain_token(
        token=token,
        source_dca=dca_b,
        now_ts=now_ts,
    )
    if not valid_token:
        return 0, {"Cross-domain token invalid": len(bls_requests)}

    for req, pk in zip(bls_requests, bls_public_keys):
        if req.target_domain != aas_a.domain_id:
            reason = "Wrong target domain"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            continue

        if abs(int(time.time()) - req.timestamp) > aas_a.replay_window:
            reason = "Timestamp expired or replayed"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            continue

        ok = bls_verify_signature(
            pk=pk,
            message=req.message_to_sign(),
            signature=req.signature,
        )
        if ok:
            success += 1
        else:
            reason = "BLS individual signature invalid"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    return success, fail_reasons


def run_batch_auth_trials(
    num_trials: int,
    batch_sizes: list[int],
    n: int = 5,
    t: int = 3,
) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []

    for trial in range(num_trials):
        dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
        dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)
        aas_a = prepare_aas(dca_a, dca_b, t)

        for batch_size in batch_sizes:
            uavs = []
            ecc_requests = []

            for i in range(batch_size):
                uav = UAV(uav_id=f"UAV-{trial}-{batch_size}-{i}", prime=PRIME)
                register_uav(dca_b, uav, t)

                req = uav.generate_access_request(
                    source_domain="B",
                    target_domain="A",
                    req_type="EDGE_SERVICE",
                    timestamp=int(time.time()),
                )
                uavs.append(uav)
                ecc_requests.append(req)

            # --------------------------------------------------
            # ECC sequential verify
            # --------------------------------------------------
            start = time.perf_counter()
            success = 0
            fail_reasons: dict[str, int] = {}

            for req in ecc_requests:
                result = aas_a.verify_access_request(
                    request=req,
                    source_dca=dca_b,
                    now_ts=int(time.time()),
                )
                if result.success:
                    success += 1
                else:
                    fail_reasons[result.reason] = fail_reasons.get(result.reason, 0) + 1

            total = time.perf_counter() - start
            row = {
                "trial": trial,
                "batch_size": batch_size,
                "mode": "ECC_seq",
                "success_count": success,
                "total_time": total,
                "avg_time": total / batch_size,
                "throughput": success / total if total > 0 else 0.0,
                "fail_reasons": str(fail_reasons),
            }
            raw_rows.append(row)

            print(
                f"[Batch ECC][trial={trial}] batch={batch_size} "
                f"success={success}/{batch_size} total={total:.6f}s "
                f"throughput={row['throughput']:.2f} fail_reasons={fail_reasons}"
            )

            # --------------------------------------------------
            # BLS paths
            # --------------------------------------------------
            if BLS_AVAILABLE:
                bls_requests = []
                bls_pks = []

                for uav in uavs:
                    bls_requests.append(
                        uav.generate_bls_access_request(
                            source_domain="B",
                            target_domain="A",
                            req_type="EDGE_SERVICE",
                            timestamp=int(time.time()),
                        )
                    )
                    bls_pks.append(uav.bls_public_key)

                # ----------------------------
                # BLS individual verify
                # ----------------------------
                start_ind = time.perf_counter()
                success_ind, fail_ind = _verify_bls_individual_requests(
                    aas_a=aas_a,
                    dca_b=dca_b,
                    bls_requests=bls_requests,
                    bls_public_keys=bls_pks,
                )
                total_ind = time.perf_counter() - start_ind

                row_ind = {
                    "trial": trial,
                    "batch_size": batch_size,
                    "mode": "BLS_individual",
                    "success_count": success_ind,
                    "total_time": total_ind,
                    "avg_time": total_ind / batch_size,
                    "throughput": success_ind / total_ind if total_ind > 0 else 0.0,
                    "fail_reasons": str(fail_ind),
                }
                raw_rows.append(row_ind)

                print(
                    f"[Batch BLS-IND][trial={trial}] batch={batch_size} "
                    f"success={success_ind}/{batch_size} total={total_ind:.6f}s "
                    f"throughput={row_ind['throughput']:.2f} fail_reasons={fail_ind}"
                )

                # ----------------------------
                # BLS aggregate verify
                # ----------------------------
                start_agg = time.perf_counter()
                result_bls = aas_a.batch_verify_bls_access_requests(
                    requests=bls_requests,
                    source_dca=dca_b,
                    public_keys=bls_pks,
                    now_ts=int(time.time()),
                )
                total_agg = time.perf_counter() - start_agg
                success_agg = batch_size if result_bls.success else 0

                row_agg = {
                    "trial": trial,
                    "batch_size": batch_size,
                    "mode": "BLS_aggregate",
                    "success_count": success_agg,
                    "total_time": total_agg,
                    "avg_time": total_agg / batch_size,
                    "throughput": success_agg / total_agg if total_agg > 0 else 0.0,
                    "fail_reasons": "{}" if result_bls.success else str({"batch_failed": batch_size}),
                }
                raw_rows.append(row_agg)

                print(
                    f"[Batch BLS-AGG][trial={trial}] batch={batch_size} "
                    f"success={success_agg}/{batch_size} total={total_agg:.6f}s "
                    f"throughput={row_agg['throughput']:.2f}"
                )

    summary_rows = summarize(
        raw_rows,
        group_key="batch_size",
        extra_group_key="mode",
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
    raw_rows: list[dict] = []

    for trial in range(num_trials):
        dca_a = DCACluster(domain_id="A", n=n, t=t, prime=PRIME)
        dca_b = DCACluster(domain_id="B", n=n, t=t, prime=PRIME)
        aas_a = prepare_aas(dca_a, dca_b, t)
        ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=PRIME)

        uav = UAV(uav_id=f"UAV-OFFLOAD-{trial}", prime=PRIME)
        register_uav(dca_b, uav, t)

        for task_size in task_sizes:
            req = uav.generate_access_request(
                source_domain="B",
                target_domain="A",
                req_type="EDGE_SERVICE",
                timestamp=int(time.time()),
            )
            result = aas_a.verify_access_request(
                request=req,
                source_dca=dca_b,
                now_ts=int(time.time()),
            )
            if not result.success:
                raise RuntimeError(result.reason)

            challenge = ecs.issue_service_challenge(timestamp=req.timestamp)
            nonce_u, _ = uav.respond_service_challenge(spid=req.spid, timestamp=challenge.timestamp)

            sk_u = uav.derive_service_key(
                spid=req.spid,
                ecs_id=ecs.ecs_id,
                domain_id="A",
                nonce_e=challenge.nonce_e,
                nonce_u=nonce_u,
                challenge_timestamp=challenge.timestamp,
                context=task_type,
            )
            sk_e = ecs.derive_service_key(
                uav_spid=req.spid,
                nonce_u=nonce_u,
                challenge=challenge,
                credential_hint=req.credential_public_key_b64,
                context=task_type,
            )
            if sk_u != sk_e:
                raise RuntimeError("Derived service keys do not match")

            start_edge = time.perf_counter()
            packet = uav.build_offload_packet(task_type=task_type, task_size=task_size)
            if not ecs.verify_offload_packet(packet, sk_e):
                raise RuntimeError("Offload verification failed")
            edge_res = ecs.process_task(task_type=task_type, task_size=task_size)
            edge_t = time.perf_counter() - start_edge

            start_local = time.perf_counter()
            local_res = uav.process_task_locally(task_type=task_type, task_size=task_size)
            local_t = time.perf_counter() - start_local

            row = {
                "trial": trial,
                "task_type": task_type,
                "task_size": task_size,
                "edge_total_time": edge_t,
                "local_total_time": local_t,
                "edge_faster": edge_t < local_t,
                "edge_processing_time": edge_res["measured_processing_time"],
                "local_processing_time": local_res["measured_processing_time"],
            }
            raw_rows.append(row)

            print(
                f"[Offload][trial={trial}] size={task_size} "
                f"edge={edge_t:.6f}s local={local_t:.6f}s"
            )

    summary_rows = summarize(
        raw_rows,
        group_key="task_size",
        value_keys=["edge_total_time", "local_total_time"],
    )
    return raw_rows, summary_rows


def plot_batch(raw_rows: list[dict]) -> None:
    if plt is None or not raw_rows:
        return

    modes = sorted(set(r["mode"] for r in raw_rows))
    plt.figure(figsize=(8, 5))

    for mode in modes:
        subset = [r for r in raw_rows if r["mode"] == mode]
        xs = sorted(set(r["batch_size"] for r in subset))
        ys = [mean(r["throughput"] for r in subset if r["batch_size"] == x) for x in xs]
        plt.plot(xs, ys, marker="o", label=mode)

    plt.xlabel("Batch size")
    plt.ylabel("Throughput (req/s)")
    plt.title("Authentication throughput comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "batch_auth_throughput.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for mode in modes:
        subset = [r for r in raw_rows if r["mode"] == mode]
        xs = sorted(set(r["batch_size"] for r in subset))
        ys = [mean(r["avg_time"] for r in subset if r["batch_size"] == x) for x in xs]
        plt.plot(xs, ys, marker="o", label=mode)

    plt.xlabel("Batch size")
    plt.ylabel("Average time per request (s)")
    plt.title("Average verification time per request")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "batch_auth_avg_time.png", dpi=200)
    plt.close()


def plot_offload(summary_rows: list[dict]) -> None:
    if plt is None or not summary_rows:
        return

    xs = [r["task_size"] for r in summary_rows]
    edge = [r["edge_total_time_mean"] for r in summary_rows]
    local = [r["local_total_time_mean"] for r in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, edge, marker="o", label="Edge offload")
    plt.plot(xs, local, marker="s", label="Local UAV")
    plt.xlabel("Task size")
    plt.ylabel("Total task time (s)")
    plt.title("MEC offload vs local processing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "offload_vs_local.png", dpi=200)
    plt.close()


def main() -> None:
    batch_raw, batch_summary = run_batch_auth_trials(
        num_trials=10,
        batch_sizes=[1, 5, 10, 20, 50],
    )
    write_csv(RESULTS_DIR / "batch_auth_raw.csv", batch_raw)
    write_csv(RESULTS_DIR / "batch_auth_summary.csv", batch_summary)
    plot_batch(batch_raw)

    off_raw, off_summary = run_task_scaling_trials(
        num_trials=10,
        task_type="VIDEO_ANALYSIS",
        task_sizes=[16, 32, 64, 128],
    )
    write_csv(RESULTS_DIR / "offload_raw.csv", off_raw)
    write_csv(RESULTS_DIR / "offload_summary.csv", off_summary)
    plot_offload(off_summary)


if __name__ == "__main__":
    main()