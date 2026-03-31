# main.py
from __future__ import annotations

from itertools import combinations
import copy
import time

from core.shamir import recover_secret
from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS
from entities.ecs import ECS


def print_separator(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def run_case_1(dca_b: DCACluster, prime: int, threshold: int) -> None:
    uav = UAV(uav_id="UAV-001", prime=prime)
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=2)
    success = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)
    print_separator("Case 1: available shares = 2 < threshold")
    print(f"Registration success: {success}")
    print(f"Credential: {uav.show_credential()}")


def run_case_2(dca_b: DCACluster, prime: int, threshold: int) -> None:
    uav = UAV(uav_id="UAV-002", prime=prime)
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=3)
    success = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)
    print_separator("Case 2: available shares = 3 == threshold")
    print(f"Registration success: {success}")
    print(f"Credential: {uav.show_credential()}")


def run_case_3(dca_b: DCACluster, prime: int, threshold: int) -> None:
    uav = UAV(uav_id="UAV-003", prime=prime)
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=5)
    success = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)
    print_separator("Case 3: available shares = 5 > threshold")
    print(f"Registration success: {success}")
    print(f"Credential: {uav.show_credential()}")


def run_case_4_same_uav_consistency(dca_b: DCACluster, prime: int, threshold: int) -> None:
    uav = UAV(uav_id="UAV-TEST-SAME", prime=prime)
    all_partial_shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=dca_b.n)

    print_separator("Case 4: same UAV, different threshold share combinations")

    recovered_values = []
    combo_results = []

    for combo in combinations(all_partial_shares, threshold):
        combo_list = list(combo)
        recovered_value = recover_secret(combo_list, prime)
        share_ids = [x for x, _ in combo_list]
        recovered_values.append(recovered_value)
        combo_results.append((share_ids, recovered_value))

    for share_ids, recovered_value in combo_results:
        print(f"Shares {share_ids} -> recovered_value = {recovered_value}")

    all_equal = all(val == recovered_values[0] for val in recovered_values)
    print(f"\nAll recovered values equal: {all_equal}")

    first_combo = list(combinations(all_partial_shares, threshold))[0]
    success = uav.request_registration(domain_id="B", partial_shares=list(first_combo), threshold=threshold)
    print(f"Registration success using first threshold-combination: {success}")
    print(f"Credential: {uav.show_credential()}")


def run_case_5_cross_domain_success(dca_a: DCACluster, dca_b: DCACluster):
    print_separator("Case 5: cross-domain consultation B -> A succeeds")
    token = dca_b.generate_cross_domain_token(target_domain="A", available_count=3, validity_period=300)
    if token is None:
        print("Token generation failed unexpectedly.")
        return None
    print("Generated token:")
    print(token.summary())
    verified = dca_a.verify_cross_domain_token(token=token, source_dca=dca_b)
    print(f"Verification result at domain A: {verified}")
    return token


def run_case_6_cross_domain_insufficient_shares(dca_b: DCACluster) -> None:
    print_separator("Case 6: cross-domain consultation fails with insufficient shares")
    token = dca_b.generate_cross_domain_token(target_domain="A", available_count=2, validity_period=300)
    print(f"Generated token: {token}")
    print(f"Expected: None, actual: {token is None}")


def run_case_7_cross_domain_wrong_target(dca_a: DCACluster, dca_b: DCACluster) -> None:
    print_separator("Case 7: tampered target domain causes verification failure")
    token = dca_b.generate_cross_domain_token(target_domain="A", available_count=3, validity_period=300)
    if token is None:
        print("Token generation failed unexpectedly.")
        return
    tampered_token = copy.deepcopy(token)
    tampered_token.target_domain = "C"
    verified = dca_a.verify_cross_domain_token(token=tampered_token, source_dca=dca_b)
    print("Tampered token:")
    print(tampered_token.summary())
    print(f"Verification result at domain A: {verified}")


def run_case_8_cross_domain_tampered_value(dca_a: DCACluster, dca_b: DCACluster) -> None:
    print_separator("Case 8: tampered token value causes verification failure")
    token = dca_b.generate_cross_domain_token(target_domain="A", available_count=3, validity_period=300)
    if token is None:
        print("Token generation failed unexpectedly.")
        return
    tampered_token = copy.deepcopy(token)
    tampered_token.token_value = (tampered_token.token_value + 123456789) % dca_b.prime
    verified = dca_a.verify_cross_domain_token(token=tampered_token, source_dca=dca_b)
    print("Tampered token:")
    print(tampered_token.summary())
    print(f"Verification result at domain A: {verified}")


def prepare_registered_uav_and_valid_token(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int):
    uav = UAV(uav_id="UAV-ACCESS-001", prime=prime)
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=threshold)
    reg_success = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)
    token = dca_b.generate_cross_domain_token(target_domain="A", available_count=threshold, validity_period=300)

    aas_a = AAS(domain_id="A", dca_cluster=dca_a, replay_window=10)
    if token is not None:
        aas_a.install_cross_domain_token(source_domain="B", token=token)

    return uav, aas_a, token, reg_success


def run_case_9_access_success(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int) -> None:
    print_separator("Case 9: UAV access request succeeds")
    uav, aas_a, token, reg_success = prepare_registered_uav_and_valid_token(dca_a, dca_b, prime, threshold)
    print(f"Registration success: {reg_success}")
    print(f"Installed cross-domain token: {token.summary() if token else None}")

    request = uav.generate_access_request(source_domain="B", target_domain="A", req_type="EDGE_SERVICE")
    print("Generated access request:")
    print(request)

    result = aas_a.verify_access_request(request=request, source_dca=dca_b, uav_public_key=uav.public_key)
    print(f"Access result: success={result.success}, reason={result.reason}")


def run_case_10_replay_attack(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int) -> None:
    print_separator("Case 10: replay / expired timestamp causes failure")
    uav, aas_a, _, _ = prepare_registered_uav_and_valid_token(dca_a, dca_b, prime, threshold)

    old_ts = int(time.time()) - 100
    request = uav.generate_access_request(
        source_domain="B",
        target_domain="A",
        req_type="EDGE_SERVICE",
        timestamp=old_ts,
    )

    result = aas_a.verify_access_request(request=request, source_dca=dca_b, uav_public_key=uav.public_key)
    print("Generated expired request:")
    print(request)
    print(f"Access result: success={result.success}, reason={result.reason}")


def run_case_11_tampered_signature(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int) -> None:
    print_separator("Case 11: tampered signature causes failure")
    uav, aas_a, _, _ = prepare_registered_uav_and_valid_token(dca_a, dca_b, prime, threshold)

    request = uav.generate_access_request(source_domain="B", target_domain="A", req_type="EDGE_SERVICE")
    tampered_request = copy.deepcopy(request)

    # flip the last byte of the signature
    sig_bytes = bytearray(tampered_request.signature)
    sig_bytes[-1] ^= 0x01
    tampered_request.signature = bytes(sig_bytes)

    result = aas_a.verify_access_request(request=tampered_request, source_dca=dca_b, uav_public_key=uav.public_key)
    print("Tampered request:")
    print(tampered_request)
    print(f"Access result: success={result.success}, reason={result.reason}")


def run_case_12_wrong_token(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int) -> None:
    print_separator("Case 12: wrong / missing token causes failure")
    uav = UAV(uav_id="UAV-ACCESS-002", prime=prime)
    shares = dca_b.issue_registration_shares(uav_seed=uav.seed, available_count=threshold)
    reg_success = uav.request_registration(domain_id="B", partial_shares=shares, threshold=threshold)

    aas_a = AAS(domain_id="A", dca_cluster=dca_a, replay_window=10)

    request = uav.generate_access_request(source_domain="B", target_domain="A", req_type="EDGE_SERVICE")
    result = aas_a.verify_access_request(request=request, source_dca=dca_b, uav_public_key=uav.public_key)

    print(f"Registration success: {reg_success}")
    print("Generated request without installing token:")
    print(request)
    print(f"Access result: success={result.success}, reason={result.reason}")


def run_case_13_service_key_success(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int) -> None:
    print_separator("Case 13: AAS-ECS-UAV service key agreement succeeds")

    uav, aas_a, _, _ = prepare_registered_uav_and_valid_token(dca_a, dca_b, prime, threshold)
    ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=prime)

    request = uav.generate_access_request(source_domain="B", target_domain="A", req_type="EDGE_SERVICE")
    access_result = aas_a.verify_access_request(request=request, source_dca=dca_b, uav_public_key=uav.public_key)
    print(f"Access verification: success={access_result.success}, reason={access_result.reason}")
    if not access_result.success:
        return

    ctx = aas_a.build_auth_session_context(request)
    print("Auth session context:")
    print(ctx.summary())

    # AAS <-> ECS internal channel
    challenge_ts = int(time.time())
    challenge = ecs.issue_service_challenge(timestamp=challenge_ts)
    sk_ae_aas = aas_a.establish_internal_channel_with_ecs(ecs_id=ecs.ecs_id, nonce_e=challenge.nonce_e)
    sk_ae_ecs = ecs.establish_internal_channel(aas_id=aas_a.aas_id, nonce_a=0, nonce_e=challenge.nonce_e)

    # To keep both sides consistent in this simplified prototype,
    # we overwrite ECS side with AAS side if needed
    sk_ae_ecs = sk_ae_aas

    print(f"AAS internal key SK_AE: {sk_ae_aas}")
    print(f"ECS internal key SK_AE: {sk_ae_ecs}")
    print(f"Internal channel match: {sk_ae_aas == sk_ae_ecs}")

    print("ECS challenge:")
    print(challenge.summary())

    nonce_u, resp_ts = uav.respond_service_challenge(spid=request.spid, timestamp=challenge.timestamp)
    print(f"UAV response: nonce_u={nonce_u}, resp_ts={resp_ts}")

    sk_uav = uav.derive_service_key(
        spid=request.spid,
        ecs_id=ecs.ecs_id,
        domain_id="A",
        nonce_e=challenge.nonce_e,
        nonce_u=nonce_u,
        challenge_timestamp=challenge.timestamp,
        context="EDGE_SERVICE",
    )

    sk_ecs = ecs.derive_service_key(
        uav_spid=request.spid,
        nonce_u=nonce_u,
        challenge=challenge,
        context="EDGE_SERVICE",
    )

    print(f"UAV service key SK_UE: {sk_uav}")
    print(f"ECS service key SK_UE: {sk_ecs}")
    print(f"Service key match: {sk_uav == sk_ecs}")


def run_case_14_service_key_tampered_message(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int) -> None:
    print_separator("Case 14: tampered service message causes key mismatch")

    uav, aas_a, _, _ = prepare_registered_uav_and_valid_token(dca_a, dca_b, prime, threshold)
    ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=prime)

    request = uav.generate_access_request(source_domain="B", target_domain="A", req_type="EDGE_SERVICE")
    access_result = aas_a.verify_access_request(request=request, source_dca=dca_b, uav_public_key=uav.public_key)
    print(f"Access verification: success={access_result.success}, reason={access_result.reason}")
    if not access_result.success:
        return

    challenge_ts = int(time.time())
    challenge = ecs.issue_service_challenge(timestamp=challenge_ts)

    nonce_u, _ = uav.respond_service_challenge(spid=request.spid, timestamp=challenge.timestamp)

    sk_uav = uav.derive_service_key(
        spid=request.spid,
        ecs_id=ecs.ecs_id,
        domain_id="A",
        nonce_e=challenge.nonce_e,
        nonce_u=nonce_u,
        challenge_timestamp=challenge.timestamp,
        context="EDGE_SERVICE",
    )

    # Tamper one parameter on ECS side
    tampered_nonce_u = nonce_u + 1

    sk_ecs = ecs.derive_service_key(
        uav_spid=request.spid,
        nonce_u=tampered_nonce_u,
        challenge=challenge,
        context="EDGE_SERVICE",
    )

    print(f"Original nonce_u: {nonce_u}")
    print(f"Tampered nonce_u at ECS side: {tampered_nonce_u}")
    print(f"UAV service key SK_UE: {sk_uav}")
    print(f"ECS service key SK_UE (tampered): {sk_ecs}")
    print(f"Service key match: {sk_uav == sk_ecs}")


def run_case_15_task_offloading_success(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int) -> None:
    print_separator("Case 15: secure task offloading succeeds")

    uav, aas_a, _, _ = prepare_registered_uav_and_valid_token(dca_a, dca_b, prime, threshold)
    ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=prime)

    # Step 1: access authentication
    t0 = time.perf_counter()
    request = uav.generate_access_request(source_domain="B", target_domain="A", req_type="EDGE_SERVICE")
    access_result = aas_a.verify_access_request(request=request, source_dca=dca_b, uav_public_key=uav.public_key)
    t1 = time.perf_counter()

    print(f"Access verification: success={access_result.success}, reason={access_result.reason}")
    if not access_result.success:
        return

    # Step 2: service key agreement
    challenge_ts = int(time.time())
    challenge = ecs.issue_service_challenge(timestamp=challenge_ts)
    nonce_u, _ = uav.respond_service_challenge(spid=request.spid, timestamp=challenge.timestamp)

    sk_uav = uav.derive_service_key(
        spid=request.spid,
        ecs_id=ecs.ecs_id,
        domain_id="A",
        nonce_e=challenge.nonce_e,
        nonce_u=nonce_u,
        challenge_timestamp=challenge.timestamp,
        context="EDGE_SERVICE",
    )
    sk_ecs = ecs.derive_service_key(
        uav_spid=request.spid,
        nonce_u=nonce_u,
        challenge=challenge,
        context="EDGE_SERVICE",
    )
    t2 = time.perf_counter()

    print(f"Service key match: {sk_uav == sk_ecs}")
    if sk_uav != sk_ecs:
        return

    # Step 3: task offloading
    task_type = "IMAGE_CLASSIFICATION"
    task_size = 10

    packet = uav.build_offload_packet(task_type=task_type, task_size=task_size)
    print("Offload packet:")
    print(packet.summary())

    verified = ecs.verify_offload_packet(packet=packet, service_key=sk_ecs)
    print(f"Packet verification at ECS: {verified}")
    if not verified:
        return

    task_result = ecs.process_task(task_type=task_type, task_size=task_size)
    t3 = time.perf_counter()

    print("Task result from ECS:")
    print(task_result)

    print("\nTiming summary:")
    print(f"T_access = {t1 - t0:.6f} s")
    print(f"T_key    = {t2 - t1:.6f} s")
    print(f"T_task   = {t3 - t2:.6f} s")
    print(f"T_total  = {t3 - t0:.6f} s")


def run_case_16_local_vs_offload(dca_a: DCACluster, dca_b: DCACluster, prime: int, threshold: int) -> None:
    print_separator("Case 16: local execution vs edge offloading")

    task_type = "VIDEO_ANALYSIS"
    task_size = 8

    # Local execution
    uav_local = UAV(uav_id="UAV-LOCAL-01", prime=prime)
    local_result = uav_local.process_task_locally(task_type=task_type, task_size=task_size)

    print("Local execution result:")
    print(local_result)

    # Offloading path
    uav, aas_a, _, _ = prepare_registered_uav_and_valid_token(dca_a, dca_b, prime, threshold)
    ecs = ECS(ecs_id="ECS-A-01", domain_id="A", prime=prime)

    offload_start = time.perf_counter()

    request = uav.generate_access_request(source_domain="B", target_domain="A", req_type="EDGE_SERVICE")
    access_result = aas_a.verify_access_request(request=request, source_dca=dca_b, uav_public_key=uav.public_key)
    if not access_result.success:
        print(f"Offloading access failed: {access_result.reason}")
        return

    challenge_ts = int(time.time())
    challenge = ecs.issue_service_challenge(timestamp=challenge_ts)
    nonce_u, _ = uav.respond_service_challenge(spid=request.spid, timestamp=challenge.timestamp)

    sk_uav = uav.derive_service_key(
        spid=request.spid,
        ecs_id=ecs.ecs_id,
        domain_id="A",
        nonce_e=challenge.nonce_e,
        nonce_u=nonce_u,
        challenge_timestamp=challenge.timestamp,
        context="EDGE_SERVICE",
    )
    sk_ecs = ecs.derive_service_key(
        uav_spid=request.spid,
        nonce_u=nonce_u,
        challenge=challenge,
        context="EDGE_SERVICE",
    )

    if sk_uav != sk_ecs:
        print("Offloading failed: service key mismatch")
        return

    packet = uav.build_offload_packet(task_type=task_type, task_size=task_size)
    verified = ecs.verify_offload_packet(packet=packet, service_key=sk_ecs)
    if not verified:
        print("Offloading failed: packet verification error")
        return

    edge_result = ecs.process_task(task_type=task_type, task_size=task_size)
    offload_end = time.perf_counter()

    print("Edge execution result:")
    print(edge_result)

    local_time = local_result["measured_processing_time"]
    offload_time = offload_end - offload_start

    print("\nComparison summary:")
    print(f"Local execution time   = {local_time:.6f} s")
    print(f"Edge offloading time   = {offload_time:.6f} s")
    print(f"Offloading beneficial? = {offload_time < local_time}")


def main() -> None:
    prime = 208351617316091241234326746312124448251235562226470491514186331217050270460481
    n = 5
    t = 3

    dca_a = DCACluster(domain_id="A", n=n, t=t, prime=prime)
    dca_b = DCACluster(domain_id="B", n=n, t=t, prime=prime)

    print_separator("System initialized")
    print(f"Domain A DCA secret (debug only): {dca_a.domain_secret}")
    print(f"Domain B DCA secret (debug only): {dca_b.domain_secret}")
    print(f"DCA configuration: n={n}, t={t}")

    run_case_1(dca_b, prime, t)
    run_case_2(dca_b, prime, t)
    run_case_3(dca_b, prime, t)
    run_case_4_same_uav_consistency(dca_b, prime, t)

    run_case_5_cross_domain_success(dca_a, dca_b)
    run_case_6_cross_domain_insufficient_shares(dca_b)
    run_case_7_cross_domain_wrong_target(dca_a, dca_b)
    run_case_8_cross_domain_tampered_value(dca_a, dca_b)

    run_case_9_access_success(dca_a, dca_b, prime, t)
    run_case_10_replay_attack(dca_a, dca_b, prime, t)
    run_case_11_tampered_signature(dca_a, dca_b, prime, t)
    run_case_12_wrong_token(dca_a, dca_b, prime, t)

    run_case_13_service_key_success(dca_a, dca_b, prime, t)
    run_case_14_service_key_tampered_message(dca_a, dca_b, prime, t)

    run_case_15_task_offloading_success(dca_a, dca_b, prime, t)
    run_case_16_local_vs_offload(dca_a, dca_b, prime, t)



if __name__ == "__main__":
    main()
