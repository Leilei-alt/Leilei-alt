from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS
from entities.ecs import ECS

PRIME = 208351617316091241234326746312124448251235562226470491514186331217050270460481


def main():
    # 1. 建域
    dca_a = DCACluster(domain_id="A", n=5, t=3, prime=PRIME)
    dca_b = DCACluster(domain_id="B", n=5, t=3, prime=PRIME)

    # 2. 目标域 A 的 AAS / ECS
    aas_a = AAS(domain_id="A", dca_cluster=dca_a, replay_window=10)
    ecs_a = ECS(ecs_id="ECS-A-01", domain_id="A", prime=PRIME)

    # 3. B 域 UAV
    uav = UAV(uav_id="UAV-001", prime=PRIME)

    # 4. B 域为 UAV 发匿名注册份额
    shares = dca_b.issue_registration_shares(
        uav_seed=uav.seed,
        available_count=3,
    )
    ok = uav.request_registration(
        domain_id="B",
        partial_shares=shares,
        threshold=3,
    )
    print("registration:", ok)
    if not ok:
        raise RuntimeError("UAV registration failed")

    # 5. B -> A 生成跨域 token，并安装到 AAS
    token = dca_b.generate_cross_domain_token(
        target_domain="A",
        available_count=3,
        validity_period=300,
    )
    if token is None:
        raise RuntimeError("cross-domain token generation failed")

    aas_a.install_cross_domain_token(source_domain="B", token=token)

    # 6. UAV 生成匿名跨域请求
    req = uav.generate_access_request(
        source_domain="B",
        target_domain="A",
        req_type="EDGE_SERVICE",
    )

    # 7. AAS 验证匿名请求
    result = aas_a.verify_access_request(
        request=req,
        source_dca=dca_b,
    )
    print("access verify:", result.success, result.reason)
    if not result.success:
        raise RuntimeError(f"Access failed: {result.reason}")

    # 8. 建立 AAS <-> ECS 内部信道
    challenge = ecs_a.issue_service_challenge(timestamp=req.timestamp)
    sk_ae = aas_a.establish_internal_channel_with_ecs(
        ecs_id=ecs_a.ecs_id,
        nonce_e=challenge.nonce_e,
    )
    ecs_a.establish_internal_channel(
        aas_id=aas_a.aas_id,
        nonce_a=int(sk_ae[:8], 16),   # 这里仅做演示，不要求强一致
        nonce_e=challenge.nonce_e,
    )

    # 9. UAV 响应 ECS 挑战，双方导出服务密钥
    nonce_u, ts_u = uav.respond_service_challenge(
        spid=req.spid,
        timestamp=challenge.timestamp,
    )

    sk_u = uav.derive_service_key(
        spid=req.spid,
        ecs_id=ecs_a.ecs_id,
        domain_id="A",
        nonce_e=challenge.nonce_e,
        nonce_u=nonce_u,
        challenge_timestamp=challenge.timestamp,
        context="EDGE_SERVICE",
    )

    sk_e = ecs_a.derive_service_key(
        uav_spid=req.spid,
        nonce_u=nonce_u,
        challenge=challenge,
        credential_hint=req.credential_public_key_b64,
        context="EDGE_SERVICE",
    )

    print("uav service key:", sk_u)
    print("ecs service key:", sk_e)

    # 10. 发送卸载任务并校验
    packet = uav.build_offload_packet(
        task_type="VIDEO_ANALYSIS",
        task_size=32,
    )
    verified = ecs_a.verify_offload_packet(packet, sk_e)
    print("offload verify:", verified)
    if not verified:
        raise RuntimeError("Offload packet verification failed")

    result = ecs_a.process_task(
        task_type=packet.task_type,
        task_size=packet.task_size,
    )
    print("task result:", result)


if __name__ == "__main__":
    main()