# entities/aas.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict

from entities.dca import DCACluster, CrossDomainToken
from entities.uav import AccessRequest, verify_uav_signature


@dataclass
class AccessResult:
    success: bool
    reason: str


@dataclass
class AuthSessionContext:
    source_domain: str
    target_domain: str
    pseudo_id: str
    spid: str
    req_type: str
    timestamp: int
    authorized: bool

    def summary(self) -> str:
        return (
            f"AuthSessionContext(source={self.source_domain}, target={self.target_domain}, "
            f"pseudo_id={self.pseudo_id}, spid={self.spid}, req_type={self.req_type}, "
            f"timestamp={self.timestamp}, authorized={self.authorized})"
        )


class AAS:
    def __init__(self, domain_id: str, dca_cluster: DCACluster, replay_window: int = 10):
        self.domain_id = domain_id
        self.dca_cluster = dca_cluster
        self.replay_window = replay_window
        self.cross_domain_tokens: Dict[str, CrossDomainToken] = {}
        self.current_sk_ae: str | None = None
        self.aas_id = f"AAS-{domain_id}"

    def install_cross_domain_token(self, source_domain: str, token: CrossDomainToken) -> None:
        self.cross_domain_tokens[source_domain] = token

    def verify_access_request(
        self,
        request: AccessRequest,
        source_dca: DCACluster,
        uav_public_key: int,
        now_ts: int | None = None,
    ) -> AccessResult:
        if now_ts is None:
            now_ts = int(time.time())

        if request.target_domain != self.domain_id:
            return AccessResult(False, "Wrong target domain")

        if abs(now_ts - request.timestamp) > self.replay_window:
            return AccessResult(False, "Timestamp expired or replayed")

        token = self.cross_domain_tokens.get(request.source_domain)
        if token is None:
            return AccessResult(False, "No cross-domain token installed")

        valid_token = self.dca_cluster.verify_cross_domain_token(
            token=token,
            source_dca=source_dca,
            now_ts=now_ts,
        )
        if not valid_token:
            return AccessResult(False, "Cross-domain token invalid")

        valid_sig = verify_uav_signature(request, uav_public_key)
        if not valid_sig:
            return AccessResult(False, "UAV signature invalid")

        return AccessResult(True, "Access granted")

    def build_auth_session_context(self, request: AccessRequest) -> AuthSessionContext:
        return AuthSessionContext(
            source_domain=request.source_domain,
            target_domain=request.target_domain,
            pseudo_id=request.pseudo_id,
            spid=request.spid,
            req_type=request.req_type,
            timestamp=request.timestamp,
            authorized=True,
        )

    def establish_internal_channel_with_ecs(self, ecs_id: str, nonce_e: int) -> str:
        nonce_a = random.randrange(1, self.dca_cluster.prime)
        import hashlib
        raw = f"SK_AE|{self.aas_id}|{ecs_id}|{self.domain_id}|{nonce_a}|{nonce_e}"
        self.current_sk_ae = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.current_sk_ae
