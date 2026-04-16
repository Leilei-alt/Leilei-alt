from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Dict

from core.bls_utils import aggregate_signatures, aggregate_verify
from core.bls_utils import verify_signature as bls_verify_signature
from core.ecdsa_utils import b64_to_pem, load_public_key_from_pem
from entities.dca import CrossDomainToken, DCACluster, RegistrationCertificate
from entities.uav import AccessRequest, BLSAccessRequest, verify_uav_signature


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

    def _verify_registration_certificate(self, request: AccessRequest, source_dca: DCACluster, now_ts: int) -> bool:
        cert = RegistrationCertificate(
            issuing_domain=request.source_domain,
            credential_public_key_b64=request.credential_public_key_b64,
            delta_digest=request.cert_delta_digest,
            issued_at=request.cert_issued_at,
            validity_until=request.cert_validity_until,
            signature=bytes.fromhex(request.certificate_signature_hex),
        )
        return source_dca.verify_registration_certificate(cert, now_ts=now_ts)

    def verify_access_request(
        self,
        request: AccessRequest,
        source_dca: DCACluster,
        uav_public_key=None,
        now_ts: int | None = None,
    ) -> AccessResult:
        del uav_public_key
        if now_ts is None:
            now_ts = int(time.time())

        if request.target_domain != self.domain_id:
            return AccessResult(False, "Wrong target domain")
        if abs(now_ts - request.timestamp) > self.replay_window:
            return AccessResult(False, "Timestamp expired or replayed")

        token = self.cross_domain_tokens.get(request.source_domain)
        if token is None:
            return AccessResult(False, "No cross-domain token installed")
        if not self.dca_cluster.verify_cross_domain_token(token=token, source_dca=source_dca, now_ts=now_ts):
            return AccessResult(False, "Cross-domain token invalid")

        expected_pid = hashlib.sha256(
            f"{request.source_domain}|{request.credential_public_key_b64}".encode("utf-8")
        ).hexdigest()[:20]
        if request.pseudo_id != f"PID-{expected_pid}":
            return AccessResult(False, "Pseudo ID does not match certified anonymous credential")

        if not self._verify_registration_certificate(request, source_dca, now_ts):
            return AccessResult(False, "Anonymous credential certificate invalid")

        try:
            _ = load_public_key_from_pem(b64_to_pem(request.credential_public_key_b64))
        except Exception:
            return AccessResult(False, "Invalid anonymous credential public key")

        if not verify_uav_signature(request, None):
            return AccessResult(False, "ECC anonymous request signature invalid")

        # Lightweight check: ephemeral session handle must be well-formed.
        try:
            _ = load_public_key_from_pem(b64_to_pem(request.ephemeral_public_key_b64))
        except Exception:
            return AccessResult(False, "Invalid ephemeral ECC handle")

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
        nonce_a = random.randrange(1, self.dca_cluster.scalar_modulus)
        raw = f"SK_AE|{self.aas_id}|{ecs_id}|{self.domain_id}|{nonce_a}|{nonce_e}"
        self.current_sk_ae = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.current_sk_ae

    def verify_bls_access_request(
        self,
        request: BLSAccessRequest,
        source_dca: DCACluster,
        uav_bls_public_key,
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
        if not self.dca_cluster.verify_cross_domain_token(token=token, source_dca=source_dca, now_ts=now_ts):
            return AccessResult(False, "Cross-domain token invalid")

        if not bls_verify_signature(uav_bls_public_key, request.message_to_sign(), request.signature):
            return AccessResult(False, "BLS signature invalid")
        return AccessResult(True, "Access granted")

    def batch_verify_bls_access_requests(
        self,
        requests: list[BLSAccessRequest],
        source_dca: DCACluster,
        public_keys: list,
        now_ts: int | None = None,
    ) -> AccessResult:
        if now_ts is None:
            now_ts = int(time.time())
        if len(requests) != len(public_keys):
            return AccessResult(False, "Request/public-key size mismatch")

        for req in requests:
            if req.target_domain != self.domain_id:
                return AccessResult(False, "Wrong target domain in batch")
            if abs(now_ts - req.timestamp) > self.replay_window:
                return AccessResult(False, "Timestamp expired or replayed in batch")
            token = self.cross_domain_tokens.get(req.source_domain)
            if token is None:
                return AccessResult(False, "Missing cross-domain token in batch")
            if not self.dca_cluster.verify_cross_domain_token(token=token, source_dca=source_dca, now_ts=now_ts):
                return AccessResult(False, "Invalid cross-domain token in batch")

        messages = [req.message_to_sign() for req in requests]
        agg_sig = aggregate_signatures([req.signature for req in requests])
        if not aggregate_verify(public_keys, messages, agg_sig):
            return AccessResult(False, "Aggregate BLS verification failed")
        return AccessResult(True, "Batch access granted")
