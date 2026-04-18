from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import List

from typing import Any

from core.bls_utils import BLS_AVAILABLE, generate_bls_keypair as generate_bls_keypair_real
from core.bls_utils import sign_message as bls_sign_message
from core.ecdsa_utils import (
    SECP256R1_ORDER,
    b64_to_pem,
    derive_private_key_from_scalar,
    generate_ecdsa_keypair,
    load_public_key_from_pem,
    pem_to_b64,
    public_key_fingerprint,
    serialize_public_key_to_pem,
    sign_message,
    verify_signature,
)
from core.shamir import recover_secret
from entities.dca import PartialRegistrationShare, RegistrationCertificate


@dataclass
class BLSAccessRequest:
    source_domain: str
    target_domain: str
    pseudo_id: str
    spid: str
    req_type: str
    timestamp: int
    signature: Any

    def message_to_sign(self) -> bytes:
        msg = (
            f"{self.source_domain}|{self.target_domain}|{self.pseudo_id}|"
            f"{self.spid}|{self.req_type}|{self.timestamp}"
        )
        return msg.encode("utf-8")


@dataclass
class RegistrationCredential:
    pseudo_id: str
    credential_scalar: int
    credential_public_key_b64: str
    certificate: RegistrationCertificate


@dataclass
class AccessRequest:
    source_domain: str
    target_domain: str
    pseudo_id: str
    spid: str
    req_type: str
    timestamp: int
    credential_public_key_b64: str
    cert_delta_digest: str
    cert_issued_at: int
    cert_validity_until: int
    certificate_signature_hex: str
    ephemeral_public_key_b64: str
    signature: bytes

    def message_to_sign(self) -> str:
        return (
            f"{self.source_domain}|{self.target_domain}|{self.pseudo_id}|{self.spid}|{self.req_type}|"
            f"{self.timestamp}|{self.credential_public_key_b64}|{self.cert_delta_digest}|"
            f"{self.cert_issued_at}|{self.cert_validity_until}|{self.certificate_signature_hex}|"
            f"{self.ephemeral_public_key_b64}"
        )


def kdf(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_uav_signature(request: AccessRequest, _unused=None) -> bool:
    public_key = load_public_key_from_pem(b64_to_pem(request.credential_public_key_b64))
    return verify_signature(public_key=public_key, message=request.message_to_sign(), signature=request.signature)


class UAV:
    def __init__(self, uav_id: str, prime: int):
        self.uav_id = uav_id
        self.prime = prime
        self.seed = random.randrange(1, SECP256R1_ORDER)

        # Real identity key: kept local only, no longer used as the main cross-domain auth key.
        self.private_key, self.public_key = generate_ecdsa_keypair()

        self.credential: RegistrationCredential | None = None
        self.current_service_key: str | None = None

        seed_material = hashlib.sha256(f"BLS|{uav_id}|{self.seed}".encode("utf-8")).digest()
        if BLS_AVAILABLE:
            self.bls_private_key, self.bls_public_key = generate_bls_keypair_real(seed_material)
        else:
            self.bls_private_key, self.bls_public_key = None, None

    def build_pseudo_id(self, domain_id: str, credential_public_key_b64: str) -> str:
        raw = f"{domain_id}|{credential_public_key_b64}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        return f"PID-{digest}"

    def request_registration(self, domain_id: str, partial_shares: List[PartialRegistrationShare], threshold: int) -> bool:
        if len(partial_shares) < threshold:
            return False

        points = [(share.share_x, share.partial_scalar) for share in partial_shares[:threshold]]
        recovered_scalar = recover_secret(points, SECP256R1_ORDER)
        credential_private_key, credential_public_key = derive_private_key_from_scalar(recovered_scalar)
        credential_public_b64 = pem_to_b64(serialize_public_key_to_pem(credential_public_key))

        cert = partial_shares[0].certificate
        if cert.issuing_domain != domain_id:
            return False
        if cert.credential_public_key_b64 != credential_public_b64:
            return False

        cert_public_key = load_public_key_from_pem(b64_to_pem(cert.credential_public_key_b64))
        # Sanity check that the certified public key is usable.
        _ = public_key_fingerprint(cert_public_key)
        del cert_public_key
        del credential_private_key

        pseudo_id = self.build_pseudo_id(domain_id, credential_public_b64)
        self.credential = RegistrationCredential(
            pseudo_id=pseudo_id,
            credential_scalar=recovered_scalar,
            credential_public_key_b64=credential_public_b64,
            certificate=cert,
        )
        return True

    def build_session_pseudonym(self, target_domain: str, timestamp: int | None = None) -> str:
        if self.credential is None:
            raise ValueError("UAV has no registration credential. Register first.")
        if timestamp is None:
            timestamp = int(time.time())
        random_nonce = random.randrange(1, SECP256R1_ORDER)
        raw = f"{self.credential.pseudo_id}|{target_domain}|{random_nonce}|{timestamp}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
        return f"SPID-{digest}"

    def generate_access_request(
        self,
        source_domain: str,
        target_domain: str,
        req_type: str = "EDGE_SERVICE",
        timestamp: int | None = None,
    ) -> AccessRequest:
        if self.credential is None:
            raise ValueError("UAV has no registration credential. Register first.")
        if timestamp is None:
            timestamp = int(time.time())

        spid = self.build_session_pseudonym(target_domain=target_domain, timestamp=timestamp)
        eph_private, eph_public = generate_ecdsa_keypair()
        eph_public_b64 = pem_to_b64(serialize_public_key_to_pem(eph_public))
        del eph_private

        temp_request = AccessRequest(
            source_domain=source_domain,
            target_domain=target_domain,
            pseudo_id=self.credential.pseudo_id,
            spid=spid,
            req_type=req_type,
            timestamp=timestamp,
            credential_public_key_b64=self.credential.credential_public_key_b64,
            cert_delta_digest=self.credential.certificate.delta_digest,
            cert_issued_at=self.credential.certificate.issued_at,
            cert_validity_until=self.credential.certificate.validity_until,
            certificate_signature_hex=self.credential.certificate.signature.hex(),
            ephemeral_public_key_b64=eph_public_b64,
            signature=b"",
        )

        credential_private_key, _ = derive_private_key_from_scalar(self.credential.credential_scalar)
        temp_request.signature = sign_message(credential_private_key, temp_request.message_to_sign())
        return temp_request

    def respond_service_challenge(self, spid: str, timestamp: int) -> tuple[int, int]:
        nonce_u = random.randrange(1, SECP256R1_ORDER)
        return nonce_u, timestamp

    def derive_service_key(
        self,
        spid: str,
        ecs_id: str,
        domain_id: str,
        nonce_e: int,
        nonce_u: int,
        challenge_timestamp: int,
        context: str = "EDGE_SERVICE",
    ) -> str:
        if self.credential is None:
            raise ValueError("No anonymous registration credential.")
        self.current_service_key = kdf(
            "SK_UE",
            spid,
            domain_id,
            ecs_id,
            str(nonce_e),
            str(nonce_u),
            str(challenge_timestamp),
            self.credential.credential_public_key_b64,
            context,
        )
        return self.current_service_key

    def build_offload_packet(self, task_type: str, task_size: int):
        if self.current_service_key is None:
            raise ValueError("No service key established.")

        payload = f"{self.credential.pseudo_id if self.credential else self.uav_id}|{task_type}|{task_size}"
        payload_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        auth_tag = hashlib.sha256(
            f"{self.current_service_key}|{task_type}|{task_size}|{payload_digest}".encode("utf-8")
        ).hexdigest()

        from entities.ecs import OffloadPacket

        return OffloadPacket(
            task_type=task_type,
            task_size=task_size,
            payload_digest=payload_digest,
            auth_tag=auth_tag,
        )

    def process_task_locally(self, task_type: str, task_size: int) -> dict:
        if task_type == "IMAGE_CLASSIFICATION":
            simulated_delay = 0.004 * task_size
        elif task_type == "PATH_PLANNING":
            simulated_delay = 0.003 * task_size
        elif task_type == "VIDEO_ANALYSIS":
            simulated_delay = 0.006 * task_size
        else:
            simulated_delay = 0.002 * task_size

        start = time.perf_counter()
        time.sleep(simulated_delay)
        end = time.perf_counter()

        return {
            "task_type": task_type,
            "task_size": task_size,
            "simulated_delay": simulated_delay,
            "measured_processing_time": end - start,
            "result": f"local_processed:{task_type}:{task_size}",
        }

    def show_credential(self) -> str:
        if self.credential is None:
            return "No credential"
        return (
            f"pseudo_id={self.credential.pseudo_id}, "
            f"credential_public_key={self.credential.credential_public_key_b64[:20]}..., "
            f"issued_by={self.credential.certificate.issuing_domain}"
        )

    def generate_bls_access_request(
        self,
        source_domain: str,
        target_domain: str,
        req_type: str = "EDGE_SERVICE",
        timestamp: int | None = None,
    ) -> BLSAccessRequest:
        if self.credential is None:
            raise ValueError("UAV has no registration credential. Register first.")
        if self.bls_private_key is None:
            raise RuntimeError("BLS batch path is unavailable because 'blspy' is not installed.")
        if timestamp is None:
            timestamp = int(time.time())

        spid = self.build_session_pseudonym(target_domain=target_domain, timestamp=timestamp)
        temp_request = BLSAccessRequest(
            source_domain=source_domain,
            target_domain=target_domain,
            pseudo_id=self.credential.pseudo_id,
            spid=spid,
            req_type=req_type,
            timestamp=timestamp,
            signature=None,
        )
        temp_request.signature = bls_sign_message(self.bls_private_key, temp_request.message_to_sign())
        return temp_request
