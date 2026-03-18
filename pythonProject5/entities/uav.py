# entities/uav.py
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

from core.shamir import recover_secret


def hash_to_int(data: str, prime: int) -> int:
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return int(digest, 16) % prime


@dataclass
class RegistrationCredential:
    pseudo_id: str
    seed: int
    credential_value: int


@dataclass
class AccessRequest:
    source_domain: str
    target_domain: str
    pseudo_id: str
    spid: str
    req_type: str
    timestamp: int
    signature: int

    def message_to_sign(self) -> str:
        return (
            f"{self.source_domain}|{self.target_domain}|{self.pseudo_id}|"
            f"{self.spid}|{self.req_type}|{self.timestamp}"
        )


def sign_message(message: str, private_key: int, prime: int) -> int:
    return hash_to_int(message + f"|{private_key}", prime)


def verify_signature(message: str, signature: int, public_key: int, prime: int) -> bool:
    expected = hash_to_int(message + f"|{public_key}", prime)
    return expected == signature


def verify_uav_signature(request: AccessRequest, uav_public_key: int) -> bool:
    prime = 208351617316091241234326746312124448251235562226470491514186331217050270460481
    return verify_signature(
        message=request.message_to_sign(),
        signature=request.signature,
        public_key=uav_public_key,
        prime=prime,
    )


def kdf(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class UAV:
    def __init__(self, uav_id: str, prime: int):
        self.uav_id = uav_id
        self.prime = prime

        self.seed = random.randrange(1, prime)
        self.private_key = random.randrange(1, prime)
        self.public_key = self.private_key  # prototype simplification

        self.credential: RegistrationCredential | None = None
        self.current_service_key: str | None = None

    def build_pseudo_id(self, domain_id: str) -> str:
        raw = f"{self.uav_id}|{domain_id}|{self.seed}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"PID-{digest}"

    def request_registration(self, domain_id: str, partial_shares: List[Tuple[int, int]], threshold: int) -> bool:
        if len(partial_shares) < threshold:
            return False

        recovered_value = recover_secret(partial_shares[:threshold], self.prime)
        pseudo_id = self.build_pseudo_id(domain_id)

        self.credential = RegistrationCredential(
            pseudo_id=pseudo_id,
            seed=self.seed,
            credential_value=recovered_value,
        )
        return True

    def build_session_pseudonym(self, target_domain: str, timestamp: int | None = None) -> str:
        if timestamp is None:
            timestamp = int(time.time())
        random_nonce = random.randrange(1, self.prime)
        raw = f"{self.uav_id}|{target_domain}|{self.seed}|{random_nonce}|{timestamp}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
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

        temp_request = AccessRequest(
            source_domain=source_domain,
            target_domain=target_domain,
            pseudo_id=self.credential.pseudo_id,
            spid=spid,
            req_type=req_type,
            timestamp=timestamp,
            signature=0,
        )

        signature = sign_message(
            message=temp_request.message_to_sign(),
            private_key=self.private_key,
            prime=self.prime,
        )

        temp_request.signature = signature
        return temp_request

    def respond_service_challenge(self, spid: str, timestamp: int) -> tuple[int, int]:
        nonce_u = random.randrange(1, self.prime)
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
        self.current_service_key = kdf(
            "SK_UE",
            spid,
            domain_id,
            ecs_id,
            str(nonce_e),
            str(nonce_u),
            str(challenge_timestamp),
            context,
        )
        return self.current_service_key

    def build_offload_packet(self, task_type: str, task_size: int):
        if self.current_service_key is None:
            raise ValueError("No service key established.")

        payload = f"{self.uav_id}|{task_type}|{task_size}"
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
            f"seed={self.credential.seed}, "
            f"credential_value={self.credential.credential_value}"
        )
