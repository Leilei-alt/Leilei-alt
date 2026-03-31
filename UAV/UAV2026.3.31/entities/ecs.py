# entities/ecs.py
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass


def kdf(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ServiceChallenge:
    ecs_id: str
    nonce_e: int
    timestamp: int

    def summary(self) -> str:
        return f"ServiceChallenge(ecs_id={self.ecs_id}, nonce_e={self.nonce_e}, timestamp={self.timestamp})"


@dataclass
class ServiceResponse:
    uav_spid: str
    nonce_u: int
    timestamp: int

    def summary(self) -> str:
        return f"ServiceResponse(uav_spid={self.uav_spid}, nonce_u={self.nonce_u}, timestamp={self.timestamp})"


@dataclass
class OffloadPacket:
    task_type: str
    task_size: int
    payload_digest: str
    auth_tag: str

    def summary(self) -> str:
        return (
            f"OffloadPacket(task_type={self.task_type}, "
            f"task_size={self.task_size}, "
            f"payload_digest={self.payload_digest[:16]}..., "
            f"auth_tag={self.auth_tag[:16]}...)"
        )


class ECS:
    def __init__(self, ecs_id: str, domain_id: str, prime: int):
        self.ecs_id = ecs_id
        self.domain_id = domain_id
        self.prime = prime

        self.local_secret = random.randrange(1, prime)
        self.current_sk_ae: str | None = None
        self.current_service_key: str | None = None

    def establish_internal_channel(self, aas_id: str, nonce_a: int, nonce_e: int) -> str:
        self.current_sk_ae = kdf(
            "SK_AE",
            aas_id,
            self.ecs_id,
            self.domain_id,
            str(nonce_a),
            str(nonce_e),
        )
        return self.current_sk_ae

    def issue_service_challenge(self, timestamp: int) -> ServiceChallenge:
        nonce_e = random.randrange(1, self.prime)
        return ServiceChallenge(
            ecs_id=self.ecs_id,
            nonce_e=nonce_e,
            timestamp=timestamp,
        )

    def derive_service_key(
        self,
        uav_spid: str,
        nonce_u: int,
        challenge: ServiceChallenge,
        context: str = "EDGE_SERVICE",
    ) -> str:
        self.current_service_key = kdf(
            "SK_UE",
            uav_spid,
            self.domain_id,
            challenge.ecs_id,
            str(challenge.nonce_e),
            str(nonce_u),
            str(challenge.timestamp),
            context,
        )
        return self.current_service_key

    def verify_offload_packet(self, packet: OffloadPacket, service_key: str) -> bool:
        expected_tag = hashlib.sha256(
            f"{service_key}|{packet.task_type}|{packet.task_size}|{packet.payload_digest}".encode("utf-8")
        ).hexdigest()
        return expected_tag == packet.auth_tag

    def process_task(self, task_type: str, task_size: int) -> dict:
        """
        Simulate task processing cost.
        task_size is an abstract size unit.
        """
        if task_type == "IMAGE_CLASSIFICATION":
            simulated_delay = 0.002 * task_size
        elif task_type == "PATH_PLANNING":
            simulated_delay = 0.0015 * task_size
        elif task_type == "VIDEO_ANALYSIS":
            simulated_delay = 0.003 * task_size
        else:
            simulated_delay = 0.001 * task_size

        start = time.perf_counter()
        time.sleep(simulated_delay)
        end = time.perf_counter()

        return {
            "task_type": task_type,
            "task_size": task_size,
            "simulated_delay": simulated_delay,
            "measured_processing_time": end - start,
            "result": f"processed:{task_type}:{task_size}",
        }
