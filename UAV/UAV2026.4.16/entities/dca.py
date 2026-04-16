from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import List

from core.ecdsa_utils import (
    SECP256R1_ORDER,
    derive_private_key_from_scalar,
    generate_ecdsa_keypair,
    pem_to_b64,
    serialize_public_key_to_pem,
    sign_message,
    verify_signature,
)
from core.shamir import recover_secret, split_secret


@dataclass
class RegistrationCertificate:
    issuing_domain: str
    credential_public_key_b64: str
    delta_digest: str
    issued_at: int
    validity_until: int
    signature: bytes

    def body_to_sign(self) -> str:
        return (
            f"REGCERT|{self.issuing_domain}|{self.credential_public_key_b64}|"
            f"{self.delta_digest}|{self.issued_at}|{self.validity_until}"
        )


@dataclass
class PartialRegistrationShare:
    share_x: int
    partial_scalar: int
    certificate: RegistrationCertificate


@dataclass
class DCAMember:
    member_id: int
    share_x: int
    share_y: int

    def issue_registration_share(self, uav_seed: int, modulus: int) -> int:
        return (self.share_y * uav_seed) % modulus

    def issue_cross_domain_share(self, target_domain_seed: int, delta_value: int, modulus: int) -> int:
        return (self.share_y * target_domain_seed * delta_value) % modulus


@dataclass
class CrossDomainToken:
    source_domain: str
    target_domain: str
    validity_until: int
    delta_digest: str
    token_value: int
    signature: bytes

    def body_to_sign(self) -> str:
        return (
            f"XDOM|{self.source_domain}|{self.target_domain}|{self.validity_until}|"
            f"{self.delta_digest}|{self.token_value}"
        )

    def summary(self) -> str:
        return (
            f"CrossDomainToken(source={self.source_domain}, target={self.target_domain}, "
            f"validity_until={self.validity_until}, delta_digest={self.delta_digest[:16]}..., "
            f"token_value={self.token_value})"
        )


class DCACluster:
    def __init__(self, domain_id: str, n: int, t: int, prime: int):
        self.domain_id = domain_id
        self.n = n
        self.t = t
        self.prime = prime
        self.scalar_modulus = SECP256R1_ORDER

        self.domain_secret = random.randrange(1, self.scalar_modulus)
        self.delta_value = random.randrange(1, self.scalar_modulus)
        self.domain_signing_private_key, self.domain_signing_public_key = generate_ecdsa_keypair()
        self.members: List[DCAMember] = []
        self._setup_members()

    def _setup_members(self) -> None:
        shares = split_secret(self.domain_secret, self.n, self.t, self.scalar_modulus)
        for idx, (x, y) in enumerate(shares, start=1):
            self.members.append(DCAMember(member_id=idx, share_x=x, share_y=y))

    def get_available_members(self, count: int) -> List[DCAMember]:
        return self.members[:count]

    def build_target_domain_seed(self, target_domain: str) -> int:
        raw = f"{self.domain_id}->{target_domain}"
        return int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16) % self.scalar_modulus or 1

    def delta_digest(self) -> str:
        return hashlib.sha256(str(self.delta_value).encode("utf-8")).hexdigest()

    def issue_registration_shares(self, uav_seed: int, available_count: int) -> List[PartialRegistrationShare]:
        if available_count < self.t:
            raise ValueError("Need at least threshold DCA members to issue registration shares.")

        available_members = self.get_available_members(available_count)
        partials = [m.issue_registration_share(uav_seed, self.scalar_modulus) for m in available_members]

        # Threshold-issued anonymous credential scalar alpha = s * seed mod q.
        credential_scalar = (self.domain_secret * (uav_seed % self.scalar_modulus)) % self.scalar_modulus
        credential_private_key, credential_public_key = derive_private_key_from_scalar(credential_scalar)
        credential_public_b64 = pem_to_b64(serialize_public_key_to_pem(credential_public_key))
        del credential_private_key

        now_ts = int(time.time())
        cert = RegistrationCertificate(
            issuing_domain=self.domain_id,
            credential_public_key_b64=credential_public_b64,
            delta_digest=self.delta_digest(),
            issued_at=now_ts,
            validity_until=now_ts + 86400,
            signature=b"",
        )
        cert.signature = sign_message(self.domain_signing_private_key, cert.body_to_sign())

        return [
            PartialRegistrationShare(
                share_x=member.share_x,
                partial_scalar=partials[idx],
                certificate=cert,
            )
            for idx, member in enumerate(available_members)
        ]

    def generate_cross_domain_token(
        self,
        target_domain: str,
        available_count: int,
        validity_period: int = 300,
    ) -> CrossDomainToken | None:
        if available_count < self.t:
            return None

        target_domain_seed = self.build_target_domain_seed(target_domain)
        available_members = self.get_available_members(available_count)
        partial_shares = [
            (member.share_x, member.issue_cross_domain_share(target_domain_seed, self.delta_value, self.scalar_modulus))
            for member in available_members
        ]
        token_value = recover_secret(partial_shares[: self.t], self.scalar_modulus)

        token = CrossDomainToken(
            source_domain=self.domain_id,
            target_domain=target_domain,
            validity_until=int(time.time()) + validity_period,
            delta_digest=self.delta_digest(),
            token_value=token_value,
            signature=b"",
        )
        token.signature = sign_message(self.domain_signing_private_key, token.body_to_sign())
        return token

    def verify_registration_certificate(self, certificate: RegistrationCertificate, now_ts: int | None = None) -> bool:
        if now_ts is None:
            now_ts = int(time.time())
        if certificate.issuing_domain != self.domain_id:
            return False
        if now_ts > certificate.validity_until:
            return False
        if certificate.delta_digest != self.delta_digest():
            return False
        return verify_signature(self.domain_signing_public_key, certificate.body_to_sign(), certificate.signature)

    def verify_cross_domain_token(
        self,
        token: CrossDomainToken,
        source_dca: "DCACluster",
        now_ts: int | None = None,
    ) -> bool:
        if now_ts is None:
            now_ts = int(time.time())
        if token.source_domain != source_dca.domain_id:
            return False
        if token.target_domain != self.domain_id:
            return False
        if now_ts > token.validity_until:
            return False
        if token.delta_digest != source_dca.delta_digest():
            return False
        expected_seed = source_dca.build_target_domain_seed(self.domain_id)
        expected_value = (source_dca.domain_secret * expected_seed * source_dca.delta_value) % self.scalar_modulus
        if expected_value != token.token_value:
            return False
        return verify_signature(source_dca.domain_signing_public_key, token.body_to_sign(), token.signature)
