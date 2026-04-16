from __future__ import annotations

import base64
import hashlib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

# Order of NIST P-256 / SECP256R1.
SECP256R1_ORDER = int(
    "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
    16,
)


def _to_bytes(message: str | bytes) -> bytes:
    if isinstance(message, bytes):
        return message
    return message.encode("utf-8")


def hash_to_int(*parts: str | bytes, modulus: int = SECP256R1_ORDER) -> int:
    h = hashlib.sha256()
    for part in parts:
        h.update(_to_bytes(part))
        h.update(b"|")
    value = int.from_bytes(h.digest(), "big") % modulus
    return value if value != 0 else 1


def generate_ecdsa_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    return private_key, private_key.public_key()


def derive_private_key_from_scalar(scalar: int):
    scalar = scalar % SECP256R1_ORDER
    if scalar == 0:
        scalar = 1
    private_key = ec.derive_private_key(scalar, ec.SECP256R1())
    return private_key, private_key.public_key()


def sign_message(private_key, message: str | bytes) -> bytes:
    return private_key.sign(_to_bytes(message), ec.ECDSA(hashes.SHA256()))


def verify_signature(public_key, message: str | bytes, signature: bytes) -> bool:
    try:
        public_key.verify(signature, _to_bytes(message), ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def serialize_public_key_to_pem(public_key) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def load_public_key_from_pem(pem_data: bytes):
    return serialization.load_pem_public_key(pem_data)


def pem_to_b64(pem_data: bytes) -> str:
    return base64.b64encode(pem_data).decode("ascii")


def b64_to_pem(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def public_key_fingerprint(public_key) -> str:
    return hashlib.sha256(serialize_public_key_to_pem(public_key)).hexdigest()


def derive_shared_secret_hex(private_key, peer_public_key) -> str:
    shared = private_key.exchange(ec.ECDH(), peer_public_key)
    return hashlib.sha256(shared).hexdigest()
