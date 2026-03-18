# core/ecdsa_utils.py
from __future__ import annotations

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature


def generate_ecdsa_keypair():
    """
    Generate an ECDSA private/public key pair using SECP256R1.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()
    return private_key, public_key


def sign_message(private_key, message: str) -> bytes:
    """
    Sign a UTF-8 string message with ECDSA(SHA256).
    """
    return private_key.sign(
        message.encode("utf-8"),
        ec.ECDSA(hashes.SHA256()),
    )


def verify_signature(public_key, message: str, signature: bytes) -> bool:
    """
    Verify ECDSA signature.
    """
    try:
        public_key.verify(
            signature,
            message.encode("utf-8"),
            ec.ECDSA(hashes.SHA256()),
        )
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
