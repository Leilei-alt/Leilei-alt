# core/bls_utils.py
from __future__ import annotations

from blspy import AugSchemeMPL, G1Element, G2Element, PrivateKey


def generate_bls_keypair(seed: bytes) -> tuple[PrivateKey, G1Element]:
    sk = AugSchemeMPL.key_gen(seed)
    pk = sk.get_g1()
    return sk, pk


def sign_message(sk: PrivateKey, message: bytes) -> G2Element:
    return AugSchemeMPL.sign(sk, message)


def verify_signature(pk: G1Element, message: bytes, signature: G2Element) -> bool:
    return AugSchemeMPL.verify(pk, message, signature)


def aggregate_signatures(signatures: list[G2Element]) -> G2Element:
    return AugSchemeMPL.aggregate(signatures)


def aggregate_verify(public_keys: list[G1Element], messages: list[bytes], agg_signature: G2Element) -> bool:
    return AugSchemeMPL.aggregate_verify(public_keys, messages, agg_signature)
