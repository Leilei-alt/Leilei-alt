from __future__ import annotations

try:
    from blspy import AugSchemeMPL, G1Element, G2Element, PrivateKey  # type: ignore
    BLS_AVAILABLE = True
except Exception:  # pragma: no cover
    AugSchemeMPL = G1Element = G2Element = PrivateKey = object
    BLS_AVAILABLE = False


def _need_bls() -> None:
    if not BLS_AVAILABLE:
        raise RuntimeError(
            "BLS batch verification requires the 'blspy' package. "
            "Install it first, or keep using the ECC anonymous main path only."
        )


def generate_bls_keypair(seed: bytes):
    _need_bls()
    sk = AugSchemeMPL.key_gen(seed)
    pk = sk.get_g1()
    return sk, pk


def sign_message(sk, message: bytes):
    _need_bls()
    return AugSchemeMPL.sign(sk, message)


def verify_signature(pk, message: bytes, signature) -> bool:
    _need_bls()
    return AugSchemeMPL.verify(pk, message, signature)


def aggregate_signatures(signatures: list):
    _need_bls()
    return AugSchemeMPL.aggregate(signatures)


def aggregate_verify(public_keys: list, messages: list[bytes], agg_signature) -> bool:
    _need_bls()
    return AugSchemeMPL.aggregate_verify(public_keys, messages, agg_signature)
