# test_bls_minimal.py
from blspy import (PrivateKey, AugSchemeMPL, G1Element, G2Element)

def main():
    # 1. keygen
    seed1 = bytes([1] * 32)
    seed2 = bytes([2] * 32)

    sk1 = AugSchemeMPL.key_gen(seed1)
    sk2 = AugSchemeMPL.key_gen(seed2)

    pk1 = sk1.get_g1()
    pk2 = sk2.get_g1()

    print("PK1:", pk1)
    print("PK2:", pk2)

    # 2. single sign / verify
    msg1 = b"uav-access-request-1"
    sig1 = AugSchemeMPL.sign(sk1, msg1)

    ok1 = AugSchemeMPL.verify(pk1, msg1, sig1)
    print("Single verify:", ok1)

    # 3. aggregate sign / verify
    msg2 = b"uav-access-request-2"
    sig2 = AugSchemeMPL.sign(sk2, msg2)

    agg_sig = AugSchemeMPL.aggregate([sig1, sig2])

    ok_agg = AugSchemeMPL.aggregate_verify(
        [pk1, pk2],
        [msg1, msg2],
        agg_sig
    )
    print("Aggregate verify:", ok_agg)


if __name__ == "__main__":
    main()
