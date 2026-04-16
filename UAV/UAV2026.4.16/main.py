from entities.dca import DCACluster
from entities.uav import UAV
from entities.aas import AAS
from core.pairing_utils import *
import os
os.environ["MCLBN256_NO_CLEANUP"] = "1"


def main():
    print("=== G1 x G2 Pairing Demo ===")

    dca = DCACluster()
    uav = UAV()
    aas = AAS()

    # 注册
    uav.register(dca)

    # 构造跨域参数（G2！！）
    seed = random_scalar()

    Fw = g2_mul(Q, seed)
    Fw_inv = g2_mul(Fw, random_scalar())

    # 请求
    req = uav.generate_request()

    # 验证
    ok = aas.verify(req, Fw, Fw_inv)

    print("Verification:", ok)


if __name__ == "__main__":
    main()