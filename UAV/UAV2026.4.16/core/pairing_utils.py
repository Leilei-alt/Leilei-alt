from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

# 这里先用抽象接口名，具体库你后面再替换
# 例如 charm / pypbc / petrelic / 自己封装的 pairing backend

@dataclass
class PairingParams:
    order: int
    P: object   # G1 generator


def setup_pairing() -> PairingParams:
    """
    初始化 pairing 参数。
    返回群阶 order 和生成元 P。
    """
    raise NotImplementedError


def rand_scalar(order: int) -> int:
    return 1 + secrets.randbelow(order - 1)


def hash_to_scalar(text: str, order: int) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % order


def scalar_mul(point, scalar: int):
    raise NotImplementedError


def point_add(p1, p2):
    raise NotImplementedError


def point_eq(p1, p2) -> bool:
    raise NotImplementedError


def pairing(p1, p2):
    raise NotImplementedError


def gt_eq(x, y) -> bool:
    return x == y


def serialize_point(point) -> bytes:
    raise NotImplementedError