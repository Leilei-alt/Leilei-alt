# fastpai.py  —— 轻量 FastPai 适配层
import secrets, random, math
from dataclasses import dataclass

# 需要一个安全素数生成器；建议用 pycryptodome:
# from Crypto.Util.number import getPrime, isPrime
# 这里给出占位函数接口；实际项目请替换为你现有的素数生成工具。
def getPrime(bits):
    from Crypto.Util.number import getPrime as _getPrime
    return _getPrime(bits)

def isPrime(n):
    try:
        from Crypto.Util.number import isPrime as _isPrime
        return _isPrime(n)
    except Exception:
        # 兜底：快速 Miller-Rabin（略，建议直接装 pycryptodome）
        raise RuntimeError("请使用 Crypto.Util.number.isPrime 或你已有的素数库")

def _gen_PQ_like(p_bits, pprime_bits):
    """
    生成 P = 2*p*p' + 1 为素数；同理生成 Q。
    返回 (P, p, p')。
    """
    while True:
        p  = getPrime(p_bits)
        p_ = getPrime(pprime_bits) | 1  # 奇数
        P  = 2*p*p_ + 1
        if isPrime(P):
            return P, p, p_

@dataclass
class FastPaiPublicKey:
    n: int
    h: int
    scale: int = 10**6  # 给浮点数一个固定小数编码（与 phe 类似）

    def __post_init__(self):
        self.n_sq   = self.n * self.n
        self.max_int = self.n // 3  # 只要比 n 小很多即可安全

    def encrypt(self, m):
        # 与 phe 一样允许 float：做定点编码
        if isinstance(m, float):
            m = int(round(m * self.scale))
        else:
            m = int(m)
        c1 = pow(1 + self.n, m, self.n_sq)
        # r 可短一些；也可扩成 256+ bits
        r  = secrets.randbits(256)
        c2 = pow(pow(self.h, r, self.n), self.n, self.n_sq)  # (h^r)^N mod N^2
        return FastPaiEncrypted((c1 * c2) % self.n_sq, self)

class FastPaiEncrypted:
    def __init__(self, c, public_key: FastPaiPublicKey):
        self.c = int(c)
        self.public_key = public_key

    # 兼容你现在的 partial_decrypt 取原始密文整数
    def ciphertext(self, raw=False):
        return self.c

    # 加法：密文 ⊕ 密文 / 密文 ⊕ 明文
    def __add__(self, other):
        if isinstance(other, FastPaiEncrypted):
            return FastPaiEncrypted((self.c * other.c) % self.public_key.n_sq, self.public_key)
        elif isinstance(other, int):
            c_plain = pow(1 + self.public_key.n, other, self.public_key.n_sq)
            return FastPaiEncrypted((self.c * c_plain) % self.public_key.n_sq, self.public_key)
        else:
            raise TypeError(f"Unsupported add with {type(other)}")

    __radd__ = __add__

    # 减法：密文 ⊖ 密文 / 密文 ⊖ 明文
    def __sub__(self, other):
        if isinstance(other, FastPaiEncrypted):
            inv = pow(other.c, -1, self.public_key.n_sq)
            return FastPaiEncrypted((self.c * inv) % self.public_key.n_sq, self.public_key)
        elif isinstance(other, int):
            # self + (-other)
            return self + (-int(other))
        else:
            raise TypeError(f"Unsupported sub with {type(other)}")

    # 标量乘（明文 k）：对应 Enc(x)^k ；k 可为负（取逆）
    def __mul__(self, k):
        if not isinstance(k, int):
            raise TypeError("Only integer scalar is supported for ciphertext * k")
        if k == 0:
            # Enc(0) 的中性元是 1（乘法群单位元）
            return FastPaiEncrypted(1, self.public_key)
        if k > 0:
            return FastPaiEncrypted(pow(self.c, k, self.public_key.n_sq), self.public_key)
        # k < 0
        pos = pow(self.c, -k, self.public_key.n_sq)
        inv = pow(pos, -1, self.public_key.n_sq)
        return FastPaiEncrypted(inv, self.public_key)

    __rmul__ = __mul__


def fastpai_generate_threshold_keypair(kappa=112):
    """
    生成 FastPai 公私钥 + 阈值分片（share0 + share1 = 2*alpha）
    返回字典字段与原代码一致（mu 存 inv_2alpha，为少改调用点）
    """
    # 设 l(kappa)=4*kappa（与论文一致，这里只用于位长分配的直觉）
    # 经验做法：让 P、Q 的比特跟标准 Paillier 类似
    # 例如总 N 的比特 ~ 2048，则可取 p_bits ~ 256, pprime_bits ~ 512 等
    p_bits, pprime_bits = 256, 512

    P, p, p_ = _gen_PQ_like(p_bits, pprime_bits)
    Q, q, q_ = _gen_PQ_like(p_bits, pprime_bits)

    N = P * Q
    alpha = p * q
    beta  = p_ * q_

    # 选 y 并构造 h = - y^{2β} mod N
    # 需要 y ∈ Z*_N（与 N 互素）
    while True:
        y = random.randrange(2, N - 1)
        if math.gcd(y, N) == 1:
            break
    h = (-pow(y, 2 * beta, N)) % N

    # 私钥用于解密的指数：2*alpha
    two_alpha = 2 * alpha
    inv_2alpha = pow(two_alpha, -1, N)

    # 简单两份加法分片：share0 + share1 = 2*alpha
    share0 = random.randrange(1, two_alpha)
    share1 = (two_alpha - share0) % (two_alpha)  # 注意：合并用乘法，指数相加，无需模 two_alpha，这里保持整数
    # 为了与旧 combine_shares 取模 N 的逻辑兼容，我们不对 share1 取模 two_alpha，直接让 share0+share1=2α

    pubkey = FastPaiPublicKey(N, h)

    return {
        'pubkey': pubkey,
        # 占位 privkey，保持结构（你不再需要 p、q）
        'privkey': None,
        'share0': share0,
        'share1': two_alpha - share0,
        # 注意：mu 字段装的是 inv_2alpha（少改其它代码）
        'mu': inv_2alpha,
        'n': N,
        'n_sq': N * N
    }
