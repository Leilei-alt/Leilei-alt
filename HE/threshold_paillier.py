from phe import paillier
import random


# ---------------------------
# 阈值密钥生成（share0 + share1 = λ）
# ---------------------------
def generate_threshold_keypair():
    pubkey, privkey = paillier.generate_paillier_keypair()
    n = pubkey.n
    n_sq = n * n

    # 通过公开API计算lambda和mu
    p, q = privkey.p, privkey.q
    lambda_val = (p - 1) * (q - 1)
    mu_val = pow(lambda_val, -1, n)  # μ = λ⁻¹ mod n

    # 简单线性分片：λ = share0 + share1
    share0 = random.randint(1, lambda_val - 1)
    share1 = lambda_val - share0

    return {
        'pubkey': pubkey,
        'privkey': privkey,
        'share0': share0,
        'share1': share1,
        'lambda': lambda_val,
        'mu': mu_val,
        'n': n,
        'n_sq': n_sq
    }


# ---------------------------
# 加密函数
# ---------------------------
def encrypt(pubkey, m):
    return pubkey.encrypt(m)


# ---------------------------
# 部分解密函数
# ---------------------------
def partial_decrypt(ciphertext_obj, share_i, pubkey, n_sq):
    c = ciphertext_obj.ciphertext(False)  # 获取原始密文整数
    u = pow(c, share_i, n_sq)
    print(f"Partial decrypt result u: {u}")  # 添加调试信息
    return u  # 返回部分解密结果u


# ---------------------------
# 聚合两份部分解密 → 明文
# ---------------------------
def combine_shares(u0, u1, mu, n, max_bits=4096):
    # ⚠️ 安全检查：防止部分解密数值过大
    if u0.bit_length() > max_bits or u1.bit_length() > max_bits:
        print("⚠️ 警告：部分解密结果位数过大！可能存在浮点误加密或原始明文过大！")
        print(f"u0 位数: {u0.bit_length()} bit")
        print(f"u1 位数: {u1.bit_length()} bit")
    u = (u0 * u1) % (n * n)  # 合并部分解密结果
    L = (u - 1) // n  # 计算L函数
    m = (L * mu) % n  # 恢复明文
    print(f"Combined shares result m: {m}")  # 添加调试信息
    return int(m)


# ---------------------------
# 快速测试主函数
# ---------------------------
if __name__ == "__main__":
    keys = generate_threshold_keypair()
    pk = keys['pubkey']
    mu = keys['mu']
    n = keys['n']
    n_sq = keys['n_sq']

    m = 200
    c = encrypt(pk, m)
    print("明文:", m)

    # 执行部分解密
    u0 = partial_decrypt(c, keys['share0'], pk, n_sq)
    u1 = partial_decrypt(c, keys['share1'], pk, n_sq)

    # 聚合部分解密结果
    m_recovered = combine_shares(u0, u1, mu, n)
    print("解密还原:", m_recovered)