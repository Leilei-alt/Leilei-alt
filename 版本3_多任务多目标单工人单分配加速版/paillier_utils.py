from phe import paillier

# --------------------------------------------------
# 🔑 生成 Paillier 密钥对
# --------------------------------------------------
def generate_keypair():
    """
    生成 Paillier 公钥与私钥
    :return: (public_key, private_key)
    """
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

# --------------------------------------------------
# 🔐 批量加密
# --------------------------------------------------
def encrypt_list(public_key, values):
    """
    使用公钥批量加密一组数值
    :param public_key: Paillier 公钥
    :param values: List[float/int] 明文列表
    :return: List[EncryptedNumber]
    """
    return [public_key.encrypt(x) for x in values]

# --------------------------------------------------
# 🔓 批量解密
# --------------------------------------------------
def decrypt_list(private_key, encrypted_values):
    """
    使用私钥批量解密一组加密数值
    :param private_key: Paillier 私钥
    :param encrypted_values: List[EncryptedNumber]
    :return: List[float]
    """
    return [private_key.decrypt(x) for x in encrypted_values]
