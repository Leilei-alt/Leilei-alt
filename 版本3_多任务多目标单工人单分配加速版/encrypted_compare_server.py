# 模拟辅助服务器 S1 的比较服务

def s1_compare_encrypted(enc_a, enc_b, privkey):
    """
    模拟 S1 服务器对两个 Paillier 密文执行比较 a < b 的判断。
    :param enc_a: EncryptedNumber
    :param enc_b: EncryptedNumber
    :param privkey: S1 持有的 Paillier 私钥
    :return: 布尔值 True if a < b else False
    """
    a = privkey.decrypt(enc_a)
    b = privkey.decrypt(enc_b)
    return a < b

