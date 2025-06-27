from phe import paillier
import random
import numpy as np
import pickle
import os
import math


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


# ----------------------------------------------------
# 📌 Fast Non-Dominated Sorting
# 输入：二维列表 obj_values，每个元素是一个个体的 [目标1, 目标2]
# 输出：Pareto 层级列表 fronts，每一层包含个体索引
# ----------------------------------------------------
def fast_non_dominated_sort(obj_values):
    S = [[] for _ in range(len(obj_values))]
    n = [0 for _ in range(len(obj_values))]
    rank = [0 for _ in range(len(obj_values))]

    fronts = [[]]

    for p in range(len(obj_values)):
        S[p] = []
        n[p] = 0
        for q in range(len(obj_values)):
            if dominates(obj_values[p], obj_values[q]):
                S[p].append(q)
            elif dominates(obj_values[q], obj_values[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)

    if len(fronts[-1]) == 0:
        fronts.pop()
    return fronts


def dominates(ind1, ind2):
    return all(x <= y for x, y in zip(ind1, ind2)) and any(x < y for x, y in zip(ind1, ind2))


# ----------------------------------------------------
# 📌 Crowding Distance
# 输入：obj_values：二维目标值列表，front为个体索引列表
# 输出：返回对应个体的 WCD 距离（np.array，未在front中的为0）
# 参数 alpha 控制权重（目标空间 vs 决策空间）
# ----------------------------------------------------
def compute_weighted_crowding_distance(obj_values, dec_values, front, alpha=0.5):
    n_obj = len(obj_values[0])
    distances = np.zeros(len(obj_values))

    front_obj = [obj_values[i] for i in front]
    front_dec = [dec_values[i] for i in front]

    obj_array = np.array(front_obj)
    dec_array = np.array(front_dec)

    norm_obj = (obj_array - obj_array.min(axis=0)) / (np.ptp(obj_array, axis=0) + 1e-9)
    norm_dec = (dec_array - dec_array.min(axis=0)) / (np.ptp(dec_array, axis=0) + 1e-9)

    dist_obj = np.zeros(len(front))
    dist_dec = np.zeros(len(front))

    for m in range(n_obj):
        idx = np.argsort(norm_obj[:, m])
        dist_obj[idx[0]] = dist_obj[idx[-1]] = float('inf')
        for i in range(1, len(front) - 1):
            dist_obj[idx[i]] += norm_obj[idx[i + 1], m] - norm_obj[idx[i - 1], m]

    for m in range(dec_array.shape[1]):
        idx = np.argsort(norm_dec[:, m])
        dist_dec[idx[0]] = dist_dec[idx[-1]] = float('inf')
        for i in range(1, len(front) - 1):
            dist_dec[idx[i]] += norm_dec[idx[i + 1], m] - norm_dec[idx[i - 1], m]

    for i, idx in enumerate(front):
        distances[idx] = alpha * dist_obj[i] + (1 - alpha) * dist_dec[i]

    return distances


# ---------------------------
# 阈值密钥生成（share0 + share1 = λ）
# ---------------------------
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def generate_threshold_keypair():
    pubkey, privkey = paillier.generate_paillier_keypair()
    n = pubkey.n
    n_sq = n * n

    # 通过公开API计算lambda和mu
    p, q = privkey.p, privkey.q
    lambda_val = lcm(p - 1, q - 1)
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
def combine_shares(u0, u1, lambda_val, mu, n, n_sq):
    """
    聚合两份部分解密结果得到明文
    """
    u = (u0 * u1) % n_sq  # 合并部分解密结果
    L = (u - 1) // n  # 计算L函数
    m = (L * mu) % n  # 恢复明文
    print(f"Combined shares result m: {m}")  # 添加调试信息
    return int(m)


def simulate_worker_upload():
    keys = generate_threshold_keypair()
    pubkey = keys['pubkey']
    share0, share1 = keys['share0'], keys['share1']
    mu, n, n_sq = keys['mu'], keys['n'], keys['n_sq']
    lambda_val = keys['lambda']  # 获取lambda_val

    # 原始数据
    costs = [10, 15, 12, 20, 5]
    raw_quals = [9, 8, 5, 7, 6]

    # 加密数据
    enc_costs = [encrypt(pubkey, c) for c in costs]
    enc_quals = [encrypt(pubkey, q) for q in raw_quals]

    with open("enc_worker_data.pkl", "wb") as f:
        pickle.dump({
            "pubkey": pubkey,
            "costs": enc_costs,
            "quals": enc_quals,
            "lambda": lambda_val  # 保存lambda_val到文件
        }, f)

    with open("threshold_key_shares.pkl", "wb") as f:
        pickle.dump({
            "share0": share0,
            "share1": share1,
            "mu": mu,
            "n": n,
            "n_sq": n_sq,
            "lambda": lambda_val  # 保存lambda_val到文件
        }, f)

    print("✅ 工人数据加密并上传成功！")


# -------------------------------
# 密文目标函数评估
# -------------------------------
def evaluate_cost_stable(x, enc_costs, pubkey):
    total = pubkey.encrypt(0)
    for i in range(len(x)):
        if x[i]:
            total += enc_costs[i]
    return total


def evaluate_quality_stable(x, enc_quals, pubkey):
    total = pubkey.encrypt(0)
    for i in range(len(x)):
        if x[i]:
            total += enc_quals[i]
    return total


# -------------------------------
# 随机解生成与变异
# -------------------------------
def generate_random_solution(n):
    return [random.randint(0, 1) for _ in range(n)]


def mutate_solution(x):
    return [1 - xi if random.random() < 0.1 else xi for xi in x]


# -------------------------------
# 主优化器（支持阈值解密 + 分配约束）
# -------------------------------
def run_moeo_wcd(pubkey, enc_costs, enc_quals,
                 share0, share1, lambda_val, mu, n, n_sq,
                 num_iter=30, pop_size=20):
    n_var = len(enc_costs)
    population = [generate_random_solution(n_var) for _ in range(pop_size)]

    for gen in range(num_iter):
        print(f"📘 Generation {gen + 1}")
        obj_list = []

        for x in population:
            if sum(x) < 2:
                # ❌ 不满足最小分配要求：设定为“劣解”
                dec_cost = 99999
                dec_qual = 0
            else:
                # ✅ 正常协同解密：成本
                cost = evaluate_cost_stable(x, enc_costs, pubkey)
                u0 = partial_decrypt(cost, share0, pubkey, n_sq)
                u1 = partial_decrypt(cost, share1, pubkey, n_sq)
                dec_cost = combine_shares(u0, u1, lambda_val, mu, n, n_sq)
                print(f"成本解密结果: {dec_cost}, 解向量: {x}")

                # ✅ 正常协同解密：质量
                qual = evaluate_quality_stable(x, enc_quals, pubkey)
                uq0 = partial_decrypt(qual, share0, pubkey, n_sq)
                uq1 = partial_decrypt(qual, share1, pubkey, n_sq)
                dec_qual = combine_shares(uq0, uq1, lambda_val, mu, n, n_sq)
                print(f"质量解密结果: {dec_qual}, 解向量: {x}")

            obj_list.append([dec_cost, dec_qual])

        # 非支配排序 + 拥塞距离
        fronts = fast_non_dominated_sort(obj_list)
        crowding = compute_weighted_crowding_distance(obj_list, population, fronts[0], alpha=0.5)

        elite_indices = sorted(fronts[0], key=lambda i: -crowding[i])[:pop_size // 2]
        new_population = [population[i] for i in elite_indices]

        while len(new_population) < pop_size:
            parent = random.choice(new_population)
            child = mutate_solution(parent)
            new_population.append(child)

        population = new_population

    # 最后一代：找质量最高解
    best_idx = max(range(len(population)), key=lambda i: obj_list[i][1])
    best_x = population[best_idx]
    best_cost, best_qual = obj_list[best_idx]

    print("✅ 优化完成，最优匹配解:")
    print(f"   解向量: {best_x}")
    print(f"   成本: {best_cost:.2f}, 质量: {best_qual:.2f}")

    # 输出实际选择的工人索引
    selected_workers = [i for i, val in enumerate(best_x) if val == 1]
    print(f"   选中的工人索引: {selected_workers}")

    return best_x


# -----------------------------
# 工人数据准备 + 平台密钥加载
# -----------------------------
def prepare_data():
    if not os.path.exists("enc_worker_data.pkl") or not os.path.exists("threshold_key_shares.pkl"):
        simulate_worker_upload()

    with open('enc_worker_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('threshold_key_shares.pkl', 'rb') as f:
        key_parts = pickle.load(f)

    # 修改这里：添加默认值或检查lambda是否存在
    lambda_val = key_parts.get('lambda', None)
    if lambda_val is None:
        # 如果没有找到lambda，重新生成密钥对
        print("警告：未找到lambda值，重新生成密钥对...")
        keys = generate_threshold_keypair()
        lambda_val = keys['lambda']
        key_parts['lambda'] = lambda_val
        # 保存更新后的密钥部分
        with open('threshold_key_shares.pkl', 'wb') as f:
            pickle.dump(key_parts, f)

    return (
        data['pubkey'], data['costs'], data['quals'],  # 公钥 + 密文目标
        key_parts['share0'], key_parts['share1'],      # 两方私钥份额
        lambda_val, key_parts['mu'], key_parts['n'], key_parts['n_sq']  # 聚合用参数
    )


# 手动开方
def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


# 后加的密文下乘法（SMUL）
def secure_multiply(enc_x, enc_y, pubkey, share0, share1, lambda_val, mu, n, n_sq):
    """
    安全密文乘法：返回加密形式的 x * y
    :param enc_x: 密文 \llbracket x \rrbracket
    :param enc_y: 密文 \llbracket y \rrbracket
    :return: 密文 \llbracket x * y \rrbracket
    """
    # 随机掩码，控制在合理范围
    max_r = min(isqrt(pubkey.n) // 8, n // 100)
    r1 = random.randint(1, max_r)
    r2 = random.randint(1, max_r)

    # 扰动掩码加密
    enc_r1 = pubkey.encrypt(r1)
    enc_r2 = pubkey.encrypt(r2)
    enc_r1r2 = pubkey.encrypt(r1 * r2)

    # 构造干扰项
    enc_x_r2 = enc_x * r2 * -1
    enc_y_r1 = enc_y * r1 * -1

    # 构造混合掩码
    enc_x_ = enc_x + enc_r1
    enc_y_ = enc_y + enc_r2

    # 密文乘法（模拟 S1 的解密协作）
    # 恢复 y + r2
    u0 = partial_decrypt(enc_y_, share0, pubkey, n_sq)
    u1 = partial_decrypt(enc_y_, share1, pubkey, n_sq)
    y_plus_r2 = combine_shares(u0, u1, lambda_val, mu, n, n_sq)

    # 执行密文乘明文
    enc_xy_noisy = enc_x_ * y_plus_r2

    # 消除干扰项
    result = enc_xy_noisy + enc_x_r2 + enc_y_r1 + enc_r1r2 * -1
    return result


# 后加的密文比较协议SCMP
def secure_compare(enc_x, enc_y, pubkey, share0, share1, lambda_val, mu, n, n_sq):
    """
    密文比较协议：判断 x < y，返回 \llbracket 1 \rrbracket 或 \llbracket 0 \rrbracket
    """
    # 适当缩小扰动范围
    r = random.randint(2, 2 ** 10)
    r_dash = random.randint(n // 5, n // 4)

    # 构造 \llbracket y - x + 1 \rrbracket
    enc_diff_base = enc_y - enc_x + pubkey.encrypt(1)

    # 逐步构造：\llbracket r·(x−y+1) \rrbracket
    enc_r_diff = enc_diff_base * r

    # 再加上扰动偏移 r'
    enc_full = enc_r_diff + pubkey.encrypt(r_dash)

    # 门限解密
    u0 = partial_decrypt(enc_full, share0, pubkey, n_sq)
    u1 = partial_decrypt(enc_full, share1, pubkey, n_sq)
    d = combine_shares(u0, u1, lambda_val, mu, n, n_sq)

    # 输出密文形式的结果
    threshold = r_dash  # 改为与 r' 本身比较
    if d > threshold:
        return pubkey.encrypt(1)
    else:
        return pubkey.encrypt(0)


# 离线加载缓存
def generate_offline_cache(pubkey, num_sets=10):
    """
    离线阶段生成密文扰动缓存元组：包括 enc(r1), enc(r2), enc(-r1*r2), enc(0), enc(1)
    :param pubkey: Paillier 公钥
    :param num_sets: 要生成的缓存元组数量
    :return: None（写入本地文件）
    """
    cache_list = []

    for _ in range(num_sets):
        r1 = random.randint(1, pubkey.n // 4)
        r2 = random.randint(1, pubkey.n // 4)
        enc_r1 = pubkey.encrypt(r1)
        enc_r2 = pubkey.encrypt(r2)
        enc_r1r2 = pubkey.encrypt(r1 * r2)
        enc_0 = pubkey.encrypt(0)
        enc_1 = pubkey.encrypt(1)

        cache_list.append({
            'r1': r1, 'r2': r2,
            'enc_r1': enc_r1,
            'enc_r2': enc_r2,
            'enc_r1r2': enc_r1r2,
            'enc_0': enc_0,
            'enc_1': enc_1
        })

    with open("offline_cache.pkl", "wb") as f:
        pickle.dump(cache_list, f)

    print(f"✅ 离线缓存生成完毕，共 {num_sets} 组扰动元组")


# 在线阶段加载缓存
def load_offline_cache():
    """
    加载本地保存的 offline 缓存扰动元组
    :return: list of dict
    """
    with open("offline_cache.pkl", "rb") as f:
        return pickle.load(f)


# -----------------------------
# 主运行流程
# -----------------------------
def main():
    print("🔐 加密众包优化系统启动...")
    pubkey, enc_costs, enc_quals, share0, share1, lambda_val, mu, n, n_sq = prepare_data()

    best_x = run_moeo_wcd(
        pubkey=pubkey,
        enc_costs=enc_costs,
        enc_quals=enc_quals,
        share0=share0,
        share1=share1,
        lambda_val=lambda_val,
        mu=mu,
        n=n,
        n_sq=n_sq,
        num_iter=10,
        pop_size=20
    )

    print(f"\n✅ 最终任务分配方案（0表示未选中，1表示被分配）：\n{best_x}")

    # 验证原始数据
    costs = [10, 15, 12, 20, 5]
    quals = [9, 8, 5, 7, 6]

    selected_cost = sum([costs[i] for i, val in enumerate(best_x) if val == 1])
    selected_qual = sum([quals[i] for i, val in enumerate(best_x) if val == 1])

    print(f"🔍 理论计算：选中工人的总成本 = {selected_cost}, 总质量 = {selected_qual}")


# -----------------------------
if __name__ == "__main__":
    main()