import random
from threshold_paillier import partial_decrypt, combine_shares
from utils_ndsort_wcd import fast_non_dominated_sort, compute_weighted_crowding_distance

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
                 share0, share1, mu, n, n_sq,
                 num_iter=30, pop_size=20):

    n_var = len(enc_costs)
    population = [generate_random_solution(n_var) for _ in range(pop_size)]

    for gen in range(num_iter):
        print(f"📘 Generation {gen+1}")
        obj_list = []

        for x in population:
            if sum(x) < 2:
                # ❌ 不满足最小分配要求：设定为“劣解”
                dec_cost = 99999
                dec_qual = 0
            else:
                # ✅ 正常协同解密：成本
                cost =  evaluate_cost_stable(x, enc_costs, pubkey)
                u0 = partial_decrypt(cost, share0, pubkey, n_sq)
                u1 = partial_decrypt(cost, share1, pubkey, n_sq)
                dec_cost = combine_shares(u0, u1, mu, n)
                print(dec_cost)

                # ✅ 正常协同解密：质量
                qual = evaluate_quality_stable(x, enc_quals, pubkey)
                uq0 = partial_decrypt(qual, share0, pubkey, n_sq)
                uq1 = partial_decrypt(qual, share1, pubkey, n_sq)
                dec_qual = combine_shares(uq0, uq1, mu, n)
                print(dec_qual)

            obj_list.append([dec_cost, dec_qual])

        # 非支配排序 + 拥塞距离
        fronts = fast_non_dominated_sort(obj_list)
        crowding = compute_weighted_crowding_distance(obj_list, population, fronts[0], alpha=0.5)

        elite_indices = sorted(fronts[0], key=lambda i: -crowding[i])[:pop_size//2]
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
    print(f"   成本: {best_cost:.2f}, 质量: {best_qual :.2f}")
    return best_x



