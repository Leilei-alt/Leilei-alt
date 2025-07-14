from use_cache import *
# NSGA-II核心实现（加密环境下）
def nsga2_non_dominated_sort_enc(obj_list, pubkey, share0, share1, mu, n, n_sq):
    """
    NSGA-II的非支配排序（支持加密目标值）
    返回：fronts（List[List[int]]）多层非支配解索引
    """
    S = {}  # 支配集合
    n_dom = {}  # 被支配次数
    fronts = [[]]  # 非支配层

    for p in range(len(obj_list)):
        S[p] = []
        n_dom[p] = 0
        for q in range(len(obj_list)):
            if p == q:
                continue
            if dominates_enc(obj_list[p], obj_list[q], pubkey, share0, share1, mu, n, n_sq):
                S[p].append(q)
            elif dominates_enc(obj_list[q], obj_list[p], pubkey, share0, share1, mu, n, n_sq):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # 去掉最后一个空层
    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


def secure_sort_encrypted_numbers(enc_numbers, pubkey, share0, share1, mu, n, n_sq):
    """
    安全排序加密数列表
    返回排序后的索引
    """
    n = len(enc_numbers)
    indices = list(range(n))

    # 简单的冒泡排序实现（可以替换为更高效的排序算法）
    for i in range(n):
        for j in range(0, n - i - 1):
            # 使用安全比较协议判断 enc_numbers[j] > enc_numbers[j+1]
            if secure_greater_than(enc_numbers[indices[j]], enc_numbers[indices[j + 1]],
                                   pubkey, share0, share1, mu, n, n_sq):
                # 交换索引
                indices[j], indices[j + 1] = indices[j + 1], indices[j]

    return indices


def secure_greater_than(enc_a, enc_b, pubkey, share0, share1, mu, n, n_sq):
    """
    安全比较 enc_a > enc_b
    返回布尔值
    """
    # 计算差值 enc_diff = enc_a - enc_b
    enc_diff = enc_a - enc_b

    # 解密差值
    partial0 = partial_decrypt(enc_diff, share0, pubkey, n_sq)
    partial1 = partial_decrypt(enc_diff, share1, pubkey, n_sq)
    diff = combine_shares(partial0, partial1, mu, n)

    # 判断差值是否大于0
    return diff > 0


# 修改拥挤距离计算函数
def nsga2_crowding_distance_assignment(obj_list, indices, pubkey, share0, share1, mu, n, n_sq):
    """
    NSGA-II的拥挤距离计算（支持加密目标值）
    返回：{个体索引: 拥挤距离}
    """
    n_obj = 2  # 成本和质量两个目标
    distances = {i: 0.0 for i in indices}

    for m in range(n_obj):
        # 提取当前目标的加密值
        enc_values = [obj_list[i][1 + m] for i in indices]

        # 使用安全排序获取索引
        sorted_indices = secure_sort_encrypted_numbers(enc_values, pubkey, share0, share1, mu, n, n_sq)
        # 映射回原始索引
        sorted_indices = [indices[i] for i in sorted_indices]

        # 边界点的拥挤距离设为无穷大
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        # 计算目标值范围（需要解密）
        min_val = combine_shares(
            partial_decrypt(obj_list[sorted_indices[0]][1 + m], share0, pubkey, n_sq),
            partial_decrypt(obj_list[sorted_indices[0]][1 + m], share1, pubkey, n_sq),
            mu, n
        )
        max_val = combine_shares(
            partial_decrypt(obj_list[sorted_indices[-1]][1 + m], share0, pubkey, n_sq),
            partial_decrypt(obj_list[sorted_indices[-1]][1 + m], share1, pubkey, n_sq),
            mu, n
        )

        if max_val - min_val == 0:  # 避免除零错误
            continue

        # 计算中间点的拥挤距离
        for i in range(1, len(sorted_indices) - 1):
            # 解密相邻点的目标值差
            current = combine_shares(
                partial_decrypt(obj_list[sorted_indices[i + 1]][1 + m], share0, pubkey, n_sq),
                partial_decrypt(obj_list[sorted_indices[i + 1]][1 + m], share1, pubkey, n_sq),
                mu, n
            )
            prev = combine_shares(
                partial_decrypt(obj_list[sorted_indices[i - 1]][1 + m], share0, pubkey, n_sq),
                partial_decrypt(obj_list[sorted_indices[i - 1]][1 + m], share1, pubkey, n_sq),
                mu, n
            )
            distances[sorted_indices[i]] += (current - prev) / (max_val - min_val)

    return distances

def binary_tournament_selection(population, obj_list, fronts, crowding_dict, pubkey, share0, share1, mu, n, n_sq):
    """
    NSGA-II的二元锦标赛选择（基于非支配层级和拥挤距离）
    """
    idx1, idx2 = random.sample(range(len(population)), 2)

    # 比较非支配层级
    front1 = next(i for i, front in enumerate(fronts) if idx1 in front)
    front2 = next(i for i, front in enumerate(fronts) if idx2 in front)

    if front1 < front2:
        return idx1
    elif front1 > front2:
        return idx2
    else:
        # 层级相同，比较拥挤距离
        if crowding_dict[idx1] > crowding_dict[idx2]:
            return idx1
        else:
            return idx2


def crossover_and_mutation(parent1, parent2, reach, min_assign):
    """
    针对任务分配矩阵的交叉和变异操作
    """
    T = len(parent1)  # 任务数
    N = len(parent1[0])  # 工人数

    # 单点交叉
    crossover_point = random.randint(1, T - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]

    # 变异：随机选择一个任务，重新分配工人
    if random.random() < 0.2:  # 变异率20%
        t = random.randint(0, T - 1)

        # 清空当前任务的分配
        child[t] = [0] * N

        # 重新分配：至少分配min_assign个可达工人
        candidates = [i for i in range(N) if reach[t][i] == 1]
        if len(candidates) >= min_assign:
            assigned = random.sample(candidates, min_assign)
            for i in assigned:
                child[t][i] = 1

    return child


def run_nsga2(pubkey, enc_costs, enc_quals, enc_weights,
              share0, share1, mu, n, n_sq, reach,
              num_iter=30, pop_size=20,
              num_tasks=3, min_assign=2,
              fast_mode=False, init_population=None):
    """
    NSGA-II主函数（加密环境下）
    """
    N = len(enc_costs[0])  # 工人数
    T = num_tasks

    print("✅ 正在初始化初始种群...")

    if init_population is not None:
        population = init_population
        print("✅ 使用外部传入的初始种群")
    else:
        population = [generate_random_matrix_with_reach(T, N, min_assign, reach)
                      for _ in range(pop_size)]
        print("✅ 默认生成初始种群")

    for gen in tqdm(range(num_iter), desc="🌱 进化轮"):
        print(f"\n🌿 第 {gen + 1}/{num_iter} 轮开始")

        # 评估种群目标
        print("⚡ 开始评估当前种群个体目标值...")
        obj_list = parallel_evaluate_population(
            population=population,
            enc_costs=enc_costs,
            enc_quals=enc_quals,
            enc_weights=enc_weights,
            pubkey=pubkey,
            share0=share0,
            share1=share1,
            mu=mu,
            n=n,
            n_sq=n_sq,
            min_assign=min_assign,
            fast_mode=fast_mode
        )
        print("✅ 个体评估完成")

        print("✅ 正在进行非支配排序...")
        fronts = nsga2_non_dominated_sort_enc(obj_list, pubkey, share0, share1, mu, n, n_sq)
        print("✅ 非支配排序完成，前层个体数量:", len(fronts[0]))

        print("📏 计算拥挤距离...")
        # 为每个前沿计算拥挤距离
        crowding_dict = {}
        for front in fronts:
            front_crowding = nsga2_crowding_distance_assignment(obj_list, front, pubkey, share0, share1, mu, n, n_sq)
            crowding_dict.update(front_crowding)
        print("✅ 拥挤距离计算完成")

        # 创建子代
        print("🧬 正在生成下一代个体...")
        offspring = []
        while len(offspring) < pop_size:
            # 选择父代
            parent1_idx = binary_tournament_selection(population, obj_list, fronts, crowding_dict, pubkey, share0,
                                                      share1, mu, n, n_sq)
            parent2_idx = binary_tournament_selection(population, obj_list, fronts, crowding_dict, pubkey, share0,
                                                      share1, mu, n, n_sq)

            # 交叉和变异
            child = crossover_and_mutation(
                population[parent1_idx],
                population[parent2_idx],
                reach,
                min_assign
            )

            offspring.append(child)

        # 合并父代和子代，选择下一代
        combined_pop = population + offspring
        combined_obj = obj_list + parallel_evaluate_population(
            population=offspring,
            enc_costs=enc_costs,
            enc_quals=enc_quals,
            enc_weights=enc_weights,
            pubkey=pubkey,
            share0=share0,
            share1=share1,
            mu=mu,
            n=n,
            n_sq=n_sq,
            min_assign=min_assign,
            fast_mode=fast_mode
        )

        # 基于非支配排序和拥挤距离选择新种群
        new_population = []
        fronts = nsga2_non_dominated_sort_enc(combined_obj, pubkey, share0, share1, mu, n, n_sq)

        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                # 整个前沿都被选中
                new_population.extend([combined_pop[i] for i in front])
            else:
                # 前沿无法全部选中，按拥挤距离排序选择
                front_crowding = nsga2_crowding_distance_assignment(combined_obj, front, pubkey, share0, share1, mu, n,
                                                                    n_sq)
                sorted_indices = sorted(front, key=lambda i: front_crowding[i], reverse=True)
                remaining = pop_size - len(new_population)
                new_population.extend([combined_pop[i] for i in sorted_indices[:remaining]])
                break

        population = new_population
        print(f"✅ 第 {gen + 1} 轮迭代完成，已生成新种群 ✅")

    # 选择最优解（前沿0中拥挤距离最大的解）
    fronts = nsga2_non_dominated_sort_enc(obj_list, pubkey, share0, share1, mu, n, n_sq)
    front0 = fronts[0]
    crowding_dict = nsga2_crowding_distance_assignment(obj_list, front0, pubkey, share0, share1, mu, n, n_sq)

    best_idx = max(front0, key=lambda i: crowding_dict[i])
    best_x, best_cost, best_qual = obj_list[best_idx]

    print("\n🏆 优化完成，最优任务分配方案如下：")
    for t in range(T):
        print(f"  任务{t + 1}: {best_x[t]}")

    return best_x, best_cost, best_qual, obj_list