import numpy as np
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
