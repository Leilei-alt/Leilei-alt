import pickle
import csv
import random
import os
from threshold_paillier import partial_decrypt, combine_shares
from encrypted_moeo_wcd import run_moeo_wcd
from worker_encrypt_upload import simulate_worker_upload
import matplotlib.pyplot as plt

# -------------------------
# 加载密文数据与密钥
# -------------------------
def prepare_data():
    if not os.path.exists("enc_worker_data.pkl") or not os.path.exists("threshold_key_shares.pkl"):
        simulate_worker_upload()

    with open('enc_worker_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('threshold_key_shares.pkl', 'rb') as f:
        keys = pickle.load(f)

    return (
        data['pubkey'], data['costs'], data['quals'],
        keys['share0'], keys['share1'],
        keys['mu'], keys['n'], keys['n_sq']
    )

# -------------------------
# 批量运行 + 可视化 + CSV
# -------------------------
def batch_run(num_runs=2):
    pubkey, enc_costs, enc_quals, share0, share1, mu, n, n_sq = prepare_data()
    results = []

    for run in range(num_runs):
        print(f"\n🚀 第 {run + 1} 次运行中...")
        random.seed(run)
        pubkey, enc_costs, enc_quals, share0, share1, mu, n, n_sq = prepare_data()
        best_x = run_moeo_wcd(
            pubkey=pubkey,
            enc_costs=enc_costs,
            enc_quals=enc_quals,
            share0=share0,
            share1=share1,
            mu=mu,
            n=n,
            n_sq=n_sq,
            num_iter=1,
            pop_size=20
        )

        # 解密成本
        cost_enc = sum([enc_costs[i] * best_x[i] for i in range(len(best_x))])
        u0 = partial_decrypt(cost_enc, share0, pubkey, n_sq)
        u1 = partial_decrypt(cost_enc, share1, pubkey, n_sq)
        dec_cost = combine_shares(u0, u1, mu, n)

        # 解密质量（注意：结果需要除以100）
        qual_enc = sum([enc_quals[i] * best_x[i] for i in range(len(best_x))])
        uq0 = partial_decrypt(qual_enc, share0, pubkey, n_sq)
        uq1 = partial_decrypt(qual_enc, share1, pubkey, n_sq)
        dec_qual = combine_shares(uq0, uq1, mu, n)

        results.append({
            'run': run + 1,
            'solution': best_x,
            'cost': round(dec_cost, 2),
            'quality': round(dec_qual / 100, 4)  # 质量解密后除以100
        })

    # 保存 CSV
    with open("best_solutions_threshold.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'solution', 'cost', 'quality'])
        writer.writeheader()
        writer.writerows(results)
    print("📄 所有最优解保存至 best_solutions_threshold.csv")

    # 绘图
    costs = [r['cost'] for r in results]
    quals = [r['quality'] for r in results]
    plt.figure()
    plt.scatter(costs, quals, color='blue')
    for i in range(len(results)):
        plt.text(costs[i], quals[i], f"R{i+1}", fontsize=8)
    plt.xlabel("成本 (Cost)")
    plt.ylabel("质量 (Quality)")
    plt.title(f"批量运行 {num_runs} 次的最优目标分布图")
    plt.grid(True)
    plt.savefig("batch_objective_distribution_threshold.png")
    plt.show()
    print("📈 图像保存至 batch_objective_distribution_threshold.png")

# -------------------------
if __name__ == "__main__":
    batch_run(num_runs=10)
