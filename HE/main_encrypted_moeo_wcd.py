import pickle
import os
from encrypted_moeo_wcd import run_moeo_wcd
from worker_encrypt_upload import simulate_worker_upload

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

    return (
        data['pubkey'], data['costs'], data['quals'],  # 公钥 + 密文目标
        key_parts['share0'], key_parts['share1'],      # 两方私钥份额
        key_parts['mu'], key_parts['n'], key_parts['n_sq']  # 聚合用参数
    )

# -----------------------------
# 主运行流程
# -----------------------------
def main():
    print("🔐 加密众包优化系统启动...")
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
        num_iter=5,
        pop_size=20
    )

    print(f"\n✅ 最终任务分配方案（0表示未选中，1表示被分配）：\n{best_x}")

# -----------------------------
if __name__ == "__main__":
    main()
