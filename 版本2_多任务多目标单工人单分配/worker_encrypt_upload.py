from threshold_paillier import generate_threshold_keypair, encrypt
import pickle

def simulate_worker_upload():
    keys = generate_threshold_keypair()
    pubkey = keys['pubkey']
    share0, share1 = keys['share0'], keys['share1']
    mu, n, n_sq = keys['mu'], keys['n'], keys['n_sq']

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
            "quals": enc_quals
        }, f)

    with open("threshold_key_shares.pkl", "wb") as f:
        pickle.dump({
            "share0": share0,
            "share1": share1,
            "mu": mu,
            "n": n,
            "n_sq": n_sq
        }, f)

    print("✅ 工人数据加密并上传成功！")
