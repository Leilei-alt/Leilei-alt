import matplotlib.pyplot as plt

schemes = ["Fusion-BLS"]
vehicle = [237]
server = [193]
ta = [167]
total = [445]

fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))

axes[0].bar(schemes, vehicle)
axes[0].set_title("(a)")
axes[0].set_ylabel("Communication Cost (Bytes)")
axes[0].set_xlabel("Vehicle/Device")

axes[1].bar(schemes, server)
axes[1].set_title("(b)")
axes[1].set_ylabel("Communication Cost (Bytes)")
axes[1].set_xlabel("RSU/MEC Server")

axes[2].bar(schemes, ta)
axes[2].set_title("(c)")
axes[2].set_ylabel("Communication Cost (Bytes)")
axes[2].set_xlabel("TA")

axes[3].bar(schemes, total)
axes[3].set_title("(d)")
axes[3].set_ylabel("Communication Cost (Bytes)")
axes[3].set_xlabel("Total Entities")

fig.suptitle("Communication Cost Comparisons of Fusion-BLS", fontsize=14)
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
plt.show()
