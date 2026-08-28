import numpy as np
import torch
import torch.fft as fft
import matplotlib.pyplot as plt
import os

# ========== 参数 ==========
export_dir = "results_9channels_20251230_1233"  # 修改为你的文件夹
npz_file = os.path.join(export_dir, "optimized_results.npz")

# 创建子文件夹保存图像
img_dir = os.path.join(export_dir, "images")
os.makedirs(img_dir, exist_ok=True)

# ========== 工具函数 ==========
def fftshift2(x: torch.Tensor):
    H, W = x.shape[-2], x.shape[-1]
    return torch.roll(torch.roll(x, shifts=(H // 2,), dims=(-2,)),
                      shifts=(W // 2,), dims=(-1))

# ========== 读取数据 ==========
data = np.load(npz_file, allow_pickle=True)
phdx = torch.tensor(data["phdx"], dtype=torch.float32)
phdy = torch.tensor(data["phdy"], dtype=torch.float32)
pairMat = data["pairMat"]
targets = data["targets"]
costs = data["costs"]
eta_history = data.get("eta_history", None)  # 可选

Nchan = targets.shape[0]

# ========== 显示目标图像 ==========
nRows = 4
nCols = int(np.ceil(Nchan / nRows))
plt.figure(figsize=(12, 8))
for c in range(Nchan):
    plt.subplot(nRows, nCols, c+1)
    plt.imshow(targets[c], cmap="gray")
    plt.axis("off")
    plt.title(f"Ch{c+1}")
plt.suptitle("Target Images")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "targets.png"), dpi=300)
plt.show()

# ========== 计算衍射图 ==========
ih_abs_all = []
ih_norm_all = []

plt.figure(figsize=(12, 8))
for c in range(Nchan):
    m, n = pairMat[c]
    ph = m * phdx + n * phdy
    field = torch.exp(1j * ph)
    F = fft.fft2(field)
    Fshift = fftshift2(F)
    Ih_abs = torch.abs(Fshift)**2  # 绝对强度
    Ih_norm = Ih_abs / (torch.max(Ih_abs) + 1e-8)  # 归一化

    ih_abs_all.append(Ih_abs.numpy())
    ih_norm_all.append(Ih_norm.numpy())

    plt.subplot(nRows, nCols, c+1)
    plt.imshow(Ih_norm.numpy(), cmap="gray")
    plt.axis("off")
    plt.title(f"Ch{c+1}")
plt.suptitle("Normalized Intensity (0-1)")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "holograms_normalized.png"), dpi=300)
plt.show()

# 绝对强度保存
plt.figure(figsize=(12, 8))
for c in range(Nchan):
    plt.subplot(nRows, nCols, c+1)
    plt.imshow(ih_abs_all[c], cmap="gray")
    plt.axis("off")
    plt.title(f"Ch{c+1}")
plt.suptitle("Absolute Intensity")
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "holograms_absolute.png"), dpi=300)
plt.show()

# ========== 保存每个通道的强度数据 ==========
for c in range(Nchan):
    np.savetxt(os.path.join(img_dir, f"ih_abs_Ch{c+1}.csv"), ih_abs_all[c], fmt="%.6f", delimiter=",")
    np.savetxt(os.path.join(img_dir, f"ih_norm_Ch{c+1}.csv"), ih_norm_all[c], fmt="%.6f", delimiter=",")

# ========== 显示迭代收敛曲线 ==========
plt.figure()
plt.plot(np.arange(1, len(costs)+1), costs, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_dir, "loss_curve.png"), dpi=300)
plt.show()

# ========== 显示能量效率 η（如果存在） ==========
if eta_history is not None:
    eta_history_np = np.array(eta_history)
    plt.figure(figsize=(10,6))
    for c in range(Nchan):
        plt.plot(eta_history_np[:, c], label=f"Ch{c+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Efficiency η")
    plt.title("Channel Efficiencies During Training")
    plt.legend(ncol=3, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "eta_curve.png"), dpi=300)
    plt.show()
    # 保存最终 η
    final_eta = eta_history_np[-1]
    np.savetxt(os.path.join(img_dir, "eta_final.csv"),
               np.stack([np.arange(1, Nchan+1), final_eta], axis=1),
               fmt="%.6f", delimiter=",", header="Channel,FinalEta")

print(f"所有图像和数据已保存到 {img_dir}")
