import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
save_path = 'outputs/encoder_outs'
os.makedirs(save_path, exist_ok=True)

outs = torch.load('outputs/outs.pt')
print(outs.shape)

# draw heatmaps (B, L, C)
B, L, C = outs.shape
print('shape:', outs.shape)
outs = torch.nn.functional.normalize(outs, p=2, dim=-1)

max_show = min(B, 32)          # 最多画 32 张
nrow = int(max_show**0.5)
ncol = (max_show + nrow - 1) // nrow

fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*2))
axes = axes.flatten() if max_show > 1 else [axes]

for i in range(max_show):
    sns.heatmap(outs[i].cpu().float(),
                ax=axes[i],
                cmap='mako',
                cbar=False,
                xticklabels=False,
                yticklabels=False)
    axes[i].set_title(f'seq {i}', fontsize=8)

# 隐藏多余的子图
for j in range(max_show, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'encoder_outs_heatmaps.png'), dpi=300)
plt.show()

