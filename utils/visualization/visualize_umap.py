import torch, os, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler

save_path = 'outputs/encoder_outs'
os.makedirs(save_path, exist_ok=True)

outs = torch.load('ckpt/20260311_catok_libero-bridge_add16D_K8_CS1024_Layer8_VQ16_CH1/035000.pth')          # (B, L, C)
B, L, C = outs.shape
print('shape:', outs.shape)

# ---------- 1. 准备数据 ----------
# 把所有 token 向量拉平：(B*L, C)
X_all = outs.reshape(-1, C).cpu().numpy()
# 对应的 batch 标签，同一条序列一个颜色
y_all = np.repeat(np.arange(B), L)

# 可选标准化：UMAP 对尺度敏感
X_all = StandardScaler().fit_transform(X_all)

# ---------- 2. UMAP 降维 ----------
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
XY_all = reducer.fit_transform(X_all)        # (B*L, 2)

# 同时也把“每句平均向量”降维，得到 B 个代表点
mean_vec = outs.mean(dim=1).cpu().numpy()    # (B, C)
mean_vec = StandardScaler().fit_transform(mean_vec)
XY_mean = reducer.transform(mean_vec)        # 用已经训好的 reducer

# ---------- 3. 静态图 ----------
plt.figure(figsize=(6,5))
sns.scatterplot(x=XY_all[:,0], y=XY_all[:,1],
                hue=y_all, palette='tab20', s=25, linewidth=0)
# 把代表点用黑色星形标出
plt.scatter(XY_mean[:,0], XY_mean[:,1],
            c='black', marker='*', s=120, label='seq-mean')
plt.legend()
plt.title('UMAP of 32×L token vectors')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'umap_tokens_static.png'), dpi=300)
plt.show()

# ---------- 4. 交互图 ----------
df = dict(x=XY_all[:,0], y=XY_all[:,1],
          batch=[f'seq{i}' for i in y_all],
          idx=np.arange(B*L))
fig = px.scatter(df, x='x', y='y', color='batch',
                 hover_data=['idx'],
                 color_discrete_sequence=px.colors.qualitative.Light24)
# 把代表点再加一层
fig.add_scatter(x=XY_mean[:,0], y=XY_mean[:,1],
                mode='markers', marker=dict(size=12, color='black',
                                            symbol='star'),
                name='seq-mean')
fig.write_html(os.path.join(save_path, 'umap_tokens_interactive.html'))
fig.show()