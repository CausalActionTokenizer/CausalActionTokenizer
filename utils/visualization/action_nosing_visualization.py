import numpy as np
import matplotlib.pyplot as plt
from catok.training.rlds_dataset import make_rlds_dataloader

H = 160

# ====== load data ======
sampler, loader, _ = make_rlds_dataloader(
    dataset_specs=[
        ("Libero_RLDS/libero_object_no_noops", 1.0),
    ],
    data_root="data",
    horizon=H,
    batch_size=1,
    action_only=True,
    num_parallel_reads=8,
    num_workers=0,
    target_action_dim=8,
    stats_path_map={
        "Libero_RLDS/libero_object_no_noops": "data/Libero_RLDS/libero_object_no_noops/stats.json",
    },
    normalizer_method_map={"Libero_RLDS/libero_object_no_noops": "qq"},
    normalizer_config={"clip": True, "action_dim": 7},
)

batch = next(iter(loader))

def add_noise(actions, noise_level=0.1):
    noise_actions = np.random.normal(0, noise_level, actions.shape)
    noised = actions.numpy().copy()
    noised[..., :-1] += noise_actions[..., :-1]
    return noised

# ====== prepare ======
actions = batch[0][..., :-1]  # (H, 7)
to_draw = actions.T  # (7, H)

# noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
noise_levels = [0.05, 0.1, 0.15, 0.2]

# ====== plot ======
num_rows = 7
num_cols = 1 + len(noise_levels)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 2*num_rows), sharex=True)

# ---- 第一列：原始 ----
for i in range(num_rows):
    axes[i, 0].plot(to_draw[i])
    axes[i, 0].set_ylabel(f"dim {i}")
    axes[i, 0].set_ylim(-1.1, 1.1)
    axes[i, 0].axhline(y=-1, linestyle='--', linewidth=0.5)
    axes[i, 0].axhline(y=1, linestyle='--', linewidth=0.5)

axes[0, 0].set_title("Original")

# ---- 后面列：不同 noise ----
for j, nl in enumerate(noise_levels):
    noised = add_noise(actions, nl).clip(-1,1).T

    for i in range(num_rows):
        axes[i, j+1].plot(noised[i])
        axes[i, j+1].set_ylim(-1.1, 1.1)
        axes[i, j+1].axhline(y=-1, linestyle='--', linewidth=0.5)
        axes[i, j+1].axhline(y=1, linestyle='--', linewidth=0.5)

    axes[0, j+1].set_title(f"noise={nl}")

# x label
for j in range(num_cols):
    axes[-1, j].set_xlabel(f"time (H={H})")

plt.tight_layout()
plt.savefig(f'action_noise_sweep_H{H}.png', dpi=200)
plt.close()

def interpolate_actions(actions, t, noise_level=0.1):
    noise = np.random.normal(0, noise_level, actions.shape)
    noised_actions = (1 - t) * actions.numpy().copy() + t * noise
    return noised_actions

ts = np.linspace(0, 1, 9) 
def draw_interpolation(actions, ts, noise_level=0.1):
    # ====== interpolation visualization ======
    num_rows = 7
    num_cols = len(ts)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 2*num_rows), sharex=True)

    for j, t in enumerate(ts):
        interp = interpolate_actions(actions, t, noise_level=noise_level)  # 可以调 noise_level
        interp = interp.T  # (7, H)

        for i in range(num_rows):
            axes[i, j].plot(interp[i])
            axes[i, j].set_ylim(-1.1, 1.1)
            axes[i, j].axhline(y=-1, linestyle='--', linewidth=0.5)
            axes[i, j].axhline(y=1, linestyle='--', linewidth=0.5)

            if j == 0:
                axes[i, j].set_ylabel(f"dim {i}")

        axes[0, j].set_title(f"t={t:.2f}")

    # x label
    for j in range(num_cols):
        axes[-1, j].set_xlabel(f"time (H={H})")

    plt.tight_layout()
    plt.savefig(f'action_interpolation_H{H}_nl{noise_level}.png', dpi=200)
    plt.close()

for nl in [0.05, 0.1, 0.2, 0.5, 1.0]:
    draw_interpolation(actions, ts, noise_level=nl)
