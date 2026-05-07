## Installation
```Bash
git clone 
cd CausalActionTokenizer
conda create -n catok python=3.10 -y
conda activate catok
pip install -r requirements.txt
pip install -e .
```

## Training and Evaluation
### Pretraining:
```Bash
sh <script_path> [mode] [gpu_id] [config_name] [config_path]

# examples:
# single gpu pretraining
sh scripts/pretrain/pretrain_vanilla_tokenizer_diffusion_libero.sh single 0 vanilla_diffusion ../conf/config_libero
# multi gpus pretraining
# (train on 4 gpus with conf/config_libero/vanilla_diffusion.yaml)
sh scripts/pretrain/pretrain_vanilla_tokenizer_diffusion_libero.sh multi 0,1,2,3 vanilla_diffusion ../conf/config_libero
```
+ `mode` (Optional): Training strategy. Use `single` for one GPU or `multi` for distributed training.

+ `gpu_id` (Optional): The index of GPUs to use (e.g., `0` or `0,1,2,3`).

+ `config_name` (Optional): The filename of your config in `config_path`. Defaults to `vanilla_diffusion`.

+ `config_path` (Optional): The directory of your config. Defaults to `../conf/config_libero`.

### VanillaFlow + DiTi config

You can enable progressive token scheduling (`DiTi`) for `VanillaFlow` with decoder-side config:

```yaml
tokenizer:
  basic:
    flow_type: vanilla
  decoder:
    vanilla_use_diti: true
    vanilla_diti_type: cont               # uniform | cont | normal
    vanilla_diti_stages: "100,600,1000"   # used by uniform/cont
    vanilla_diti_k_per_stage: "2,10,4"    # used by uniform/cont
    vanilla_t2k: 1.0
    vanilla_diti_input_mode: auto         # auto | normalized | scaled_1000
```

Quick launch (single GPU):

```bash
sh scripts/pretrain/train_vanilla_diti.sh single 0
```

Time semantics check:
- `VanillaFlow` (`catok/models/catok_ddt/sd3/vanilla_flow.py`): `t=0` is closer to noise, `t=1` is closer to data.
- `RectifiedFlow` in the same file: `t=1` is closer to noise, `t=0` is closer to data.

### VanillaFlow sampling controls and self-consistency

Recent updates add three inference controls for VanillaFlow:

- `solver`: ODE solver for flow sampling, supports `euler` and `heun`.
- `one_step` (flow): force flow solver to run one ODE step.
- `full_tokens` (flow): use full-token conditioning mask at inference (all tokens visible).

And one training regularizer:

- `consistency loss`: enforces local self-consistency of velocity predictions across nearby timesteps.

Decoder-side config example:

```yaml
tokenizer:
  basic:
    flow_type: vanilla
  decoder:
    # Existing controls
    vanilla_use_diti: true
    vanilla_diti_type: cont
    vanilla_t2k: 1.0
    vanilla_diti_input_mode: auto

    # New: self-consistency loss (training-time)
    vanilla_consistency_weight: 0.05          # 0.0 disables consistency loss
    vanilla_consistency_delta_t: 0.05         # random delta_t sampled in [0, this value]
    vanilla_consistency_detach_target: true   # recommended for stable optimization
```

Recommended starting values:

- `vanilla_consistency_weight`: `0.05` (or `0.1` for stronger regularization)
- `vanilla_consistency_delta_t`: `0.05`
- `vanilla_consistency_detach_target`: `true`

### VanillaFlow logit-normal timestep sampling

By default `VanillaFlow` samples the training timestep `t` uniformly from `[0, 1]`. You can switch to **logit-normal** sampling to concentrate training signal near `t ≈ 0.5` (or any target region), which is the scheme used in SD3.

Sampling rule: `u ~ N(mean, std)`, then `t = sigmoid(u)`.

Configure via the `tokenizer.flow` section:

```yaml
tokenizer:
  flow:
    use_logit_normal: true
    logit_normal_mean: 0.0   # 0.0 → concentrate around t=0.5
    logit_normal_std: 1.0    # larger → wider spread
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `use_logit_normal` | `false` | Enable logit-normal sampling (uniform when `false`) |
| `logit_normal_mean` | `0.0` | Shift peak of `t` distribution (`>0` → toward `t=1`, `<0` → toward `t=0`) |
| `logit_normal_std` | `1.0` | Spread of the distribution (larger → closer to uniform) |

### Finetuning (one-step decoder):
```Bash
# single GPU
torchrun --nproc_per_node=1 \
  scripts/finetune/finetune_multidataset_onestep.py \
  --ckpt /path/to/your/checkpoint.pth \
  --use_ddp \
  --epochs 800 \
  --lr 1e-4

# multi GPU
torchrun --nproc_per_node=4 \
  scripts/finetune/finetune_multidataset_onestep.py \
  --ckpt /path/to/your/checkpoint.pth \
  --use_ddp \
  --epochs 800 \
  --lr 1e-4
```

### Evaluation:

```Bash
# multi dataset (all datasets in config)
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth --one_step

# flow sampling with explicit method controls
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth \
  --sampling_method flow --flow_solver euler --steps 20
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth \
  --sampling_method flow --flow_solver heun --steps 20
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth \
  --sampling_method flow --flow_solver heun --steps 20 --flow_full_tokens
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth \
  --sampling_method flow --flow_solver euler --flow_one_step
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth \
  --sampling_method decoder_one_step

# per-dataset evaluation (evaluate each dataset individually, output summary table)
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth --per_dataset

# single dataset evaluation (evaluate only the specified dataset)
python utils/multidatasets/evaluation_vanilla_diffusion.py -m ckpt/best.pth --dataset bridge

# single dataset (Libero only, legacy script)
python utils/libero/evaluation_vanilla_diffusion.py -m ckpt/best.pth
```
+ `model_path`: Path to your model checkpoint (e.g., best.pth).

+ `steps` (Optional): Number of sampling steps for the ODE solver. Default to `20`.

+ `sampling_method` (Optional): `flow` or `decoder_one_step`. Default to `flow`.

+ `flow_solver` (Optional): ODE solver for flow sampling. Choices: `euler`, `heun`.

+ `flow_one_step` (Optional): Force flow sampling to run only one ODE step.

+ `flow_full_tokens` (Optional): Use full-token mask during flow inference.

+ `one_step` (Optional, deprecated): Same as `--sampling_method decoder_one_step`.

+ `per_dataset` (Optional): Evaluate each dataset individually and report per-dataset metrics with a summary table. Results are saved to `eval_*_per_dataset.json`.

+ `dataset` (Optional): Evaluate only the specified dataset by name (e.g., `bridge`). Implies `--per_dataset`.

### Visualization:
```Bash
python <script_path> --model_path <ckpt> [--steps <steps>] [--one_step]

# single dataset (Libero)
python utils/libero/visualization_vanilla_diffusion.py -m ckpt/best.pth

# multi dataset (bridge + Libero), per-episode trajectory plots
python utils/multidatasets/visualization_vanilla_diffusion.py -m ckpt/best.pth
python utils/multidatasets/visualization_vanilla_diffusion.py -m ckpt/best.pth --one_step --max_episodes_per_dataset 5
```
+ `max_episodes_per_dataset` (Optional): Number of episodes to visualize per dataset. Default to `3`.

### Token-level Analysis:
```Bash
python utils/multidatasets/visualize_diffusion_tokens.py -m <ckpt> [--steps <steps>] [--mode {tokens,embedding,both}]

# run both visualizations
python utils/multidatasets/visualize_diffusion_tokens.py -m ckpt/best.pth --steps 20 --mode both

# only incremental-token reconstruction (1, 2, ..., num_tokens)
python utils/multidatasets/visualize_diffusion_tokens.py -m ckpt/best.pth --steps 20 --mode tokens

# only encoder embedding t-SNE distribution
python utils/multidatasets/visualize_diffusion_tokens.py -m ckpt/best.pth --mode embedding --max_tsne_samples 3000 --tsne_perplexity 50
```
+ `mode` (Optional): Which visualization to run. `tokens` for incremental-token diffusion reconstruction, `embedding` for encoder embedding t-SNE, `both` for both. Default to `both`.
+ `max_tsne_samples` (Optional): Max samples for t-SNE to keep runtime tractable. Default to `2000`.
+ `tsne_perplexity` (Optional): t-SNE perplexity hyperparameter. Default to `30`.

## Dataset Convetion


### Weighted Dataloader Usage

Use 'catok.training.rlds_dataset' to load RLDS dataset.
```python
from catok.training.rlds_dataset import make_rlds_dataloader
sampler, loader = make_rlds_dataloader(
    dataset_specs=[
        ("bridge", 1.0),
        ("Libero_RLDS/libero_spatial_no_noops", 5.0),
        ("Libero_RLDS/libero_goal_no_noops", 5.0),
        ("Libero_RLDS/libero_object_no_noops", 5.0),
        ("Libero_RLDS/libero_10_no_noops", 5.0),
    ],
    data_root="data",
    horizon=10,
    batch_size=64,
    action_only=True,
    num_parallel_reads=8,
    num_workers=0,
    target_action_dim=8,
    stats_path_map={
        "bridge": "data/bridge/stats.json",
        "Libero_RLDS/libero_spatial_no_noops": "data/Libero_RLDS/libero_spatial_no_noops/stats.json",
        "Libero_RLDS/libero_goal_no_noops": "data/Libero_RLDS/libero_goal_no_noops/stats.json",
        "Libero_RLDS/libero_object_no_noops": "data/Libero_RLDS/libero_object_no_noops/stats.json",
        "Libero_RLDS/libero_10_no_noops": "data/Libero_RLDS/libero_10_no_noops/stats.json",
    },
    normalizer_method_map={"bridge": "bridge"},
    normalizer_config={"clip": True, "action_dim": 7},
    action_dim_map={
        "bridge": 7,
        "Libero_RLDS/libero_spatial_no_noops": 7,
        "Libero_RLDS/libero_goal_no_noops": 7,
        "Libero_RLDS/libero_object_no_noops": 7,
        "Libero_RLDS/libero_10_no_noops": 7,
    },
)
batch = next(iter(loader))
print("action:", batch.shape, batch.dtype)  # (64, 10, 8)
```