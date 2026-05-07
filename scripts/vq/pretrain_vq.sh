
export CUDA_VISIBLE_DEVICES=1
python scripts/pretrain_vq.py --data_dir data/datasets_rlds/modified_libero_rlds \
        --data_mix libero_all_no_noops --action_dim 7 --future_action_horizon 7 --vqvae_n_embed 256