import torch
import time
import argparse

from catok.models.catok_ddt.vanilla_utils import load_checkpoint


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def benchmark_inference(model, device, batch_size=32, seq_len=16, action_dim=7, warmup=10, iters=50):
    model.eval()

    x = torch.randn(batch_size, seq_len, action_dim, device=device)

    # warmup
    for _ in range(warmup):
        _ = model.encoder(x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = model.encoder(x)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    avg_time = (end - start) / iters
    throughput = batch_size / avg_time

    print(f"Avg inference time per batch: {avg_time * 1000:.3f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    model, cfg = load_checkpoint(args.model_path, device)
    model.to(device)

    # ===== 参数量 =====
    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")

    # ===== inference speed =====
    action_dim = cfg['tokenizer']['basic']['action_dim']
    benchmark_inference(
        model,
        device,
        batch_size=64,
        seq_len=cfg['data']['horizon'],
        action_dim=action_dim
    )


if __name__ == "__main__":
    main()