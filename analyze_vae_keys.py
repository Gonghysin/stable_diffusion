"""
分析 VAE 权重的 key 结构
"""
import torch

ckpt_path = "data/v1-5-pruned-emaonly.ckpt"

print("加载 checkpoint...")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["state_dict"]

print("\n=== VAE Encoder Keys ===")
encoder_keys = [k for k in state_dict.keys() if k.startswith("first_stage_model.encoder.")]
for i, key in enumerate(sorted(encoder_keys)[:30]):
    print(f"{i+1}. {key}")

print(f"\n总共 {len(encoder_keys)} 个 encoder keys")

print("\n=== VAE Decoder Keys ===")
decoder_keys = [k for k in state_dict.keys() if k.startswith("first_stage_model.decoder.")]
for i, key in enumerate(sorted(decoder_keys)[:30]):
    print(f"{i+1}. {key}")

print(f"\n总共 {len(decoder_keys)} 个 decoder keys")
