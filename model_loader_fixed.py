"""
模型权重加载工具。

支持从 Hugging Face 下载和加载 Stable Diffusion v1.5 权重。
"""

import torch
from torch import nn
from pathlib import Path
import os

from sd.clip import CLIP
from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
from sd.diffusion import Diffusion


def download_from_huggingface(model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", cache_dir: str = "./data"):
    """
    从 Hugging Face 下载模型权重。

    参数:
        model_id: Hugging Face 模型 ID
        cache_dir: 缓存目录

    返回:
        权重文件路径
    """
    try:
        from huggingface_hub import hf_hub_download

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        print(f"从 Hugging Face 下载模型: {model_id}")
        print("这可能需要几分钟时间...")

        # 优先尝试下载 safetensors 格式（更安全、更快）
        try:
            print("尝试下载 safetensors 格式...")
            ckpt_path = hf_hub_download(
                repo_id=model_id,
                filename="v1-5-pruned-emaonly.safetensors",
                cache_dir=cache_dir
            )
            print(f"✓ 权重文件已下载到: {ckpt_path}")
            return ckpt_path
        except Exception as e:
            print(f"safetensors 下载失败，尝试 ckpt 格式...")

        # 备用：下载 ckpt 格式
        ckpt_path = hf_hub_download(
            repo_id=model_id,
            filename="v1-5-pruned-emaonly.ckpt",
            cache_dir=cache_dir
        )

        print(f"✓ 权重文件已下载到: {ckpt_path}")
        return ckpt_path

    except ImportError:
        print("错误: 未安装 huggingface_hub 库")
        print("请运行: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def convert_clip_keys(official_key: str) -> str:
    """
    将官方 CLIP 权重 key 转换为我们模型的 key。
    
    官方格式: cond_stage_model.transformer.text_model.embeddings.token_embedding.weight
    我们格式: embedding.token_embedding.weight
    """
    if not official_key.startswith("cond_stage_model.transformer.text_model."):
        return None
    
    key = official_key.replace("cond_stage_model.transformer.text_model.", "")
    
    # embeddings -> embedding
    key = key.replace("embeddings.token_embedding", "embedding.token_embedding")
    key = key.replace("embeddings.position_embedding", "embedding.position_embedding")
    
    # encoder.layers -> layers
    key = key.replace("encoder.layers", "layers")
    
    # layer_norm -> layernorm
    key = key.replace("self_attn.q_proj", "attention.q_proj")
    key = key.replace("self_attn.k_proj", "attention.k_proj")
    key = key.replace("self_attn.v_proj", "attention.v_proj")
    key = key.replace("self_attn.out_proj", "attention.out_proj")
    
    # Handle combined QKV projection
    if "attention.in_proj" in key:
        return key
    
    # mlp -> linear
    key = key.replace("mlp.fc1", "linear_1")
    key = key.replace("mlp.fc2", "linear_2")
    
    # final_layer_norm -> layernorm
    key = key.replace("final_layer_norm", "layernorm")
    
    return key


def convert_diffusion_keys(official_key: str) -> str:
    """
    将官方 U-Net 权重 key 转换为我们模型的 key。
    
    官方格式: model.diffusion_model.time_embed.0.weight
    我们格式: time_embedding.linear_1.weight
    """
    if not official_key.startswith("model.diffusion_model."):
        return None
    
    key = official_key.replace("model.diffusion_model.", "")
    
    # Time embedding
    if key.startswith("time_embed."):
        key = key.replace("time_embed.0", "time_embedding.linear_1")
        key = key.replace("time_embed.2", "time_embedding.linear_2")
        return key
    
    # Input blocks -> encoders
    if key.startswith("input_blocks."):
        parts = key.split(".")
        block_idx = int(parts[1])
        rest = ".".join(parts[2:])
        
        # Map block structure
        new_key = f"unet.encoders.{block_idx}.{rest}"
        new_key = _convert_resblock_attention_keys(new_key)
        return new_key
    
    # Middle block -> bottleneck
    if key.startswith("middle_block."):
        parts = key.split(".")
        block_idx = int(parts[1])
        rest = ".".join(parts[2:])
        
        new_key = f"unet.bottleneck.{block_idx}.{rest}"
        new_key = _convert_resblock_attention_keys(new_key)
        return new_key
    
    # Output blocks -> decoder
    if key.startswith("output_blocks."):
        parts = key.split(".")
        block_idx = int(parts[1])
        rest = ".".join(parts[2:])
        
        new_key = f"unet.decoder.{block_idx}.{rest}"
        new_key = _convert_resblock_attention_keys(new_key)
        return new_key
    
    # Output layer
    if key.startswith("out."):
        key = key.replace("out.0", "final.groupnorm")
        key = key.replace("out.2", "final.conv")
        return key
    
    return None


def _convert_resblock_attention_keys(key: str) -> str:
    """
    转换 ResBlock 和 Attention 块内部的 key 名称。
    """
    # ResBlock 映射
    key = key.replace("in_layers.0", "groupnorm_feature")
    key = key.replace("in_layers.2", "conv_feature")
    key = key.replace("emb_layers.1", "linear_time")
    key = key.replace("out_layers.0", "groupnorm_merged")
    key = key.replace("out_layers.3", "conv_merged")
    key = key.replace("skip_connection", "residual_layer")
    
    # Attention 块映射
    key = key.replace("norm.", "groupnorm.")
    key = key.replace("proj_in", "conv_input")
    key = key.replace("transformer_blocks.0.norm1", "layernorm_1")
    key = key.replace("transformer_blocks.0.attn1", "attention_1")
    key = key.replace("transformer_blocks.0.norm2", "layernorm_2")
    key = key.replace("transformer_blocks.0.attn2", "attention_2")
    key = key.replace("transformer_blocks.0.norm3", "layernorm_3")
    key = key.replace("transformer_blocks.0.ff", "linear_geglu")
    key = key.replace("proj_out", "conv_output")
    
    # Attention 内部映射
    key = key.replace("to_q", "q_proj")
    key = key.replace("to_k", "k_proj")
    key = key.replace("to_v", "v_proj")
    key = key.replace("to_out.0", "out_proj")
    
    return key


def convert_vae_encoder_keys(official_key: str) -> str:
    """
    将官方 VAE Encoder 权重 key 转换为我们模型的 key。
    
    官方格式: first_stage_model.encoder.conv_in.weight
    我们格式: 0.weight (Sequential 索引)
    """
    if not official_key.startswith("first_stage_model.encoder."):
        return None
    
    key = official_key.replace("first_stage_model.encoder.", "")
    
    # 这里需要根据 VAE_Encoder 的 Sequential 结构进行映射
    # 由于 VAE_Encoder 使用 nn.Sequential，需要映射到正确的索引
    
    # conv_in -> 0
    if key.startswith("conv_in."):
        return key.replace("conv_in.", "0.")
    
    # down blocks
    if key.startswith("down."):
        # 需要根据具体的 down block 结构进行映射
        # 这里简化处理，保留原始 key 结构
        return key
    
    # mid block
    if key.startswith("mid."):
        return key
    
    # norm_out and conv_out
    if key.startswith("norm_out."):
        return key
    if key.startswith("conv_out."):
        return key
    
    return key


def convert_vae_decoder_keys(official_key: str) -> str:
    """
    将官方 VAE Decoder 权重 key 转换为我们模型的 key。
    """
    if not official_key.startswith("first_stage_model.decoder."):
        return None
    
    key = official_key.replace("first_stage_model.decoder.", "")
    
    # 类似 encoder 的映射逻辑
    if key.startswith("conv_in."):
        return key.replace("conv_in.", "1.")
    
    return key


def load_from_standard_weights(
    ckpt_path: str,
    device: torch.device
) -> dict:
    """
    从 .ckpt 文件加载权重。

    参数:
        ckpt_path: 权重文件路径
        device: 目标设备 (cpu / cuda / mps)

    返回:
        {
            "clip": CLIP 实例 (已加载权重),
            "encoder": VAE_Encoder 实例 (已加载权重),
            "decoder": VAE_Decoder 实例 (已加载权重),
            "diffusion": Diffusion 实例 (已加载权重),
        }
    """
    print(f"加载权重文件: {ckpt_path}")

    # 检查文件是否存在
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")

    # 加载 checkpoint
    print("正在加载 checkpoint...")

    # 检查文件格式
    if ckpt_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path, device="cpu")
            print(f"✓ SafeTensors 文件已加载，包含 {len(state_dict)} 个参数")
        except ImportError:
            print("错误: 未安装 safetensors 库")
            print("运行: pip install safetensors")
            raise
    else:
        # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含 PyTorch Lightning 的 checkpoint
        try:
            import pytorch_lightning
            torch.serialization.add_safe_globals([pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint])
        except ImportError:
            pass

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        print(f"✓ Checkpoint 已加载，包含 {len(state_dict)} 个参数")

    # 实例化模型
    print("初始化模型...")
    models = {
        "clip": CLIP().to(device),
        "encoder": VAE_Encoder().to(device),
        "decoder": VAE_Decoder().to(device),
        "diffusion": Diffusion().to(device),
    }

    # 调试：打印前 20 个 keys 来检查前缀
    print("\n调试信息 - 权重文件中的前 20 个 keys:")
    for i, key in enumerate(list(state_dict.keys())[:20]):
        print(f"  {i+1}. {key}")
    print()

    # 转换并加载权重
    print("转换并加载权重...")
    
    # 统计匹配的权重数量
    matched_counts = {
        "clip": 0,
        "encoder": 0,
        "decoder": 0,
        "diffusion": 0
    }
    
    # 获取模型的参数名称集合
    model_params = {
        "clip": set(models["clip"].state_dict().keys()),
        "encoder": set(models["encoder"].state_dict().keys()),
        "decoder": set(models["decoder"].state_dict().keys()),
        "diffusion": set(models["diffusion"].state_dict().keys())
    }
    
    # 转换所有 keys
    converted_weights = {
        "clip": {},
        "encoder": {},
        "decoder": {},
        "diffusion": {}
    }
    
    for official_key, value in state_dict.items():
        # 尝试转换为 CLIP key
        clip_key = convert_clip_keys(official_key)
        if clip_key and clip_key in model_params["clip"]:
            converted_weights["clip"][clip_key] = value
            matched_counts["clip"] += 1
        
        # 尝试转换为 Diffusion key
        diff_key = convert_diffusion_keys(official_key)
        if diff_key and diff_key in model_params["diffusion"]:
            converted_weights["diffusion"][diff_key] = value
            matched_counts["diffusion"] += 1
        
        # 尝试转换为 VAE Encoder key
        enc_key = convert_vae_encoder_keys(official_key)
        if enc_key and enc_key in model_params["encoder"]:
            converted_weights["encoder"][enc_key] = value
            matched_counts["encoder"] += 1
        
        # 尝试转换为 VAE Decoder key
        dec_key = convert_vae_decoder_keys(official_key)
        if dec_key and dec_key in model_params["decoder"]:
            converted_weights["decoder"][dec_key] = value
            matched_counts["decoder"] += 1
    
    # 加载转换后的权重
    print("\n加载 CLIP 权重...")
    if converted_weights["clip"]:
        missing, unexpected = models["clip"].load_state_dict(converted_weights["clip"], strict=False)
        print(f"✓ CLIP: 匹配 {matched_counts['clip']}/{len(model_params['clip'])} 个参数")
        if missing:
            print(f"  缺失: {len(missing)} 个参数")
        if unexpected:
            print(f"  多余: {len(unexpected)} 个参数")
    else:
        print("警告: 未找到 CLIP 权重")

    print("\n加载 VAE Encoder 权重...")
    if converted_weights["encoder"]:
        missing, unexpected = models["encoder"].load_state_dict(converted_weights["encoder"], strict=False)
        print(f"✓ Encoder: 匹配 {matched_counts['encoder']}/{len(model_params['encoder'])} 个参数")
        if missing:
            print(f"  缺失: {len(missing)} 个参数")
        if unexpected:
            print(f"  多余: {len(unexpected)} 个参数")
    else:
        print("警告: 未找到 VAE Encoder 权重")

    print("\n加载 VAE Decoder 权重...")
    if converted_weights["decoder"]:
        missing, unexpected = models["decoder"].load_state_dict(converted_weights["decoder"], strict=False)
        print(f"✓ Decoder: 匹配 {matched_counts['decoder']}/{len(model_params['decoder'])} 个参数")
        if missing:
            print(f"  缺失: {len(missing)} 个参数")
        if unexpected:
            print(f"  多余: {len(unexpected)} 个参数")
    else:
        print("警告: 未找到 VAE Decoder 权重")

    print("\n加载 U-Net 权重...")
    if converted_weights["diffusion"]:
        missing, unexpected = models["diffusion"].load_state_dict(converted_weights["diffusion"], strict=False)
        print(f"✓ Diffusion: 匹配 {matched_counts['diffusion']}/{len(model_params['diffusion'])} 个参数")
        if missing:
            print(f"  缺失: {len(missing)} 个参数")
        if unexpected:
            print(f"  多余: {len(unexpected)} 个参数")
    else:
        print("警告: 未找到 U-Net 权重")

    print("\n✓ 权重加载完成")
    print(f"\n总结:")
    print(f"  CLIP: {matched_counts['clip']}/{len(model_params['clip'])} 参数")
    print(f"  Encoder: {matched_counts['encoder']}/{len(model_params['encoder'])} 参数")
    print(f"  Decoder: {matched_counts['decoder']}/{len(model_params['decoder'])} 参数")
    print(f"  Diffusion: {matched_counts['diffusion']}/{len(model_params['diffusion'])} 参数")

    return models


def preload_models_from_standard_weights(ckpt_path: str, device: torch.device):
    """
    预加载模型权重的便捷函数。

    参数:
        ckpt_path: 权重文件路径，如果不存在会尝试从 Hugging Face 下载
        device: 目标设备

    返回:
        模型字典
    """
    # 如果文件不存在，尝试下载
    if not os.path.exists(ckpt_path):
        print(f"权重文件不存在: {ckpt_path}")
        print("尝试从 Hugging Face 下载...")

        downloaded_path = download_from_huggingface()
        if downloaded_path:
            ckpt_path = downloaded_path
        else:
            raise FileNotFoundError(f"无法获取权重文件: {ckpt_path}")

    return load_from_standard_weights(ckpt_path, device)
