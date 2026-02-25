"""
模型权重加载工具。

支持从 Hugging Face 下载和加载 Stable Diffusion v1.5 权重。
"""

import torch
from torch import nn
from pathlib import Path
import os
import re

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


def convert_clip_text_model_state_dict(state_dict):
    """
    转换 CLIP 文本模型的 state_dict。
    
    官方格式: cond_stage_model.transformer.text_model.*
    我们格式: embedding.*, layers.*, layernorm.*
    """
    converted = {}
    
    for key, value in state_dict.items():
        if not key.startswith("cond_stage_model.transformer.text_model."):
            continue
        
        # 移除前缀
        new_key = key.replace("cond_stage_model.transformer.text_model.", "")
        
        # embeddings.token_embedding -> embedding.token_embedding
        new_key = new_key.replace("embeddings.token_embedding", "embedding.token_embedding")
        # embeddings.position_embedding -> embedding.position_embedding  
        new_key = new_key.replace("embeddings.position_embedding", "embedding.position_embedding")
        
        # encoder.layers.N -> layers.N
        new_key = new_key.replace("encoder.layers.", "layers.")
        
        # self_attn -> attention (但保留 in_proj 用于合并的 QKV)
        new_key = new_key.replace("self_attn.", "attention.")
        
        # mlp.fc1 -> linear_1, mlp.fc2 -> linear_2
        new_key = new_key.replace("mlp.fc1", "linear_1")
        new_key = new_key.replace("mlp.fc2", "linear_2")
        
        # layer_norm1 -> layernorm_1, layer_norm2 -> layernorm_2
        new_key = new_key.replace("layer_norm1", "layernorm_1")
        new_key = new_key.replace("layer_norm2", "layernorm_2")
        
        # final_layer_norm -> layernorm
        new_key = new_key.replace("final_layer_norm", "layernorm")
        
        converted[new_key] = value
    
    return converted


def convert_unet_state_dict(state_dict):
    """
    转换 U-Net 的 state_dict。
    
    官方格式: model.diffusion_model.*
    我们格式: time_embedding.*, unet.encoders.*, unet.bottleneck.*, unet.decoder.*, final.*
    """
    converted = {}
    
    for key, value in state_dict.items():
        if not key.startswith("model.diffusion_model."):
            continue
        
        # 移除前缀
        new_key = key.replace("model.diffusion_model.", "")
        
        # Time embedding: time_embed.0 -> time_embedding.linear_1, time_embed.2 -> time_embedding.linear_2
        if new_key.startswith("time_embed."):
            new_key = new_key.replace("time_embed.0.", "time_embedding.linear_1.")
            new_key = new_key.replace("time_embed.2.", "time_embedding.linear_2.")
            converted[new_key] = value
            continue
        
        # Input blocks -> encoders
        if new_key.startswith("input_blocks."):
            match = re.match(r"input_blocks\.(\d+)\.(.*)", new_key)
            if match:
                block_idx = match.group(1)
                rest = match.group(2)
                new_key = f"unet.encoders.{block_idx}.{rest}"
                new_key = _convert_resblock_attention_keys(new_key)
                converted[new_key] = value
            continue
        
        # Middle block -> bottleneck
        if new_key.startswith("middle_block."):
            match = re.match(r"middle_block\.(\d+)\.(.*)", new_key)
            if match:
                block_idx = match.group(1)
                rest = match.group(2)
                new_key = f"unet.bottleneck.{block_idx}.{rest}"
                new_key = _convert_resblock_attention_keys(new_key)
                converted[new_key] = value
            continue
        
        # Output blocks -> decoder
        if new_key.startswith("output_blocks."):
            match = re.match(r"output_blocks\.(\d+)\.(.*)", new_key)
            if match:
                block_idx = match.group(1)
                rest = match.group(2)
                new_key = f"unet.decoder.{block_idx}.{rest}"
                new_key = _convert_resblock_attention_keys(new_key)
                converted[new_key] = value
            continue
        
        # Output layer: out.0 -> final.groupnorm, out.2 -> final.conv
        if new_key.startswith("out."):
            new_key = new_key.replace("out.0.", "final.groupnorm.")
            new_key = new_key.replace("out.2.", "final.conv.")
            converted[new_key] = value
            continue
    
    return converted


def _convert_resblock_attention_keys(key: str) -> str:
    """
    转换 ResBlock 和 Attention 块内部的 key 名称。
    
    ResBlock 官方格式:
        in_layers.0 -> groupnorm_feature
        in_layers.2 -> conv_feature
        emb_layers.1 -> linear_time
        out_layers.0 -> groupnorm_merged
        out_layers.3 -> conv_merged
        skip_connection -> residual_layer
    
    Attention 官方格式:
        norm -> groupnorm
        proj_in -> conv_input
        transformer_blocks.0.norm1 -> layernorm_1
        transformer_blocks.0.attn1.to_q -> attention_1.q_proj
        transformer_blocks.0.attn1.to_k -> attention_1.k_proj
        transformer_blocks.0.attn1.to_v -> attention_1.v_proj
        transformer_blocks.0.attn1.to_out.0 -> attention_1.out_proj
        transformer_blocks.0.norm2 -> layernorm_2
        transformer_blocks.0.attn2 -> attention_2 (cross attention)
        transformer_blocks.0.norm3 -> layernorm_3
        transformer_blocks.0.ff.net.0.proj -> linear_geglu_1
        transformer_blocks.0.ff.net.2 -> linear_geglu_2
        proj_out -> conv_output
    """
    # ResBlock 映射
    key = key.replace(".in_layers.0.", ".groupnorm_feature.")
    key = key.replace(".in_layers.2.", ".conv_feature.")
    key = key.replace(".emb_layers.1.", ".linear_time.")
    key = key.replace(".out_layers.0.", ".groupnorm_merged.")
    key = key.replace(".out_layers.3.", ".conv_merged.")
    key = key.replace(".skip_connection.", ".residual_layer.")
    
    # Attention 块映射
    key = key.replace(".norm.", ".groupnorm.")
    key = key.replace(".proj_in.", ".conv_input.")
    
    # Transformer blocks
    key = key.replace(".transformer_blocks.0.norm1.", ".layernorm_1.")
    key = key.replace(".transformer_blocks.0.norm2.", ".layernorm_2.")
    key = key.replace(".transformer_blocks.0.norm3.", ".layernorm_3.")
    
    # Self attention (attn1)
    key = key.replace(".transformer_blocks.0.attn1.to_q.", ".attention_1.q_proj.")
    key = key.replace(".transformer_blocks.0.attn1.to_k.", ".attention_1.k_proj.")
    key = key.replace(".transformer_blocks.0.attn1.to_v.", ".attention_1.v_proj.")
    key = key.replace(".transformer_blocks.0.attn1.to_out.0.", ".attention_1.out_proj.")
    
    # Cross attention (attn2)
    key = key.replace(".transformer_blocks.0.attn2.to_q.", ".attention_2.q_proj.")
    key = key.replace(".transformer_blocks.0.attn2.to_k.", ".attention_2.k_proj.")
    key = key.replace(".transformer_blocks.0.attn2.to_v.", ".attention_2.v_proj.")
    key = key.replace(".transformer_blocks.0.attn2.to_out.0.", ".attention_2.out_proj.")
    
    # Feed-forward (GeGLU)
    key = key.replace(".transformer_blocks.0.ff.net.0.proj.", ".linear_geglu_1.")
    key = key.replace(".transformer_blocks.0.ff.net.2.", ".linear_geglu_2.")
    
    key = key.replace(".proj_out.", ".conv_output.")
    
    return key


def convert_vae_encoder_state_dict(state_dict):
    """
    转换 VAE Encoder 的 state_dict。
    
    由于 VAE_Encoder 使用 nn.Sequential，需要映射到数字索引。
    这个映射比较复杂，需要根据实际的 Sequential 结构来确定。
    """
    converted = {}
    
    for key, value in state_dict.items():
        if not key.startswith("first_stage_model.encoder."):
            continue
        
        # 移除前缀
        new_key = key.replace("first_stage_model.encoder.", "")
        
        # 这里需要根据 VAE_Encoder 的实际结构进行映射
        # 由于结构复杂，我们保留原始的层次结构名称
        # PyTorch 的 load_state_dict 会尝试匹配
        
        converted[new_key] = value
    
    return converted


def convert_vae_decoder_state_dict(state_dict):
    """
    转换 VAE Decoder 的 state_dict。
    """
    converted = {}
    
    for key, value in state_dict.items():
        if not key.startswith("first_stage_model.decoder."):
            continue
        
        # 移除前缀
        new_key = key.replace("first_stage_model.decoder.", "")
        
        converted[new_key] = value
    
    return converted


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

    # 转换权重
    print("转换权重格式...")
    
    clip_state_dict = convert_clip_text_model_state_dict(state_dict)
    diffusion_state_dict = convert_unet_state_dict(state_dict)
    encoder_state_dict = convert_vae_encoder_state_dict(state_dict)
    decoder_state_dict = convert_vae_decoder_state_dict(state_dict)
    
    # 加载 CLIP 权重
    print("\n加载 CLIP 权重...")
    if clip_state_dict:
        missing, unexpected = models["clip"].load_state_dict(clip_state_dict, strict=False)
        total_params = len(models["clip"].state_dict())
        loaded_params = len(clip_state_dict)
        matched_params = loaded_params - len(unexpected)
        
        print(f"✓ CLIP: 匹配 {matched_params}/{total_params} 个参数")
        if missing:
            print(f"  缺失 {len(missing)} 个参数 (将使用随机初始化)")
            if len(missing) <= 10:
                for m in missing:
                    print(f"    - {m}")
        if unexpected:
            print(f"  忽略 {len(unexpected)} 个不匹配的参数")
            if len(unexpected) <= 10:
                for u in unexpected:
                    print(f"    - {u}")
    else:
        print("警告: 未找到 CLIP 权重")

    # 加载 VAE Encoder 权重
    print("\n加载 VAE Encoder 权重...")
    if encoder_state_dict:
        missing, unexpected = models["encoder"].load_state_dict(encoder_state_dict, strict=False)
        total_params = len(models["encoder"].state_dict())
        loaded_params = len(encoder_state_dict)
        matched_params = loaded_params - len(unexpected)
        
        print(f"✓ Encoder: 匹配 {matched_params}/{total_params} 个参数")
        if missing:
            print(f"  缺失 {len(missing)} 个参数")
        if unexpected:
            print(f"  忽略 {len(unexpected)} 个不匹配的参数")
    else:
        print("警告: 未找到 VAE Encoder 权重")

    # 加载 VAE Decoder 权重
    print("\n加载 VAE Decoder 权重...")
    if decoder_state_dict:
        missing, unexpected = models["decoder"].load_state_dict(decoder_state_dict, strict=False)
        total_params = len(models["decoder"].state_dict())
        loaded_params = len(decoder_state_dict)
        matched_params = loaded_params - len(unexpected)
        
        print(f"✓ Decoder: 匹配 {matched_params}/{total_params} 个参数")
        if missing:
            print(f"  缺失 {len(missing)} 个参数")
        if unexpected:
            print(f"  忽略 {len(unexpected)} 个不匹配的参数")
    else:
        print("警告: 未找到 VAE Decoder 权重")

    # 加载 U-Net (Diffusion) 权重
    print("\n加载 U-Net 权重...")
    if diffusion_state_dict:
        missing, unexpected = models["diffusion"].load_state_dict(diffusion_state_dict, strict=False)
        total_params = len(models["diffusion"].state_dict())
        loaded_params = len(diffusion_state_dict)
        matched_params = loaded_params - len(unexpected)
        
        print(f"✓ Diffusion: 匹配 {matched_params}/{total_params} 个参数")
        if missing:
            print(f"  缺失 {len(missing)} 个参数 (将使用随机初始化)")
            if len(missing) <= 10:
                for m in missing[:10]:
                    print(f"    - {m}")
                if len(missing) > 10:
                    print(f"    ... 还有 {len(missing) - 10} 个")
        if unexpected:
            print(f"  忽略 {len(unexpected)} 个不匹配的参数")
            if len(unexpected) <= 10:
                for u in unexpected[:10]:
                    print(f"    - {u}")
                if len(unexpected) > 10:
                    print(f"    ... 还有 {len(unexpected) - 10} 个")
    else:
        print("警告: 未找到 U-Net 权重")

    print("\n✓ 权重加载完成")

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
