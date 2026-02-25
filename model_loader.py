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
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # 获取 state_dict
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

    # 加载 CLIP 权重
    print("加载 CLIP 权重...")
    clip_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("cond_stage_model.transformer.text_model"):
            new_key = key.replace("cond_stage_model.transformer.text_model.", "")
            clip_state_dict[new_key] = value

    if clip_state_dict:
        try:
            models["clip"].load_state_dict(clip_state_dict, strict=False)
            print(f"✓ CLIP 权重已加载 ({len(clip_state_dict)} 个参数)")
        except Exception as e:
            print(f"警告: CLIP 权重加载部分失败: {e}")
    else:
        print("警告: 未找到 CLIP 权重")

    # 加载 VAE Encoder 权重
    print("加载 VAE Encoder 权重...")
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("first_stage_model.encoder"):
            new_key = key.replace("first_stage_model.encoder.", "")
            encoder_state_dict[new_key] = value

    if encoder_state_dict:
        try:
            models["encoder"].load_state_dict(encoder_state_dict, strict=False)
            print(f"✓ VAE Encoder 权重已加载 ({len(encoder_state_dict)} 个参数)")
        except Exception as e:
            print(f"警告: VAE Encoder 权重加载部分失败: {e}")
    else:
        print("警告: 未找到 VAE Encoder 权重")

    # 加载 VAE Decoder 权重
    print("加载 VAE Decoder 权重...")
    decoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("first_stage_model.decoder"):
            new_key = key.replace("first_stage_model.decoder.", "")
            decoder_state_dict[new_key] = value

    if decoder_state_dict:
        try:
            models["decoder"].load_state_dict(decoder_state_dict, strict=False)
            print(f"✓ VAE Decoder 权重已加载 ({len(decoder_state_dict)} 个参数)")
        except Exception as e:
            print(f"警告: VAE Decoder 权重加载部分失败: {e}")
    else:
        print("警告: 未找到 VAE Decoder 权重")

    # 加载 U-Net (Diffusion) 权重
    print("加载 U-Net 权重...")
    diffusion_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.diffusion_model"):
            new_key = key.replace("model.diffusion_model.", "")
            diffusion_state_dict[new_key] = value

    if diffusion_state_dict:
        try:
            models["diffusion"].load_state_dict(diffusion_state_dict, strict=False)
            print(f"✓ U-Net 权重已加载 ({len(diffusion_state_dict)} 个参数)")
        except Exception as e:
            print(f"警告: U-Net 权重加载部分失败: {e}")
    else:
        print("警告: 未找到 U-Net 权重")

    print("✓ 所有模型权重加载完成")

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
