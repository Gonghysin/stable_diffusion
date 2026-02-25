#!/usr/bin/env python3
"""
测试推理流程（使用随机权重）
"""
import torch
from sd.pipeline import generate
from PIL import Image
import numpy as np

def test_txt2img():
    """测试文生图功能"""
    print("开始测试文生图...")

    # 导入模型类
    from sd.clip import CLIP
    from sd.encoder import VAE_Encoder
    from sd.decoder import VAE_Decoder
    from sd.diffusion import Diffusion
    from transformers import CLIPTokenizer

    # 使用随机初始化的模型（仅用于测试架构）
    print("初始化模型...")
    models = {
        "clip": CLIP(),
        "encoder": VAE_Encoder(),
        "decoder": VAE_Decoder(),
        "diffusion": Diffusion()
    }

    # 初始化 tokenizer
    try:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    except:
        print("警告: 无法加载 tokenizer，使用 None")
        tokenizer = None

    # 生成图像
    print("开始生成图像...")
    output_image = generate(
        prompt="a cat on the moon, digital art",
        uncond_prompt="",
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=20,
        models=models,
        seed=42,
        device="cpu",  # 使用 CPU 测试
        idle_device="cpu",
        tokenizer=tokenizer
    )

    print(f"生成图像尺寸: {output_image.size}")
    print("测试完成！")

    # 保存测试图像
    output_image.save("test_output.png")
    print("测试图像已保存到 test_output.png")

if __name__ == "__main__":
    test_txt2img()
