import torch
import numpy as np
from PIL import Image
from typing import Optional
from tqdm import tqdm

from sd.ddpm import DDPMSampler
from sd.clip import CLIP
from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
from sd.diffusion import Diffusion


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def rescale(x, old_range, new_range, clamp=False):
    """
    线性映射工具函数。

    将 x 从 old_range 映射到 new_range。
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    """
    将标量时间步转换为正弦位置编码向量。

    参数:
        timestep: 整数时间步 (0-999)

    返回:
        shape (1, 320) 的张量
    """
    # 320 维 = 160 个频率 * 2 (cos + sin)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def generate(
    prompt: str,
    uncond_prompt: str = "",
    input_image: Optional[Image.Image] = None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    sampler_name: str = "ddpm",
    n_inference_steps: int = 50,
    models: dict = {},
    seed: Optional[int] = None,
    device: str = None,
    idle_device: str = None,
    tokenizer=None,
):
    """
    Stable Diffusion 推理 Pipeline。

    参数:
        prompt: 正面提示词
        uncond_prompt: 负面提示词 (用于 CFG)
        input_image: img2img 的输入图像，None 表示 txt2img
        strength: img2img 强度 (0-1)，越大改变越多
        do_cfg: 是否使用 Classifier-Free Guidance
        cfg_scale: CFG 强度，典型值 7.5
        sampler_name: 采样器名称 (目前只支持 "ddpm")
        n_inference_steps: 推理步数
        models: 模型字典 {"clip": ..., "encoder": ..., "decoder": ..., "diffusion": ...}
        seed: 随机种子
        device: 计算设备
        idle_device: 模型不用时移到的设备 (省显存)
        tokenizer: CLIP tokenizer

    返回:
        PIL.Image 对象
    """
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # 初始化随机数生成器
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # 获取模型
        clip = models["clip"]
        encoder = models.get("encoder")
        decoder = models["decoder"]
        diffusion = models["diffusion"]

        # 1. CLIP 文本编码
        # (Batch_Size, Seq_Len, Dim)
        cond_tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77
        ).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        cond_context = clip(cond_tokens)

        if do_cfg:
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            # (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            context = cond_context

        to_idle(clip)

        # 2. 创建采样器
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator=generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # 3. 准备初始 latent
        if input_image:
            # img2img 模式
            input_image = input_image.resize((WIDTH, HEIGHT))
            input_image_array = np.array(input_image)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_array, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # 归一化到 [-1, 1]
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # 编码到 latent 空间
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            # 根据 strength 决定从哪一步开始去噪
            sampler.set_inference_timesteps(n_inference_steps)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # txt2img 模式：从纯噪声开始
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # 4. 去噪循环
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # 扩展 latent 以同时处理条件和无条件
                model_input = model_input.repeat(2, 1, 1, 1)

            # U-Net 预测噪声
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # 去噪一步
            latents = sampler.step(model_output, timestep, latents)

        to_idle(diffusion)

        # 5. VAE 解码
        images = decoder(latents)
        to_idle(decoder)

        # 6. 后处理
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()

        return Image.fromarray(images[0])
