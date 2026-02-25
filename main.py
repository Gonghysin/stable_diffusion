"""
Stable Diffusion v1.5 推理入口。

用法:
    # txt2img
    python main.py \\
        --prompt "a photo of an astronaut riding a horse" \\
        --output output.png

    # img2img
    python main.py \\
        --prompt "oil painting style" \\
        --input input.jpg \\
        --strength 0.7 \\
        --output output.png
"""

import argparse
import torch
from PIL import Image

from sd.pipeline import generate
from model_loader import preload_models_from_standard_weights


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion 推理工具")

    # 必填参数
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="文本提示词"
    )

    # 可选参数
    parser.add_argument(
        "--uncond-prompt",
        type=str,
        default="",
        help="负面提示词 (用于 Classifier-Free Guidance)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="img2img 输入图像路径 (不提供则为 txt2img 模式)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="输出图像路径"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="img2img 强度 (0-1)，越大改变越多"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.5,
        help="Classifier-Free Guidance 强度"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子 (用于可复现生成)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="模型权重路径 (.ckpt 或 .safetensors)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备 (cuda/mps/cpu)，不指定则自动选择"
    )
    parser.add_argument(
        "--no-cfg",
        action="store_true",
        help="禁用 Classifier-Free Guidance"
    )

    args = parser.parse_args()

    # 确定设备
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"使用设备: {device}")

    # 加载 tokenizer
    try:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        print("已加载 CLIP tokenizer")
    except ImportError:
        print("警告: 未安装 transformers 库，无法加载 tokenizer")
        print("请运行: pip install transformers")
        return
    except Exception as e:
        print(f"警告: 加载 tokenizer 失败: {e}")
        print("将使用简化的 tokenizer")
        tokenizer = None

    # 加载模型
    if args.ckpt:
        print(f"从 {args.ckpt} 加载模型权重...")
        models = preload_models_from_standard_weights(args.ckpt, device)
    else:
        print("警告: 未指定 --ckpt 参数，将使用随机初始化的权重")
        print("生成的图像将是随机噪声")
        from sd.clip import CLIP
        from sd.encoder import VAE_Encoder
        from sd.decoder import VAE_Decoder
        from sd.diffusion import Diffusion

        models = {
            "clip": CLIP().to(device),
            "encoder": VAE_Encoder().to(device),
            "decoder": VAE_Decoder().to(device),
            "diffusion": Diffusion().to(device),
        }

    # 加载输入图像 (img2img 模式)
    input_image = None
    if args.input:
        try:
            input_image = Image.open(args.input)
            print(f"已加载输入图像: {args.input}")
        except Exception as e:
            print(f"错误: 无法加载输入图像: {e}")
            return

    # 生成图像
    print(f"\n开始生成...")
    print(f"提示词: {args.prompt}")
    if args.uncond_prompt:
        print(f"负面提示词: {args.uncond_prompt}")
    print(f"步数: {args.steps}")
    print(f"CFG 强度: {args.cfg_scale}")
    if args.seed is not None:
        print(f"随机种子: {args.seed}")

    try:
        output_image = generate(
            prompt=args.prompt,
            uncond_prompt=args.uncond_prompt,
            input_image=input_image,
            strength=args.strength,
            do_cfg=not args.no_cfg,
            cfg_scale=args.cfg_scale,
            sampler_name="ddpm",
            n_inference_steps=args.steps,
            models=models,
            seed=args.seed,
            device=device,
            idle_device="cpu" if device.type != "cpu" else None,
            tokenizer=tokenizer,
        )

        # 保存输出
        output_image.save(args.output)
        print(f"\n✓ 图像已保存到: {args.output}")

    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
