"""
Stable Diffusion 服务器端部署和测试脚本

用法:
    python setup_and_test.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(e.stdout)
        print(e.stderr)
        return False


def check_python_version():
    """检查 Python 版本"""
    print(f"✓ Python 版本: {sys.version}")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        print("错误: 需要 Python 3.8 或更高版本")
        sys.exit(1)


def install_dependencies():
    """安装依赖"""
    print("\n[1/5] 安装 Python 依赖...")

    packages = [
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "tqdm",
        "transformers",
        "huggingface_hub",
        "safetensors"
    ]

    for package in packages:
        print(f"安装 {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package, "-q"], check=True)

    print("✓ 依赖安装完成")


def create_directories():
    """创建必要的目录"""
    print("\n[2/5] 创建目录...")
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    print("✓ 目录创建完成")


def download_models():
    """下载模型权重"""
    print("\n[3/5] 下载模型权重...")
    print("这可能需要 10-20 分钟，请耐心等待...")

    try:
        from huggingface_hub import hf_hub_download

        # 下载主权重文件
        print("\n下载 Stable Diffusion v1.5 权重...")

        # 优先尝试 safetensors 格式
        try:
            print("尝试下载 safetensors 格式...")
            ckpt_path = hf_hub_download(
                repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                filename="v1-5-pruned-emaonly.safetensors",
                cache_dir="./data"
            )
            print(f"✓ 权重文件: {ckpt_path}")
        except Exception as e:
            print(f"safetensors 下载失败: {e}")
            print("尝试 ckpt 格式...")
            ckpt_path = hf_hub_download(
                repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                filename="v1-5-pruned-emaonly.ckpt",
                cache_dir="./data"
            )
            print(f"✓ 权重文件: {ckpt_path}")

        # 下载 tokenizer 文件
        print("\n下载 CLIP tokenizer...")
        vocab_path = hf_hub_download(
            repo_id="openai/clip-vit-large-patch14",
            filename="vocab.json",
            cache_dir="./data",
            resume_download=True
        )
        print(f"✓ vocab.json: {vocab_path}")

        merges_path = hf_hub_download(
            repo_id="openai/clip-vit-large-patch14",
            filename="merges.txt",
            cache_dir="./data",
            resume_download=True
        )
        print(f"✓ merges.txt: {merges_path}")

        print("\n✓ 所有文件下载完成")
        return True

    except Exception as e:
        print(f"下载失败: {e}")
        return False


def check_device():
    """检测可用设备"""
    print("\n[4/5] 检测计算设备...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  显存: {total_memory:.1f} GB")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) 可用")
            return "mps"
        else:
            print("⚠ 仅 CPU 可用（速度较慢）")
            return "cpu"

    except Exception as e:
        print(f"设备检测失败: {e}")
        return "cpu"


def run_test():
    """运行推理测试"""
    print("\n[5/5] 运行推理测试...")
    print("生成测试图像: 'a cat on the moon, digital art'")

    cmd = [
        sys.executable,
        "main.py",
        "--prompt", "a cat on the moon, digital art",
        "--output", "outputs/test_output.png",
        "--steps", "50",
        "--seed", "42",
        "--cfg-scale", "7.5",
        "--ckpt", "data/v1-5-pruned-emaonly.ckpt"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n✓ 测试完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 测试失败: {e}")
        return False


def print_usage_examples():
    """打印使用示例"""
    print("\n" + "="*60)
    print("✓ 部署和测试完成！")
    print("="*60)
    print("\n生成的图像保存在: outputs/test_output.png")
    print("\n更多使用示例:")
    print("\n# txt2img 生成")
    print('python main.py --prompt "your prompt here" --output output.png --steps 50')
    print("\n# img2img 转换")
    print('python main.py --prompt "oil painting style" --input photo.jpg --strength 0.7 --output output.png')
    print("\n# 使用负面提示词")
    print('python main.py --prompt "beautiful landscape" --uncond-prompt "blurry, low quality" --output output.png')
    print("\n# 指定设备")
    print('python main.py --prompt "..." --device cuda  # 或 mps 或 cpu')
    print()


def main():
    """主函数"""
    print("="*60)
    print("Stable Diffusion 服务器端部署脚本")
    print("="*60)

    # # 检查 Python 版本
    # check_python_version()

    # # 安装依赖
    # install_dependencies()

    # # 创建目录
    # create_directories()

    # # 下载模型
    # if not download_models():
    #     print("\n错误: 模型下载失败")
    #     sys.exit(1)

    # 检测设备
    device = check_device()

    # 运行测试
    if not run_test():
        print("\n警告: 测试失败，但部署已完成")
        print("你可以手动运行 main.py 进行测试")

    # 打印使用示例
    print_usage_examples()


if __name__ == "__main__":
    main()
