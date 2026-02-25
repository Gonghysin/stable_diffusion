#!/usr/bin/env python3
"""
快速测试脚本 - 验证安装是否成功

用法:
    python quick_test.py
"""

import sys
import subprocess


def test_imports():
    """测试必要的库是否已安装"""
    print("测试 Python 库...")

    required = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "tqdm": "tqdm",
        "transformers": "Transformers",
        "huggingface_hub": "Hugging Face Hub"
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (未安装)")
            missing.append(name)

    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("运行: pip install " + " ".join(missing))
        return False

    print("\n✓ 所有依赖已安装")
    return True


def test_device():
    """测试可用设备"""
    print("\n测试计算设备...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA 可用")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) 可用")
        else:
            print("⚠ 仅 CPU 可用")

        return True
    except Exception as e:
        print(f"✗ 设备检测失败: {e}")
        return False


def test_model_files():
    """检查模型文件"""
    print("\n检查模型文件...")

    import os
    from pathlib import Path

    data_dir = Path("data")
    if not data_dir.exists():
        print("⚠ data/ 目录不存在")
        print("首次运行会自动下载模型")
        return True

    # 检查是否有 .ckpt 文件
    ckpt_files = list(data_dir.rglob("*.ckpt"))
    if ckpt_files:
        print(f"✓ 找到权重文件: {ckpt_files[0]}")
        return True
    else:
        print("⚠ 未找到权重文件")
        print("首次运行会自动下载")
        return True


def run_quick_test():
    """运行快速生成测试"""
    print("\n运行快速测试...")
    print("生成图像: 'a red apple'")
    print("使用 10 步快速采样...")

    cmd = [
        sys.executable,
        "main.py",
        "--prompt", "a red apple",
        "--output", "quick_test.png",
        "--steps", "10",
        "--seed", "42"
    ]

    try:
        subprocess.run(cmd, check=True, timeout=600)
        print("\n✓ 测试成功！")
        print("生成的图像: quick_test.png")
        return True
    except subprocess.TimeoutExpired:
        print("\n⚠ 测试超时（可能是 CPU 模式）")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 测试失败: {e}")
        return False


def main():
    print("="*60)
    print("Stable Diffusion 快速测试")
    print("="*60)

    # 测试导入
    if not test_imports():
        sys.exit(1)

    # 测试设备
    test_device()

    # 检查模型文件
    test_model_files()

    # 询问是否运行测试
    print("\n" + "="*60)
    response = input("是否运行快速生成测试？(y/n): ").strip().lower()

    if response == 'y':
        run_quick_test()
    else:
        print("\n跳过生成测试")
        print("你可以手动运行: python main.py --prompt \"test\" --output test.png")

    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
