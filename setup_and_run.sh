#!/bin/bash
# Stable Diffusion 服务器端部署和测试脚本
# 用法: bash setup_and_run.sh

set -e  # 遇到错误立即退出

echo "========================================="
echo "Stable Diffusion 服务器端部署脚本"
echo "========================================="
echo ""

# 1. 检查 Python 版本
echo "[1/6] 检查 Python 环境..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python 版本: $python_version"
echo ""

# 2. 安装依赖
echo "[2/6] 安装 Python 依赖..."
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip3 install numpy pillow tqdm transformers huggingface_hub safetensors
echo "✓ 依赖安装完成"
echo ""

# 3. 创建数据目录
echo "[3/6] 创建数据目录..."
mkdir -p data
mkdir -p outputs
echo "✓ 目录创建完成"
echo ""

# 4. 下载模型权重和 tokenizer
echo "[4/6] 下载模型权重..."
echo "这可能需要 10-20 分钟，请耐心等待..."

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

# 下载主权重文件
print("下载 Stable Diffusion v1.5 权重...")
ckpt_path = hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned-emaonly.ckpt",
    cache_dir="./data",
    resume_download=True
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
EOF

echo ""

# 5. 检测可用设备
echo "[5/6] 检测计算设备..."
python3 << 'EOF'
import torch

if torch.cuda.is_available():
    print(f"✓ CUDA 可用")
    print(f"  GPU 数量: {torch.cuda.device_count()}")
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif torch.backends.mps.is_available():
    print("✓ MPS (Apple Silicon) 可用")
else:
    print("⚠ 仅 CPU 可用（速度较慢）")
EOF

echo ""

# 6. 运行测试
echo "[6/6] 运行推理测试..."
echo "生成测试图像: 'a cat on the moon, digital art'"
echo ""

python3 main.py \
    --prompt "a cat on the moon, digital art" \
    --output outputs/test_output.png \
    --steps 20 \
    --seed 42 \
    --cfg-scale 7.5

echo ""
echo "========================================="
echo "✓ 部署和测试完成！"
echo "========================================="
echo ""
echo "生成的图像保存在: outputs/test_output.png"
echo ""
echo "更多使用示例:"
echo ""
echo "# txt2img 生成"
echo "python3 main.py --prompt \"your prompt here\" --output output.png --steps 50"
echo ""
echo "# img2img 转换"
echo "python3 main.py --prompt \"oil painting style\" --input photo.jpg --strength 0.7 --output output.png"
echo ""
echo "# 使用负面提示词"
echo "python3 main.py --prompt \"beautiful landscape\" --uncond-prompt \"blurry, low quality\" --output output.png"
echo ""
