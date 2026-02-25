# Stable Diffusion 实现

从零开始实现的 Stable Diffusion v1.5 项目，用于学习扩散模型原理。

## 项目结构

```
sd/
├── clip.py          # CLIP 文本编码器 (77 tokens → 768-dim embeddings)
├── encoder.py       # VAE 编码器 (512×512 image → 64×64 latent)
├── decoder.py       # VAE 解码器 (64×64 latent → 512×512 image)
├── diffusion.py     # U-Net 扩散模型 (预测噪声)
├── ddpm.py          # DDPM 调度器 (加噪/去噪采样)
└── pipeline.py      # 推理 Pipeline (编排所有模块)

attention.py         # Self-Attention & Cross-Attention
model_loader.py      # 权重加载工具 (简化版)
main.py              # 命令行推理入口
```

## 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy pillow tqdm transformers
```

### txt2img 生成

```bash
python main.py \
    --prompt "a photo of an astronaut riding a horse" \
    --output output.png \
    --steps 50 \
    --seed 42
```

### img2img 转换

```bash
python main.py \
    --prompt "oil painting style" \
    --input photo.jpg \
    --strength 0.7 \
    --output output.png
```

## 核心模块

### DDPMSampler (sd/ddpm.py)
噪声调度器，管理扩散过程：
- `add_noise()`: 前向加噪 x_t = √α̅_t·x_0 + √(1-α̅_t)·ε
- `step()`: 反向去噪，单步采样

### Pipeline (sd/pipeline.py)
完整推理流程：
1. CLIP 文本编码 (条件 + 无条件)
2. 初始化 latent (纯噪声或编码输入图像)
3. 去噪循环 (50 步)
   - U-Net 预测噪声
   - CFG 组合: ε = ε_uncond + scale·(ε_cond - ε_uncond)
   - DDPM 去噪一步
4. VAE 解码为图像

### U-Net (sd/diffusion.py)
降噪网络架构：
- Encoder: 4 个下采样级别 (320→640→1280 通道)
- Bottleneck: 最深层特征处理
- Decoder: 4 个上采样级别 + 跳跃连接
- 每层包含 ResBlock + AttentionBlock (Self + Cross)

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--prompt` | 文本提示词 (必填) | - |
| `--uncond-prompt` | 负面提示词 | "" |
| `--input` | 输入图像 (img2img) | None |
| `--strength` | img2img 强度 (0-1) | 0.8 |
| `--cfg-scale` | CFG 强度 | 7.5 |
| `--steps` | 推理步数 | 50 |
| `--seed` | 随机种子 | None |
| `--ckpt` | 权重路径 | None |
| `--device` | 设备 (cuda/mps/cpu) | 自动 |

## 关键技术

### Classifier-Free Guidance (CFG)
```python
ε_final = ε_uncond + cfg_scale * (ε_cond - ε_uncond)
```
- cfg_scale=1: 无引导
- cfg_scale=7-8: 典型值
- cfg_scale>15: 过度引导

### img2img 强度控制
```python
start_step = int(steps * (1 - strength))
# strength=0.8 → 从第 20% 步开始去噪
```

## 数据流

```
"a cat on the moon" → CLIP → (1,77,768)
                                  ↓
纯噪声 (1,4,64,64) → 去噪50步 → 清晰latent
                                  ↓
                            VAE解码 → 图像
```

## 注意事项

1. **权重加载**: `model_loader.py` 是简化版，需根据实际 checkpoint 实现 key 映射
2. **显存优化**: 使用 `idle_device` 参数节省显存
3. **性能**: CPU 慢 (~10分钟)，GPU 快 (~10秒)，MPS 中等 (~30秒)

## 学习资源

- [DDPM 论文](https://arxiv.org/abs/2006.11239)
- [Stable Diffusion 论文](https://arxiv.org/abs/2112.10752)
- [CFG 论文](https://arxiv.org/abs/2207.12598)
