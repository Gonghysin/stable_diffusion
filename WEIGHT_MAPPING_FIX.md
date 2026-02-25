# 权重映射修复说明

## 问题描述

原始的 `model_loader.py` 使用简单的前缀替换来加载权重，导致以下问题:

1. **权重未正确匹配**: 使用 `strict=False` 时，不匹配的 key 被静默跳过
2. **生成噪声图像**: 由于权重未加载，模型使用随机初始化的参数
3. **缺少详细诊断**: 无法知道实际匹配了多少权重

## 修复内容

### 1. CLIP 文本编码器映射

**官方格式** → **我们的格式**:
- `cond_stage_model.transformer.text_model.embeddings.token_embedding` → `embedding.token_embedding`
- `cond_stage_model.transformer.text_model.embeddings.position_embedding` → `embedding.position_embedding`
- `cond_stage_model.transformer.text_model.encoder.layers.N` → `layers.N`
- `self_attn` → `attention`
- `mlp.fc1` → `linear_1`
- `mlp.fc2` → `linear_2`
- `layer_norm1` → `layernorm_1`
- `layer_norm2` → `layernorm_2`
- `final_layer_norm` → `layernorm`

### 2. U-Net 扩散模型映射

**时间嵌入**:
- `model.diffusion_model.time_embed.0` → `time_embedding.linear_1`
- `model.diffusion_model.time_embed.2` → `time_embedding.linear_2`

**编码器块**:
- `model.diffusion_model.input_blocks.N` → `unet.encoders.N`

**瓶颈层**:
- `model.diffusion_model.middle_block.N` → `unet.bottleneck.N`

**解码器块**:
- `model.diffusion_model.output_blocks.N` → `unet.decoder.N`

**输出层**:
- `model.diffusion_model.out.0` → `final.groupnorm`
- `model.diffusion_model.out.2` → `final.conv`

### 3. ResBlock 内部映射

- `in_layers.0` → `groupnorm_feature`
- `in_layers.2` → `conv_feature`
- `emb_layers.1` → `linear_time`
- `out_layers.0` → `groupnorm_merged`
- `out_layers.3` → `conv_merged`
- `skip_connection` → `residual_layer`

### 4. Attention 块内部映射

- `norm` → `groupnorm`
- `proj_in` → `conv_input`
- `transformer_blocks.0.norm1` → `layernorm_1`
- `transformer_blocks.0.attn1.to_q` → `attention_1.q_proj`
- `transformer_blocks.0.attn1.to_k` → `attention_1.k_proj`
- `transformer_blocks.0.attn1.to_v` → `attention_1.v_proj`
- `transformer_blocks.0.attn1.to_out.0` → `attention_1.out_proj`
- `transformer_blocks.0.norm2` → `layernorm_2`
- `transformer_blocks.0.attn2` → `attention_2` (交叉注意力)
- `transformer_blocks.0.norm3` → `layernorm_3`
- `transformer_blocks.0.ff.net.0.proj` → `linear_geglu_1`
- `transformer_blocks.0.ff.net.2` → `linear_geglu_2`
- `proj_out` → `conv_output`

### 5. VAE 编码器/解码器映射

- `first_stage_model.encoder.*` → 保留层次结构，让 PyTorch 自动匹配
- `first_stage_model.decoder.*` → 保留层次结构，让 PyTorch 自动匹配

## 改进的诊断输出

新版本会显示:
- 实际匹配的参数数量 vs 总参数数量
- 缺失的参数列表 (前 10 个)
- 不匹配的参数列表 (前 10 个)

示例输出:
```
✓ CLIP: 匹配 199/201 个参数
  缺失 2 个参数 (将使用随机初始化)
    - embedding.position_embedding
    - layernorm.weight

✓ Diffusion: 匹配 856/860 个参数
  缺失 4 个参数 (将使用随机初始化)
  忽略 0 个不匹配的参数
```

## 使用方法

### 1. 下载并加载权重

```python
from model_loader import preload_models_from_standard_weights
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自动下载并加载
models = preload_models_from_standard_weights(
    ckpt_path="./data/v1-5-pruned-emaonly.safetensors",
    device=device
)
```

### 2. 手动指定权重文件

```bash
python main.py \
    --prompt "a beautiful landscape" \
    --ckpt /path/to/v1-5-pruned-emaonly.safetensors \
    --output output.png
```

### 3. 验证权重加载

运行 `analyze_checkpoint.py` 来查看 checkpoint 的详细结构:

```bash
python analyze_checkpoint.py
```

## 测试

1. **语法检查**: `python -m py_compile model_loader.py`
2. **加载测试**: 运行 `main.py` 并检查输出中的匹配统计
3. **生成测试**: 生成图像并验证不再是纯噪声

## 注意事项

1. **VAE 权重**: 由于 VAE 使用 `nn.Sequential` 结构，某些权重可能需要手动调整索引映射
2. **版本兼容性**: 此映射基于 Stable Diffusion v1.5 官方权重格式
3. **strict=False**: 仍然使用非严格模式，但现在会报告缺失的参数

## 备份

原始文件已备份到 `model_loader.py.backup`

## 下一步

如果权重仍未完全匹配:
1. 运行 `analyze_checkpoint.py` 查看实际的 key 结构
2. 对比模型定义和 checkpoint 结构
3. 调整映射函数中的转换规则
4. 特别关注 VAE 的 Sequential 索引映射
