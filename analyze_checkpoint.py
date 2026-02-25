"""
分析 checkpoint 文件的 key 结构，帮助创建正确的映射。
"""
import torch
from pathlib import Path
from collections import defaultdict

def analyze_checkpoint(ckpt_path: str):
    """分析 checkpoint 的 key 结构"""
    print(f"分析文件: {ckpt_path}\n")
    
    # 加载 checkpoint
    if ckpt_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path, device="cpu")
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
    
    print(f"总共 {len(state_dict)} 个参数\n")
    
    # 按前缀分组
    groups = defaultdict(list)
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = f"{parts[0]}.{parts[1]}"
        else:
            prefix = parts[0]
        groups[prefix].append(key)
    
    # 打印每个组的示例
    print("=" * 80)
    print("按前缀分组的 Keys:")
    print("=" * 80)
    
    for prefix in sorted(groups.keys()):
        keys = groups[prefix]
        print(f"\n【{prefix}】 ({len(keys)} 个参数)")
        print("-" * 80)
        
        # 打印前 10 个 keys
        for key in keys[:10]:
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {key}")
            print(f"    shape: {shape}")
        
        if len(keys) > 10:
            print(f"  ... 还有 {len(keys) - 10} 个参数")
    
    # 特别关注 U-Net 的结构
    print("\n" + "=" * 80)
    print("U-Net 详细结构分析:")
    print("=" * 80)
    
    unet_keys = [k for k in state_dict.keys() if k.startswith("model.diffusion_model.")]
    
    # 分析 input_blocks
    input_blocks = defaultdict(list)
    for key in unet_keys:
        if "input_blocks" in key:
            parts = key.split('.')
            block_idx = parts[2]  # input_blocks.X
            input_blocks[block_idx].append(key)
    
    print("\nInput Blocks:")
    for idx in sorted(input_blocks.keys(), key=int):
        print(f"\n  Block {idx}: ({len(input_blocks[idx])} 参数)")
        for key in input_blocks[idx][:5]:
            print(f"    {key}")
        if len(input_blocks[idx]) > 5:
            print(f"    ... 还有 {len(input_blocks[idx]) - 5} 个")
    
    # 分析 middle_block
    middle_keys = [k for k in unet_keys if "middle_block" in k]
    print(f"\n\nMiddle Block: ({len(middle_keys)} 参数)")
    for key in middle_keys[:10]:
        print(f"  {key}")
    
    # 分析 output_blocks
    output_blocks = defaultdict(list)
    for key in unet_keys:
        if "output_blocks" in key:
            parts = key.split('.')
            block_idx = parts[2]
            output_blocks[block_idx].append(key)
    
    print("\n\nOutput Blocks:")
    for idx in sorted(output_blocks.keys(), key=int):
        print(f"\n  Block {idx}: ({len(output_blocks[idx])} 参数)")
        for key in output_blocks[idx][:5]:
            print(f"    {key}")
        if len(output_blocks[idx]) > 5:
            print(f"    ... 还有 {len(output_blocks[idx]) - 5} 个")

if __name__ == "__main__":
    # 查找 checkpoint 文件
    cache_dir = Path("./data")
    ckpt_files = list(cache_dir.rglob("*.safetensors")) + list(cache_dir.rglob("*.ckpt"))
    
    if not ckpt_files:
        print("未找到 checkpoint 文件")
        print("请先下载权重文件到 ./data 目录")
    else:
        analyze_checkpoint(str(ckpt_files[0]))
