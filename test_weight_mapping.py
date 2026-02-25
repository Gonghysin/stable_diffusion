#!/usr/bin/env python3
"""
测试权重映射功能（不需要实际的 checkpoint 文件）
"""

def test_clip_mapping():
    """测试 CLIP 键名转换"""
    from model_loader import convert_clip_text_model_state_dict
    
    # 模拟官方格式的 keys
    test_keys = {
        "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight": "dummy",
        "cond_stage_model.transformer.text_model.embeddings.position_embedding": "dummy",
        "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": "dummy",
        "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight": "dummy",
        "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.v_proj.weight": "dummy",
        "cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight": "dummy",
        "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm1.weight": "dummy",
        "cond_stage_model.transformer.text_model.encoder.layers.0.layer_norm2.weight": "dummy",
        "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc1.weight": "dummy",
        "cond_stage_model.transformer.text_model.encoder.layers.0.mlp.fc2.weight": "dummy",
        "cond_stage_model.transformer.text_model.final_layer_norm.weight": "dummy",
    }
    
    converted = convert_clip_text_model_state_dict(test_keys)
    
    print("=" * 80)
    print("CLIP 键名映射测试")
    print("=" * 80)
    print(f"\n转换了 {len(converted)} 个键:\n")
    
    for old_key in test_keys.keys():
        new_key = [k for k in converted.keys() if old_key.endswith(k.split('.')[-1])]
        if new_key:
            print(f"✓ {old_key}")
            print(f"  → {new_key[0]}\n")
    
    return len(converted) == len(test_keys)


def test_unet_mapping():
    """测试 U-Net 键名转换"""
    from model_loader import convert_unet_state_dict
    
    # 模拟官方格式的 keys
    test_keys = {
        "model.diffusion_model.time_embed.0.weight": "dummy",
        "model.diffusion_model.time_embed.2.weight": "dummy",
        "model.diffusion_model.input_blocks.0.0.weight": "dummy",
        "model.diffusion_model.input_blocks.1.0.in_layers.0.weight": "dummy",
        "model.diffusion_model.input_blocks.1.0.in_layers.2.weight": "dummy",
        "model.diffusion_model.input_blocks.1.0.emb_layers.1.weight": "dummy",
        "model.diffusion_model.input_blocks.1.0.out_layers.0.weight": "dummy",
        "model.diffusion_model.input_blocks.1.0.out_layers.3.weight": "dummy",
        "model.diffusion_model.input_blocks.1.1.norm.weight": "dummy",
        "model.diffusion_model.input_blocks.1.1.proj_in.weight": "dummy",
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1.weight": "dummy",
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight": "dummy",
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight": "dummy",
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight": "dummy",
        "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight": "dummy",
        "model.diffusion_model.middle_block.0.in_layers.0.weight": "dummy",
        "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": "dummy",
        "model.diffusion_model.out.0.weight": "dummy",
        "model.diffusion_model.out.2.weight": "dummy",
    }
    
    converted = convert_unet_state_dict(test_keys)
    
    print("\n" + "=" * 80)
    print("U-Net 键名映射测试")
    print("=" * 80)
    print(f"\n转换了 {len(converted)} 个键:\n")
    
    # 显示一些关键的映射
    key_mappings = [
        ("model.diffusion_model.time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("model.diffusion_model.input_blocks.0.0.weight", "unet.encoders.0.0.weight"),
        ("model.diffusion_model.input_blocks.1.0.in_layers.0.weight", "unet.encoders.1.0.groupnorm_feature.weight"),
        ("model.diffusion_model.input_blocks.1.0.emb_layers.1.weight", "unet.encoders.1.0.linear_time.weight"),
        ("model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight", "unet.encoders.1.1.transformer_blocks.0.attention_1.q_proj.weight"),
        ("model.diffusion_model.out.0.weight", "final.groupnorm.weight"),
    ]
    
    for old_key, expected_new_key in key_mappings:
        if old_key in test_keys:
            actual_new_key = [k for k in converted.keys() if k == expected_new_key]
            if actual_new_key:
                print(f"✓ {old_key}")
                print(f"  → {actual_new_key[0]}\n")
            else:
                print(f"✗ {old_key}")
                print(f"  期望: {expected_new_key}")
                print(f"  实际: 未找到匹配\n")
    
    return len(converted) == len(test_keys)


def test_model_structure():
    """测试模型结构是否正确"""
    from sd.clip import CLIP
    from sd.diffusion import Diffusion
    
    print("\n" + "=" * 80)
    print("模型结构验证")
    print("=" * 80)
    
    # 检查 CLIP 结构
    clip = CLIP()
    clip_params = list(clip.named_parameters())
    print(f"\nCLIP 模型参数数量: {len(clip_params)}")
    print("前 5 个参数:")
    for name, _ in clip_params[:5]:
        print(f"  - {name}")
    
    # 检查 Diffusion 结构
    diffusion = Diffusion()
    diff_params = list(diffusion.named_parameters())
    print(f"\nDiffusion 模型参数数量: {len(diff_params)}")
    print("前 5 个参数:")
    for name, _ in diff_params[:5]:
        print(f"  - {name}")
    
    # 检查关键层是否存在
    print("\n关键层检查:")
    checks = [
        ("time_embedding.linear_1.weight", diffusion),
        ("unet.encoders.0.0.weight", diffusion),
        ("unet.bottleneck.0.groupnorm_feature.weight", diffusion),
        ("final.groupnorm.weight", diffusion),
    ]
    
    for param_name, model in checks:
        exists = any(param_name in name for name, _ in model.named_parameters())
        status = "✓" if exists else "✗"
        print(f"  {status} {param_name}")
    
    return True


def main():
    print("\n" + "=" * 80)
    print("权重映射功能测试")
    print("=" * 80)
    
    results = []
    
    # 测试 CLIP 映射
    try:
        results.append(("CLIP 映射", test_clip_mapping()))
    except Exception as e:
        print(f"\n✗ CLIP 映射测试失败: {e}")
        results.append(("CLIP 映射", False))
    
    # 测试 U-Net 映射
    try:
        results.append(("U-Net 映射", test_unet_mapping()))
    except Exception as e:
        print(f"\n✗ U-Net 映射测试失败: {e}")
        results.append(("U-Net 映射", False))
    
    # 测试模型结构
    try:
        results.append(("模型结构", test_model_structure()))
    except Exception as e:
        print(f"\n✗ 模型结构测试失败: {e}")
        results.append(("模型结构", False))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败")
    
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
