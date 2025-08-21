#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Module调用错误修复工具
专门用于解决 Module._wrapped_call_impl 错误
"""

import torch
import torch.nn as nn
import traceback
from typing import Any, Optional


def diagnose_module_error(model: nn.Module, x: Any) -> dict:
    """诊断Module调用错误"""
    diagnosis = {
        'error_type': 'Module._wrapped_call_impl',
        'issues': [],
        'suggestions': []
    }
    
    print("=== 错误诊断 ===")
    
    # 检查模型类型
    if not isinstance(model, nn.Module):
        diagnosis['issues'].append("模型不是nn.Module的实例")
        diagnosis['suggestions'].append("确保模型继承自nn.Module")
        return diagnosis
    
    # 检查输入类型
    if not isinstance(x, torch.Tensor):
        diagnosis['issues'].append(f"输入不是torch.Tensor，而是 {type(x)}")
        diagnosis['suggestions'].append("将输入转换为torch.Tensor")
    
    # 检查模型是否有forward方法
    if not hasattr(model, 'forward'):
        diagnosis['issues'].append("模型没有forward方法")
        diagnosis['suggestions'].append("在模型中实现forward方法")
    
    # 检查输入形状
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            diagnosis['issues'].append("输入是标量，需要添加维度")
            diagnosis['suggestions'].append("使用x.unsqueeze(0)添加批次维度")
        elif x.dim() == 1:
            diagnosis['issues'].append("输入是一维的，可能需要添加批次维度")
            diagnosis['suggestions'].append("使用x.unsqueeze(0)添加批次维度")
    
    # 检查设备匹配
    if isinstance(x, torch.Tensor) and list(model.parameters()):
        model_device = next(model.parameters()).device
        if x.device != model_device:
            diagnosis['issues'].append(f"设备不匹配：输入在 {x.device}，模型在 {model_device}")
            diagnosis['suggestions'].append(f"使用x.to({model_device})或model.to({x.device})")
    
    return diagnosis


def fix_module_call(model: nn.Module, x: Any) -> Optional[torch.Tensor]:
    """修复Module调用错误"""
    print("=== 尝试修复Module调用错误 ===")
    
    # 诊断问题
    diagnosis = diagnose_module_error(model, x)
    
    if diagnosis['issues']:
        print("发现的问题:")
        for i, issue in enumerate(diagnosis['issues'], 1):
            print(f"{i}. {issue}")
        
        print("\n建议的解决方案:")
        for i, suggestion in enumerate(diagnosis['suggestions'], 1):
            print(f"{i}. {suggestion}")
    
    try:
        # 修复输入数据
        if not isinstance(x, torch.Tensor):
            print("将输入转换为torch.Tensor...")
            x = torch.tensor(x, dtype=torch.float32)
        
        # 修复维度问题
        if x.dim() == 0:
            print("添加批次维度...")
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            print("添加批次维度...")
            x = x.unsqueeze(0)
        
        # 修复设备问题
        if list(model.parameters()):
            model_device = next(model.parameters()).device
            if x.device != model_device:
                print(f"将输入移动到模型设备: {model_device}")
                x = x.to(model_device)
        
        # 尝试调用模型
        print("尝试调用模型...")
        with torch.no_grad():
            output = model(x)
        
        print(f"修复成功！输出形状: {output.shape}")
        return output
        
    except Exception as e:
        print(f"修复失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return None


def create_safe_model_call(model: nn.Module):
    """创建安全的模型调用函数"""
    def safe_call(x: Any) -> Optional[torch.Tensor]:
        """安全的模型调用"""
        try:
            # 确保输入是tensor
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            # 确保有批次维度
            if x.dim() == 0:
                x = x.unsqueeze(0)
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            
            # 确保设备匹配
            if list(model.parameters()):
                model_device = next(model.parameters()).device
                if x.device != model_device:
                    x = x.to(model_device)
            
            # 调用模型
            with torch.no_grad():
                return model(x)
                
        except Exception as e:
            print(f"模型调用错误: {e}")
            return None
    
    return safe_call


def test_fix():
    """测试修复功能"""
    print("=== 测试Module错误修复 ===")
    
    # 创建测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.linear = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.linear(x)
    
    model = TestModel()
    
    # 测试各种错误情况
    test_cases = [
        ("一维输入", torch.randn(10)),
        ("标量输入", torch.tensor(1.0)),
        ("列表输入", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ("正确输入", torch.randn(1, 10))
    ]
    
    for name, x in test_cases:
        print(f"\n--- 测试: {name} ---")
        output = fix_module_call(model, x)
        if output is not None:
            print(f"✓ {name} 修复成功")
        else:
            print(f"✗ {name} 修复失败")


def main():
    """主函数"""
    print("PyTorch Module错误修复工具")
    print("=" * 50)
    
    # 运行测试
    test_fix()
    
    print("\n" + "=" * 50)
    print("使用说明:")
    print("1. 如果遇到Module._wrapped_call_impl错误，请使用fix_module_call函数")
    print("2. 对于重复调用，建议使用create_safe_model_call创建安全调用函数")
    print("3. 使用diagnose_module_error函数诊断具体问题")


if __name__ == "__main__":
    main()




