#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Module调试工具
用于诊断和解决PyTorch模型调用错误
"""

import torch
import torch.nn as nn
import traceback
from typing import Any, Dict, Optional, Tuple


class ModelDebugger:
    """PyTorch模型调试器"""
    
    def __init__(self):
        self.debug_info = {}
    
    def inspect_model(self, model: nn.Module, x: torch.Tensor) -> Dict[str, Any]:
        """检查模型结构和输入数据"""
        info = {
            'model_structure': str(model),
            'input_shape': x.shape,
            'input_dtype': x.dtype,
            'input_device': x.device,
            'model_device': next(model.parameters()).device if list(model.parameters()) else 'cpu',
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'model_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        print("=== 模型检查 ===")
        print(f"模型结构:\n{info['model_structure']}")
        print(f"输入形状: {info['input_shape']}")
        print(f"输入类型: {info['input_dtype']}")
        print(f"输入设备: {info['input_device']}")
        print(f"模型设备: {info['model_device']}")
        print(f"模型参数数量: {info['model_parameters']}")
        print(f"可训练参数数量: {info['model_trainable_parameters']}")
        
        return info
    
    def validate_input(self, model: nn.Module, x: torch.Tensor) -> Tuple[bool, str]:
        """验证输入数据是否适合模型"""
        issues = []
        
        # 检查数据类型
        if not isinstance(x, torch.Tensor):
            issues.append(f"输入不是torch.Tensor，而是 {type(x)}")
        
        # 检查设备匹配
        model_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        if x.device != model_device:
            issues.append(f"设备不匹配：输入在 {x.device}，模型在 {model_device}")
        
        # 检查维度
        if x.dim() == 0:
            issues.append("输入是标量，可能需要添加维度")
        elif x.dim() == 1:
            issues.append("输入是一维的，可能需要添加批次维度")
        
        # 检查数值范围
        if torch.isnan(x).any():
            issues.append("输入包含NaN值")
        if torch.isinf(x).any():
            issues.append("输入包含无穷大值")
        
        is_valid = len(issues) == 0
        return is_valid, "\n".join(issues) if issues else "输入验证通过"
    
    def safe_model_call(self, model: nn.Module, x: torch.Tensor, 
                       enable_grad: bool = False) -> Optional[torch.Tensor]:
        """安全的模型调用"""
        try:
            # 验证输入
            is_valid, message = self.validate_input(model, x)
            print(f"输入验证: {message}")
            
            if not is_valid:
                print("输入验证失败，尝试修复...")
                x = self._fix_input(model, x)
            
            # 确保设备匹配
            model_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
            if x.device != model_device:
                print(f"将输入移动到模型设备: {model_device}")
                x = x.to(model_device)
            
            # 调用模型
            if enable_grad:
                output = model(x)
            else:
                with torch.no_grad():
                    output = model(x)
            
            print(f"模型调用成功，输出形状: {output.shape}")
            return output
            
        except Exception as e:
            print(f"模型调用失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            return None
    
    def _fix_input(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """尝试修复输入数据"""
        print("尝试修复输入数据...")
        
        # 确保是tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            print(f"转换为tensor: {x.shape}")
        
        # 添加批次维度
        if x.dim() == 0:
            x = x.unsqueeze(0)
            print(f"添加批次维度: {x.shape}")
        elif x.dim() == 1:
            x = x.unsqueeze(0)
            print(f"添加批次维度: {x.shape}")
        
        # 确保数据类型
        if x.dtype != torch.float32:
            x = x.float()
            print(f"转换为float32: {x.dtype}")
        
        return x
    
    def debug_forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """逐步调试前向传播"""
        print("=== 逐步调试前向传播 ===")
        print(f"初始输入: {x.shape}")
        
        # 获取所有模块
        modules = list(model.named_modules())
        
        for name, module in modules:
            if name == "":  # 跳过根模块
                continue
                
            if isinstance(module, nn.Linear):
                print(f"\n{name} (Linear): {module.in_features} -> {module.out_features}")
                x = module(x)
                print(f"输出形状: {x.shape}")
                
            elif isinstance(module, nn.ReLU):
                print(f"\n{name} (ReLU)")
                x = module(x)
                print(f"输出形状: {x.shape}")
                
            elif isinstance(module, nn.Sigmoid):
                print(f"\n{name} (Sigmoid)")
                x = module(x)
                print(f"输出形状: {x.shape}")
                
            elif isinstance(module, nn.Tanh):
                print(f"\n{name} (Tanh)")
                x = module(x)
                print(f"输出形状: {x.shape}")
        
        print(f"\n最终输出: {x.shape}")
        return x


def create_test_model() -> nn.Module:
    """创建测试模型"""
    class TestModel(nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 5)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.sigmoid(x)
            return x
    
    return TestModel()


def main():
    """主函数 - 演示调试工具的使用"""
    print("PyTorch Module调试工具")
    print("=" * 50)
    
    # 创建调试器
    debugger = ModelDebugger()
    
    # 创建测试模型
    model = create_test_model()
    print("创建测试模型完成")
    
    # 测试正确的输入
    print("\n=== 测试1: 正确的输入 ===")
    x_correct = torch.randn(1, 10)
    debugger.inspect_model(model, x_correct)
    output = debugger.safe_model_call(model, x_correct)
    
    if output is not None:
        print(f"测试1成功，输出: {output}")
    
    # 测试错误的输入
    print("\n=== 测试2: 错误的输入 ===")
    x_wrong = torch.randn(10)  # 一维输入
    debugger.inspect_model(model, x_wrong)
    output = debugger.safe_model_call(model, x_wrong)
    
    if output is not None:
        print(f"测试2成功，输出: {output}")
    
    # 测试设备不匹配
    print("\n=== 测试3: 设备不匹配 ===")
    if torch.cuda.is_available():
        model = model.cuda()
        x_cpu = torch.randn(1, 10)
        debugger.inspect_model(model, x_cpu)
        output = debugger.safe_model_call(model, x_cpu)
        
        if output is not None:
            print(f"测试3成功，输出: {output}")
    
    # 逐步调试
    print("\n=== 测试4: 逐步调试 ===")
    x_debug = torch.randn(1, 10)
    debugger.debug_forward(model, x_debug)


if __name__ == "__main__":
    main()




