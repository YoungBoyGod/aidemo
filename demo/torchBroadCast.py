import torch

print("1. 基础广播示例")
# 1. 标量与张量的广播
a = torch.randn(3, 4)
b = 2
c = a + b  # b被广播到a的形状
print("标量广播:")
print("a shape:", a.shape)
print("c shape:", c.shape)
# 标量广播:
    # a shape: torch.Size([3, 4])
    # c shape: torch.Size([3, 4])

# 2. 不同维度张量的广播
print("\n2. 维度广播示例")
x = torch.randn(3, 1, 4)
y = torch.randn(1, 2, 4)
z = x + y  # 广播后形状为(3, 2, 4)
print("x shape:", x.shape)
print("y shape:", y.shape)
print("z shape:", z.shape)
# 2. 维度广播示例
    # x shape: torch.Size([3, 1, 4])
    # y shape: torch.Size([1, 2, 4])
    # z shape: torch.Size([3, 2, 4])

# 3. 常见的广播场景
print("\n3. 常见广播场景")
# 3.1 添加偏置项
features = torch.randn(32, 10)  # 批次数据
bias = torch.randn(10)          # 偏置项
output = features + bias        # bias广播到每个样本
print("添加偏置:")
print("features shape:", features.shape)
print("bias shape:", bias.shape)
print("output shape:", output.shape)

# 3. 常见广播场景
    #   添加偏置:
        # features shape: torch.Size([32, 10])
        # bias shape: torch.Size([10])
        # output shape: torch.Size([32, 10])

# 3.2 归一化操作
data = torch.randn(32, 3, 64, 64)  # 批次图像数据
mean = torch.mean(data, dim=[0, 2, 3], keepdim=True)  # 计算通道均值
std = torch.std(data, dim=[0, 2, 3], keepdim=True)    # 计算通道标准差
normalized = (data - mean) / std
print("\n归一化操作:")
print("data shape:", data.shape)
print("mean shape:", mean.shape)
print("normalized shape:", normalized.shape)

# 归一化操作:
    # data shape: torch.Size([32, 3, 64, 64])
    # mean shape: torch.Size([1, 3, 1, 1])
    # normalized shape: torch.Size([32, 3, 64, 64])

# 4. 广播规则演示
print("\n4. 广播规则详解")
def show_broadcastable(shape1, shape2):
    try:
        # 创建示例张量
        t1 = torch.randn(shape1)
        t2 = torch.randn(shape2)
        # 尝试相加
        result = t1 + t2
        print(f"Shape {shape1} 和 {shape2} 可以广播到: {result.shape}")
    except RuntimeError as e:
        print(f"Shape {shape1} 和 {shape2} 不能广播: {str(e)}")

# 测试不同形状组合
test_shapes = [
    ((3, 1, 4), (1, 2, 4)),    # 可以广播
    ((3, 2), (2,)),            # 可以广播
    ((3, 2), (3, 3)),          # 不能广播
    ((5, 1, 4, 1), (3, 1, 1)), # 可以广播
]

for shape1, shape2 in test_shapes:
    show_broadcastable(shape1, shape2)

# 4. 广播规则详解
    # Shape (3, 1, 4) 和 (1, 2, 4) 可以广播到: torch.Size([3, 2, 4])
    # Shape (3, 2) 和 (2,) 可以广播到: torch.Size([3, 2])
    # Shape (3, 2) 和 (3, 3) 不能广播: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
    # Shape (5, 1, 4, 1) 和 (3, 1, 1) 可以广播到: torch.Size([5, 3, 4, 1])

# 5. 广播性能注意事项
print("\n5. 性能示例")
import time

# 比较显式扩展与广播
def compare_performance():
    # 准备数据
    a = torch.randn(1000, 1, 100)
    b = torch.randn(1, 50, 100)
    
    # 使用广播
    start = time.time()
    c = a + b
    broadcast_time = time.time() - start
    
    # 显式扩展
    start = time.time()
    a_expanded = a.expand(-1, 50, -1)
    b_expanded = b.expand(1000, -1, -1)
    d = a_expanded + b_expanded
    expand_time = time.time() - start
    
    print(f"广播时间: {broadcast_time:.6f}秒")
    print(f"显式扩展时间: {expand_time:.6f}秒")

compare_performance()