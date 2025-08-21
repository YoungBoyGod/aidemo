import torch

# 创建测试数据
data1 = torch.tensor([[1,2],[3,4]])
data2 = torch.tensor([[5,6],[7,8]])

print("原始数据:")
print("data1:", data1)
print("data2:", data2)
print()

# 1. 元素级别的乘法（Hadamard积）
print("=== 元素级别乘法 ===")
data3 = data1 * data2
print("data1 * data2 (元素级别):")
print(data3)
print()

# 2. 矩阵乘法 - 方法1：使用 matmul()
print("=== 矩阵乘法 - matmul() ===")
data4 = data1.matmul(data2)
print("data1.matmul(data2):")
print(data4)
print()

# 3. 矩阵乘法 - 方法2：使用 torch.matmul()
print("=== 矩阵乘法 - torch.matmul() ===")
data5 = torch.matmul(data1, data2)
print("torch.matmul(data1, data2):")
print(data5)
print()

# 4. 矩阵乘法 - 方法3：使用 @ 运算符
print("=== 矩阵乘法 - @ 运算符 ===")
data6 = data1 @ data2
print("data1 @ data2:")
print(data6)
print()

# 5. 如果要使用 dot() 函数，需要先展平
print("=== 向量点积 - dot() ===")
data1_flat = data1.flatten()
data2_flat = data2.flatten()
data7 = data1_flat.dot(data2_flat)
print("data1.flatten().dot(data2.flatten()):")
print(data7)
print()

# 6. 验证所有矩阵乘法方法结果相同
print("=== 验证结果一致性 ===")
print("matmul() == torch.matmul():", torch.equal(data4, data5))
print("matmul() == @ 运算符:", torch.equal(data4, data6))
print("所有矩阵乘法结果相同:", torch.equal(data4, data5) and torch.equal(data4, data6))

# 7. 错误示例（注释掉）
print("\n=== 错误示例（已注释） ===")
# data5_error = data1.dot(data2)  # 这会报错：RuntimeError: 1D tensors expected, but got 2D and 2D tensors
print("# data1.dot(data2)  # ❌ 错误：dot()只能用于一维张量") 