import torch

# 假设我们有100个样本的数据集
total_samples = 100
feature_size = 10

# 1. 不使用batch（一次处理所有数据）
all_data = torch.randn(total_samples, feature_size)
print("全量数据形状:", all_data.shape)  # torch.Size([100, 10])

# 2. 使用batch_size=32进行分批处理
batch_size = 32
num_batches = total_samples // batch_size  # 得到批次数量
print("批次数量:", num_batches)

# 模拟批次处理
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    
    # 获取当前批次的数据
    batch_data = all_data[start_idx:end_idx]
    print(f"第{i+1}批数据形状:", batch_data.shape)  # torch.Size([32, 10])
    # print(f"第{i+1}批数据:", batch_data)

# 3. 实际训练中的batch示例
# 图像分类任务的数据格式
image_batch = torch.randn(batch_size, 3, 224, 224)  # [批次, 通道, 高度, 宽度]
print("图像批次数据形状:", image_batch.shape)  # torch.Size([32, 3, 224, 224])

# 序列数据的批次格式
sequence_batch = torch.randn(batch_size, 50, 256)  # [批次, 序列长度, 特征维度]
print("序列批次数据形状:", sequence_batch.shape)  # torch.Size([32, 50, 256])