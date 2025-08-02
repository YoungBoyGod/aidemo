import torch
import matplotlib.pyplot as plt

# 1. 一维张量 (Vector)
vector = torch.randn(5)  # size: [5]
print("1D tensor vector:", vector.shape)
# 常见用途：单个特征序列，一维数组

# 2. 二维张量 (Matrix)
matrix = torch.randn(3, 4)  # size: [3, 4]
print("2D tensor matrix:", matrix.shape)
# 常见用途：
# - 样本数×特征数
# - 词嵌入矩阵（词数×嵌入维度）

# 3. 三维张量
tensor_3d = torch.randn(2, 3, 4)  # size: [2, 3, 4]
print("3D tensor tensor_3d:", tensor_3d.shape)
# 常见用途：
# - batch_size × seq_length × features（NLP）
# - batch_size × height × width（灰度图像）
# - batch_size × time_steps × features（时间序列）

# 4. 四维张量
tensor_4d = torch.randn(2, 3, 4, 4)  # size: [2, 3, 4, 4]
print("4D tensor tensor_4d:", tensor_4d.shape)
# 常见用途：
# - batch_size × channels × height × width（彩色图像/CNN）
# - batch_size × time_steps × height × width（视频帧）

# 5. 五维张量
tensor_5d = torch.randn(2, 3, 4, 4, 4)  # size: [2, 3, 4, 4, 4]
print("5D tensor tensor_5d:", tensor_5d.shape)
# 常见用途：
# - batch_size × channels × frames × height × width（视频数据）

# 6. 六维张量
tensor_6d = torch.randn(2, 3, 4, 4, 4, 4)  # size: [2, 3, 4, 4, 4, 4]
print("6D tensor tensor_6d:", tensor_6d.shape)
# 常见用途：
# - batch_size × channels × frames × height × width × depth（3D医学图像）

# 7. 七维张量
tensor_7d = torch.randn(2, 3, 4, 4, 4, 4, 4)  # size: [2, 3, 4, 4, 4, 4, 4]
print("7D tensor tensor_7d:", tensor_7d.shape)
# 常见用途：
# - batch_size × channels × frames × height × width × depth × time_steps（4D医学图像） # 应用：脑成像数据、多人运动捕捉数据

# 8维张量示例
tensor_8d = torch.randn(2,  # batch_size
                       10, # time_steps
                       100,# particles
                       3,  # dimensions
                       5,  # states
                       4,  # parameters
                       3,  # forces
                       2)  # constraints
print("8D tensor tensor_8d:", tensor_8d.shape)
# 应用：物理模拟、粒子系统

# 9维张量示例
tensor_9d = torch.randn(2,  # batch_size
                       5,  # agents
                       10, # time_steps
                       20, # states
                       15, # actions
                       3,  # rewards
                       4,  # policies
                       8,  # env_states
                       6)  # interactions
print("9D tensor tensor_9d:", tensor_9d.shape)
# 应用：多智能体强化学习

# 10维张量示例
tensor_10d = torch.randn(2,  # batch_size
                        3,  # modalities
                        5,  # subjects
                        10, # time_points
                        4,  # conditions
                        6,  # features
                        3,  # dimensions
                        2,  # states
                        4,  # parameters
                        5)  # measurements
print("10D tensor tensor_10d:", tensor_10d.shape)
# 应用：多模态数据分析、神经科学研究
def visualize_tensor_dimensions():
    # 创建示例数据
    fig = plt.figure(figsize=(15, 10))
    
    # 1D
    plt.subplot(231)
    plt.plot(vector.numpy())
    plt.title('1D tensor vector')
    
    # 2D
    plt.subplot(232)
    plt.imshow(matrix.numpy(), cmap='viridis')
    plt.title('2D tensor matrix')
    plt.colorbar()
    
    # 3D
    plt.subplot(233)
    plt.imshow(tensor_3d[0].numpy(), cmap='viridis')
    plt.title('3D tensor tensor_3d')
    plt.colorbar()
    
    # 4D
    plt.subplot(234)
    plt.imshow(tensor_4d[0,0].numpy(), cmap='viridis')
    plt.title('4D tensor tensor_4d')
    plt.colorbar()
    
    # 5D
    plt.subplot(235)
    plt.imshow(tensor_5d[0,0,0].numpy(), cmap='viridis')
    plt.title('5D tensor tensor_5d')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

    # 6D
    plt.subplot(236)
    plt.imshow(tensor_6d[0,0,0,0].numpy(), cmap='viridis')
    plt.title('6D张量 (第一个batch,第一个channel,第一个frame)')
    plt.colorbar()
    
    # 7D
    plt.subplot(237)
    plt.imshow(tensor_7d[0,0,0,0,0].numpy(), cmap='viridis')
    plt.title('7D张量 (第一个batch,第一个channel,第一个frame,第一个height,第一个width)')
    plt.colorbar()
    
    # 8D
    plt.subplot(238)
    plt.imshow(tensor_8d[0,0,0,0,0,0].numpy(), cmap='viridis')
    plt.title('8D张量 (第一个batch,第一个channel,第一个frame,第一个height,第一个width,第一个depth)')
    plt.colorbar()
    
    
    # 9D
    plt.subplot(239)
    plt.imshow(tensor_9d[0,0,0,0,0,0,0].numpy(), cmap='viridis')
    plt.title('9D张量 (第一个batch,第一个channel,第一个frame,第一个height,第一个width,第一个depth,第一个time_steps)')
    plt.colorbar()
    
    # 10D
    plt.subplot(2310)
    plt.imshow(tensor_10d[0,0,0,0,0,0,0,0].numpy(), cmap='viridis')
    plt.title('10D张量 (第一个batch,第一个channel,第一个frame,第一个height,第一个width,第一个depth,第一个time_steps,第一个state,第一个parameter,第一个measurement)')
    plt.colorbar()
    
    
    

# 显示可视化结果
visualize_tensor_dimensions()