import torch
import matplotlib.pyplot as plt

def visualize_4d_tensor(tensor):
    # 假设输入tensor的形状是[1, 2, 3, 3]
    batch_size, channels, height, width = tensor.shape
    
    # 创建一个图表来显示所有通道
    fig, axes = plt.subplots(1, channels, figsize=(10, 4))
    
    # 对每个通道进行可视化
    for i in range(channels):
        # 获取第一个批次的第i个通道
        channel_data = tensor[0, i, :, :]
        
        # 显示热力图
        im = axes[i].imshow(channel_data, cmap='viridis')
        axes[i].set_title(f'channel {i+1}')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle('4D tensor views [1, 2, 3, 3]')
    plt.show()

# 创建示例数据
sample_tensor = torch.randn(1, 2, 3, 3)
visualize_4d_tensor(sample_tensor)

# 第一维 (1): 表示批次大小（Batch Size），即这个张量包含1个样本
# 第二维 (2): 表示通道数（Channels），比如可以理解为2个特征图
# 第三维 (3): 表示高度（Height），每个特征图有3行
# 第四维 (3): 表示宽度（Width），每个特征图有3列