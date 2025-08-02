
1. torch的属性
 - torch.device: 表示设备类型,cpu或gpu
 - torch.dtype: 表示数据类型,float32,float64,int32,int64,bool等
 - torch.layout: 表示数据布局,默认是torch.strided,表示连续的内存块,torch.sparse_coo表示稀疏矩阵
   - 稀疏矩阵: 矩阵中大部分元素为0的矩阵,只存储非0元素的值和位置 降低内存占用
    - 定义的示例：
   - 稠密矩阵: 矩阵中大部分元素不为0的矩阵,存储所有元素的值
   

 - torch.requires_grad: 表示是否需要计算梯度,默认是False
 - torch.grad_fn: 表示梯度函数,用于计算梯度
 - torch.data: 表示数据,用于存储数据
 - torch.grad: 表示梯度,用于存储梯度