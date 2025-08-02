import torch

# 原地操作
a=torch.tensor([1,2,3])
print("a",a)
a1=a.add(1)
print("a1",a1)
print("a",a)

# 广播机制  张量参数可以自动扩展为相同大小
# 每个张量维度相同，或者其中一个维度为1
# 满足右对齐
# 对于一个tensor 从右向左看，维度相同，或者其中一个维度为1
a=torch.tensor([1,2,3])
b=torch.tensor([4,5,6])
c=a+b
print("c",c)

d=torch.rand(2,1,1)+torch.rand(3)
print("d",d)
print("d.shape",d.shape)

e=torch.rand(2,1,1)+torch.rand(2,3,1)
print("e",e)
print("e.shape",e.shape)