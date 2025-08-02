import torch
a=torch.tensor([1,2,3])
b=torch.tensor([4,5,6])
c=a+b
print("c",c) # c tensor([5, 7, 9])
c1=a.add(b)
print("c1",c1) # c1 tensor([5, 7, 9])
c2=a.add_(b)
print("c2",c2) # c2 tensor([5, 7, 9])
print("a",a) # a tensor([5, 7, 9])
# add_ 会修改a的值

d=a-b
print("d",d) # d tensor([1,2,3])
d1=a.sub(b)
print("d1",d1) # d1 tensor([1,2,3])
d2=a.sub_(b)
print("d2",d2) # d2 tensor([1,2,3])
print("a",a) # a tensor([1,2,3])

# 哈拉玛积
e=a*b
print("e",e) # e tensor([4, 10, 18])
e1=a.mul(b)
print("e1",e1) # e1 tensor([4, 10, 18])
e2=a.mul_(b)
print("e2",e2) # e2 tensor([4, 10, 18])
print("a",a) # a tensor([4, 10, 18])

a=a.float()
b=b.float()
# 除法  
f=a/b
print("f",f) # f tensor([1,2,3])
f0=torch.div(a,b)
print("f0",f0) # f0 tensor([0.25, 0.4, 0.5])
f1=a.div(b)
print("f1",f1) # f1 tensor([0.25, 0.4, 0.5])
f2=a.div_(b)
print("f2",f2) # f2 tensor([0.25, 0.4, 0.5])
# Traceback (most recent call last):
#   File "/Users/haitang/PyTorchDemo/demo/torchCompute.py", line 37, in <module>
#     f2=a.div_(b)
#        ^^^^^^^^^
# RuntimeError: result type Float can't be cast to the desired output type Long
print("a",a) # a tensor([0.25, 0.4, 0.5])

# 二维矩阵乘法
# torch.mm
# torch.matmul
aa= torch.ones(2,1)
bb= torch.ones(1,2)
cc=torch.mm(aa,bb)
print("cc",cc) # cc tensor([[1., 1.]])
cc1=torch.matmul(aa,bb)
print("cc1",cc1) # cc1 tensor([[1., 1.]])
cc2=aa@bb
print("cc2",cc2) # cc2 tensor([[1., 1.]])


aaa=torch.randn(1,2,3,4)
bbb=torch.randn(1,2,4,3)
ccc=torch.matmul(aaa,bbb)
print("ccc",ccc)
print("ccc.shape",ccc.shape)
ccc1=aaa@bbb
print("ccc1",ccc1)
print("ccc1.shape",ccc1.shape)
ccc2=aaa.matmul(bbb)
print("ccc2",ccc2)
print("ccc2.shape",ccc2.shape)
print("aaa",aaa)
print("aaa.shape",aaa.shape)
print("bbb",bbb)
print("bbb.shape",bbb.shape)


# 幂运算
a=torch.tensor([1,2,3])
print("a",a)
a1=a.pow(2)
print("a1",a1)
a2=a**2
print("a2",a2)

# 指数运算
a=torch.tensor([1,2,3])
print("a",a)
a1=a.exp()
print("a1",a1)

# 对数运算
a=torch.tensor([1,2,3])
print("a",a)
a1=a.log()
print("a1",a1)

# 开方运算
a=torch.tensor([1,2,3])
print("a",a)
a1=a.sqrt()
print("a1",a1)

