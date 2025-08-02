import torch
# 随机生成2行3列的矩阵
a=torch.rand(2,3)
print("a random",a)
print("a type",type(a))
# 生成2行3列的0矩阵
b=torch.zeros(2,3)
print("b zeros",b)
print("b type",type(b))
# 生成2行3列的1矩阵
c=torch.ones(2,3)
print("c ones",c)
# 生成2行3列的单位矩阵
d=torch.eye(2,3)
print("d eye",d)
# 生成2行3列的正态分布矩阵
e=torch.randn(2,3)
print("e randn",e)
# 生成2行3列的随机整数矩阵
f=torch.randint(1,10,[2,3])
print("f randint",f)
# 生成2行3列的均匀分布矩阵
g=torch.rand(2,3)
print("g rand",g)
# 生成2行3列的正态分布矩阵
h=torch.randn(2,3)
print("h randn",h)
# 生成2行3列的均匀分布矩阵
i=torch.rand(2,3)
print("i rand",i)
# 生成2行3列的正态分布矩阵
j=torch.randn(2,3)
print("j randn",j)
# 生成2行3列的正态分布矩阵
k=torch.normal(0,1,[2,3])
print("k normal",k)
# 生成10个随机整数
l=torch.randperm(10)
print("l randperm",l)
# 生成10个正态分布的随机数,均值为0,标准差为1
kk=torch.normal(mean=0.0,std=torch.rand(10))
print("kk normal",kk)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device",device)

aa= torch.tensor([2,2],device=device,dtype=torch.float32)
print("aa",aa)
bb=torch.tensor([[0,1,2],[0,1,2]])
v=torch.tensor([1,2,3])
cc=torch.sparse_coo_tensor(bb,v,size=(4,4),device=device,dtype=torch.float32)
print("cc sparse_coo_tensor",cc)

ccc=cc.to_dense()
print("ccc dense",ccc)



#  算术运算






