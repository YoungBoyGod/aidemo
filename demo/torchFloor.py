import torch 
a=torch.tensor([1.5,2.5,3.5])
print(a)
print(torch.floor(a)) # 向下取整
print(torch.ceil(a)) # 向上取整
print(torch.round(a)) # 四舍五入
print(torch.trunc(a)) # 取整数部分
print(torch.frac(a)) # 取小数部分
print(torch.sign(a)) # 取符号
print(torch.abs(a)) # 取绝对值
print(torch.neg(a)) # 取相反数



b=torch.tensor(-1.5)
print(b)
print(torch.floor(b)) # 向下取整
print(torch.ceil(b)) # 向上取整
print(torch.round(b)) # 四舍五入