import torch 
a=torch.tensor([1,2,3])
b=torch.tensor([4,5,6])
print(a>b)
print(torch.gt(a,b))
print(a<b)
print(torch.lt(a,b))
print(a==b)
print(torch.eq(a,b))
print(a!=b)
print(torch.ne(a,b))
print(a>=b)
print(torch.ge(a,b))
print(a<=b)
print(torch.le(a,b))

print(torch.all(a>b))
print(torch.any(a>b))


print(torch.where(a>b,a,b)) # 根据条件选择值
print(torch.sort(a)) # 排序
print(torch.argsort(a)) # 排序后的索引
print(torch.topk(a,2)) # 取前k个值
print(torch.kthvalue(a,2))# 取第k个值
print(torch.max(a)) # 取最大值
print(torch.min(a)) # 取最小值
print(torch.argmax(a)) # 取最大值的索引
print(torch.argmin(a)) # 取最小值的索引



