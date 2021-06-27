# numpy 数据
import numpy as np
# 读取文件
import pandas as pd
# 机器学习框架
import torch
# 预处理
import sklearn




#把数据类型从list变成tensor
# a = [[1,2], [2,3]]
# print(a)
# print(type(a))
# tensor = torch.tensor(a) 
# print(tensor)
# print(type(tensor))


#把数据类型从NumPy array变成tensor
# a = np.arange(0,10,1).reshape(2,5)
# print(a)
# print(type(a))
# b = torch.from_numpy(a)
# print(b)
# print(type(b))


#数据转换路径
#DataFrame --> ndarray --> tensor


# From another tensor
# a1 = torch.ones_like(b)
# a2 = torch.rand_like(b.float()) #把tensor变成浮点型数据
# a3 = torch.zeros_like(b)
# print(a1)
# print(a2)
# print(a3)


#自定义已知尺寸的tensor
# size = (5, 5,)
# a = torch.rand(size)
# a = torch.ones(size)
# a = torch.zeros(size)
# print(a)
# print(a.shape)
# print(a.dtype)
# print(a.device)


#矩阵相同位置元素相乘用mul或者*；矩阵之间点乘用matmul或者@
# a1 = torch.from_numpy(np.arange(0, 4,1).reshape(2,2))
# a2 = torch.from_numpy(np.arange(4, 8,1).reshape(2,2))
# print(a1)
# print(a2)
# b = a1.matmul(a2)
# b = a1 @ a2
# print(b)


# 数据类型从ndarray转换成tensor（tensor无法存储数据，ndarray可以）
# tensor = torch.tensor([1,2])
# a = [1, 2]
# c = tensor.numpy()
# print(a)
# print(c)
# print(type(c))


# gradient descent 梯度下降算法(基于 torch 变量)————求导数！
# a = torch.tensor([1., 2.], requires_grad=True)
# print(a)
# x = torch.tensor([[1., 1.],[1., 1.]], requires_grad=True)
# z = 3 * (x + 2)**2
# out = z.mean() 
# print(out)
# out.backward() #backward指令可用于梯度回溯追踪工作(backward的对象只能是一个数，而不能是一个矩阵)
# print(x.grad)


