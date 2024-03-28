import torch
from torch.nn import Conv1d
conv1 = Conv1d(5, 2, 4)
conv2 = Conv1d(5, 2, 3)

inputs = torch.rand(2, 5, 6)
# print(inputs)
outputs1 = conv1(inputs)
outputs2 = conv2(inputs)
print(outputs1)
print(outputs2)

from torch.nn import MaxPool1d
pool1 = MaxPool1d(3)
pool2 = MaxPool1d(4)
outputs_pool1 = pool1(outputs1)
outputs_pool2 = pool1(outputs2)
print(outputs_pool1)
print(outputs_pool2)

#也可以使用池化函数，无须事先指定池化层核的大小，当处理不定序列时更合适
import torch.nn.functional as F
outputs_pool1 = F.max_pool1d(outputs1, kernel_size=outputs1.shape[2])
print(outputs_pool1)
outputs_pool2 = F.max_pool1d(outputs2, kernel_size=outputs2.shape[2])
print(outputs_pool2)

outputs_pool_squeeze1 = outputs_pool1.squeeze(dim=2)
print(outputs_pool_squeeze1)
outputs_pool_squeeze2 = outputs_pool2.squeeze(dim=2)
print(outputs_pool_squeeze2)
outputs_pool = torch.cat([outputs_pool_squeeze1, outputs_pool_squeeze2], dim=1)
print(outputs_pool)

from torch.nn import Linear
linear = Linear(4,2)
outputs_linear = linear(outputs_pool)
print(outputs_linear)
