# import torch
# from torch import nn
# # linear = nn.Linear(in_features, out_features)
# linear = nn.Linear(32,2)
# inputs = torch.rand(3, 32)
# outputs = linear(inputs)
# print(outputs)

# from torch.nn import functional as F
# activation = F.sigmoid(outputs)
# print(activation)
# activation = F.softmax(outputs, dim=1)
# print(activation)
# activation = F.relu(outputs)
# print(activation)

#自定义神经网络模型 例如多层感知机
import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)
    
    def forward(self,inputs):
        hidden = self.linear1(inputs)
        activation = self.activate(hidden)
        outputs = self.linear2(activation)
        probs = F.softmax(outputs, dim=1)
        return probs

mlp = MLP(input_dim=4,hidden_dim=5,num_class=2)
inputs = torch.rand(3, 4)
probs = mlp(inputs)
print(probs)