from torch.nn import RNN
import torch
rnn = RNN(input_size = 4, hidden_size = 5, batch_first=True)
inputs = torch.rand(2, 3 ,4)
outputs, hn = rnn(inputs)
print(outputs)
print(hn)
print(outputs.shape, hn.shape)

from torch.nn import LSTM
lstm = LSTM(input_size=4, hidden_size=5, batch_first = True)
inputs= torch.rand(2,3,4)
outputs,(hn,cn)= lstm(inputs)
print(outputs)
print(hn)
print(cn)
print(outputs.shape,hn.shape,cn.shape)