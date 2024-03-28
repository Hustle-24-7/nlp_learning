import torch
from torch import nn, optim
from torch.nn import functional as F
 
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)
    
    def forward(self, inputs):
        hidden = self.linear1(inputs)
        activation = self.activate(hidden)
        outputs = self.linear2(activation)
        
        log_probs = F.log_softmax(outputs, dim=1)
        # 取对数是因为避免计算softmax时产生数值溢出
        return log_probs

x_train = torch.tensor([[0.0, 0.0],[0.0, 1.0],[1.0,0.0],[1.0,1.0]])
y_trian = torch.tensor([0,1,1,0])

model = MLP(input_dim=2,hidden_dim=5, num_class=2)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

for epoch in range(500):
    y_pred = model(x_train)
    loss = criterion(y_pred,y_trian)
    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step()
    
print("Parameters:")
for name, param in model.named_parameters():
    print(name, param.data)
    
y_pred = model(x_train)
print("Predicted results:", y_pred.argmax(axis=1))
