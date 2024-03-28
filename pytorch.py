import torch
torch.empty(2,3)
torch.rand(2,3)
torch.randn(2,3)

torch.zeros(2,3,dtype=torch.long)
#张量数据类型为整数
torch.zeros(2,3,dtype=torch.double)
#张量数据类型为双精度浮点

torch.tensor([[1,3.8,2],[86,4,24.0]])

# a=1
# type(a)

torch.rand(2,3).cuda()
torch.rand(2,3).to("cuda")
torch.rand(2,3,device="cuda")

x = torch.tensor([1,2,3],dtype=torch.double)
y = torch.tensor([4,5,6],dtype=torch.double)
print(x+y)
print(x-y)
print(x*y)
print(x/y)
x.dot(y)
x.sin()
x.exp()

x=torch.tensor([[1,2,3],[4,5,6.0]])
# mean(): input dtype should be either floating point or complex dtypes. Got Long instead.
x.mean()
x.mean(dim=0)
x.mean(dim=1)

x.mean(dim=0,keepdim=True)
x.mean(dim=1,keepdim=True)

x=torch.tensor([[1,2,3],[4,5,6]])
y=torch.tensor([[7,8,9],[10,11,12]])
torch.cat((x,y),dim=0)
torch.cat((x,y),dim=1)

x=torch.tensor([2.], requires_grad=True)
y=torch.tensor([3.], requires_grad=True)
z=(x+y)*(y-2)
print(z)
z.backward()
print(x.grad, y.grad)

x=torch.tensor([1,2,3,4,5,6])
print(x, x.shape)
x.view(2,3)
x.view(3,2)
x.view(-1,3)

x=torch.tensor([[1,2,3],[4,5,6]])
x.transpose(0,1)

x=torch.tensor([[[1,2,3],[4,5,6]]])
x=x.permute(2,0,1)
print(x, x.shape)

x=torch.arange(1,4).view(3,1)
y=torch.arange(4,6).view(1,2)
print(x)
print(y)
x+y

x=torch.arange(12).view(3,4)
print(x)
x[1,3]
x[1]
x[1:3]
x[:,2]
x[:,2:4]
x[:,2:4]=100
print(x)

a=torch.tensor([1,2,3,4])
print(a.shape)
b=torch.unsqueeze(a,dim=0)
print(b,b.shape)
b=a.unsqueeze(dim=0)
print(b,b.shape)
c=b.squeeze()
print(c,c.shape)

a=torch.tensor([[[1,2,3]],[[2,3,4]]])
a.shape

