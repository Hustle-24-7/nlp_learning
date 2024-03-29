from collections import defaultdict
class Vocab:
    def __init__(self,tokens = None):
        self.idx_to_token = list()
        self.token_to_idx = dict()
        
        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx["<unk>"]
            
    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token]+=1
                
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)
        # return uniq_tokens

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)
    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]
    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]
    
import torch
from torch import nn
from torch.nn import functional as F

embedding = nn.Embedding(8,3)
input = torch.tensor([[0,1,2,1],[4,6,6,7]], dtype=torch.long)
output = embedding(input)
print(output)
print(output.shape) 

class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class) -> None:
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)
    
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embedding = embeddings.mean(dim=1) #将每个序列中多个词向量聚合成一个向量
        
        hidden = self.linear1(embedding)
        activation = self.activate(hidden)
        outputs = self.linear2(activation)
        
        log_probs = F.log_softmax(outputs, dim=1)
        # 取对数是因为避免计算softmax时产生数值溢出
        return log_probs

mlp = MLP(vocab_size=8,embedding_dim=3, hidden_dim=5,num_class=2)
inputs = torch.tensor([[0,1,2,1],[4,6,6,7]], dtype=torch.long)
outputs = mlp(inputs)
print(outputs)

input1 = torch.tensor([0,1,2,1], dtype=torch.long)
input2 = torch.tensor([2,1,3,7,5], dtype=torch.long)
input3 = torch.tensor([6,4,2], dtype=torch.long)
input4 = torch.tensor([1,3,4,3,5,7], dtype=torch.long)
inputs = [input1,input2,input3,input4]
offsets = [0]+[i.shape[0] for i in inputs]
print(offsets)
offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
inputs = torch.cat(inputs)
print(inputs)
embeddingbag = nn.EmbeddingBag(num_embeddings=8,embedding_dim=3)
embeddings = embeddingbag(inputs,offsets)
print(embeddings)

class MLP_final(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_class):
        super(MLP_final, self).__init__()
        # 词向量层
        self.embedding = nn.EmbeddingBag(vocab_size,embedding_dim)
        # 线性变换： 词向量层 → 隐含层
        self.linear1 = nn.Linear(embedding_dim,hidden_dim)
        #  使用relu激活函数
        self.activate = torch.relu
        # 线性变换： 激活层 → 输出层
        self.linear2 = nn.Linear(hidden_dim,num_class)

    def forward(self,inputs,offsets):
        embedding = self.embedding(inputs,offsets)
        hidden = self.activate(self.linear1(embedding))
        outputs = self.linear2(hidden)
        log_probs = torch.log_softmax(outputs,dim = 1)
        return log_probs

def load_sentence_polarity():
    from nltk.corpus import sentence_polarity
    
    vocab = Vocab.build(sentence_polarity.sents())
    train_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[:4000]]
    test_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data,test_data,vocab

from torch.utils.data import DataLoader,Dataset
#data_loader = DataLoader(dataset, batch_size=64,collate_fn=collate_fn,shuffle=True)

class BowDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    offsets = [0] + [i.shape[0] for i in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    inputs=torch.cat(inputs)
    return inputs, offsets, targets

#MLP的训练与测试
from tqdm.auto import tqdm
from torch import optim

#超参数设置
embedding_dim=128
hidden_dim=256
num_class=2
batch_size = 32
num_epoch = 20

#加载数据
train_data, test_data, vocab = load_sentence_polarity()
train_dataset = BowDataset(train_data)
test_dataset = BowDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

#加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP_final(len(vocab), embedding_dim,hidden_dim,num_class)
model.to(device)

#训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, offsets, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, offsets)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Loss: {total_loss:.2f}")
    
#测试过程
acc=0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, offsets, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, offsets)
        acc += (output.argmax(dim=1)==targets).sum().item()
        
#输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")


        