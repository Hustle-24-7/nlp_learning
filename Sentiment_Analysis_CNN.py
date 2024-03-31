import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence #使得一个批次中全部序列长度相同，用0补
from nltk.corpus import sentence_polarity
from tqdm.auto import tqdm
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

class CNNDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

#CNN网络模型
class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)#padding=1表示在卷积前，将序列前后各补充1个输入
        self.activate = F.relu
        self.linear = nn.Linear(num_filter, num_class)
    def forward(self, inputs):
        embedding = self.embedding(inputs)
        convolution = self.activate(self.conv1d(embedding.permute(0,2,1)))
        pooling = torch.max_pool1d(convolution,kernel_size=convolution.shape[2])
        outputs = self.linear(pooling.squeeze(dim=2))
        log_probs = F.log_softmax(outputs, dim = 1)
        return log_probs

def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # offsets = [0] + [i.shape[0] for i in inputs]
    # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    inputs=pad_sequence(inputs, batch_first=True)
    return inputs, targets


#超参数设置
embedding_dim=128
hidden_dim=256
num_class=2
batch_size = 32
num_epoch = 5
filter_size = 3
num_filter = 100

#加载数据
def load_sentence_polarity():
    from nltk.corpus import sentence_polarity
    
    vocab = Vocab.build(sentence_polarity.sents())
    train_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[:4000]]
    test_data = [(vocab.convert_tokens_to_ids(sentence), 0) for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
        + [(vocab.convert_tokens_to_ids(sentence), 1) for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    return train_data,test_data,vocab

train_data, test_data, vocab = load_sentence_polarity()
train_dataset = CNNDataset(train_data)
test_dataset = CNNDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

#加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(len(vocab), embedding_dim, filter_size, num_filter, num_class)
model.to(device)

#训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, targets = [x.to(device) for x in batch]
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Loss: {total_loss:.2f}")
    
#测试过程
acc=0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs)
        acc += (output.argmax(dim=1)==targets).sum().item()
        
#输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")