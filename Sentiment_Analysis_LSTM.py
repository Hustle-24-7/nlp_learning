import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence #使得一个批次中全部序列长度相同，用0补
from torch.nn.utils.rnn import pack_padded_sequence #将变长序列打包
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

class LSTMDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

#LSTM网络模型
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_class)
    def forward(self, inputs, lengths):
        embeddings = self.embedding(inputs)
        #使用pack_padded_sequence函数将变长序列打包
        x_pack = pack_padded_sequence(embeddings, lengths.to('cpu'), batch_first=True, enforce_sorted=False) #RuntimeError: ‘lengths‘ argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor. 转到cpu运行
        hidden, (hn, cn) = self.lstm(x_pack)
        outputs = self.output(hn[-1])
        log_probs = F.log_softmax(outputs, dim = -1)
        return log_probs

def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs=pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


#超参数设置
embedding_dim=128
hidden_dim=256
num_class=2
batch_size = 32
num_epoch = 5
# filter_size = 3
# num_filter = 100

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
train_dataset = LSTMDataset(train_data)
test_dataset = LSTMDataset(test_data)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

#加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
model.to(device)

#训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
        inputs, lengths, targets = [x.to(device) for x in batch]
        log_probs = model(inputs, lengths)
        loss = nll_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Loss: {total_loss:.2f}")
    
#测试过程
acc=0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, lengths, targets = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs, lengths)
        acc += (output.argmax(dim=1)==targets).sum().item()
        
#输出在测试集上的准确率
print(f"Acc: {acc / len(test_data_loader):.2f}")

