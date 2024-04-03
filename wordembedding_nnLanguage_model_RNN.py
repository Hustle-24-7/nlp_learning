#由于在语言模型的训练中需要引入一些预留的标记，如句首、句尾，构建批次时用于补齐的标记
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch import nn,optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

class Vocab:
    def __init__(self,tokens = None):
        # 使用列表存储所有的标记，从而可根据索引值获取相应的标记
        self.idx_to_token = list()
        # 使用字典实现标记到索引值的映射
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

            self.unk = self.token_to_idx["<unk>"]

    # 创建词表，text包含若干句子，每个句子由若干标记构成
    @classmethod
    def build(cls,text,min_freq = 1 , reserved_tokens = None):
        # 存储标记及其出现次数的映射字典
        token_freqs = defaultdict(int)
        # 无重复地进行标记
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1

        # 用户自定义的预留标记
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token,freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]

        return cls(uniq_tokens)

    # 返回词表的大小，即词表中有多少个互不相同的标记
    def __len__(self):
        return len(self.idx_to_token)

    # 查找输入标记对应的索引值
    def __getitem__(self, token):
        return self.token_to_idx.get(token,self.unk)

    # 查找一系列输入标记对应的索引值
    def convert_tokens_to_ids(self,tokens):
        return [self[token] for token in tokens]

    # 查找一系列索引值对应的标记
    def convert_ids_to_tokens(self,indices):
        return [self.idx_to_token[index] for index in indices]


def load_reuters():
    from nltk.corpus import reuters #新闻类文档
    text = reuters.sents()
    text = [[word.lower() for word in sentence] for sentence in text]
    vocab = Vocab.build(text, reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]
    return corpus, vocab

#创建RNN语言模型的数据处理类RnnlmDataset
class RnnlmDataset(Dataset):
    def __init__(self, corpus, vocab):
        self.data = []
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        self.pad = vocab[PAD_TOKEN]
        
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            input = [self.bos] + sentence 
            target = sentence +[self.eos]
            self.data.append((input, target))
            # if len(sentence) < context_size:
            #     continue
            # for i in range(context_size, len(sentence)):
            #     context = sentence[i-context_size:i]
            #     target = sentence[i]
            #     self.data.append((context,target))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]     
    
    def collate_fn(self, examples):
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = [torch.tensor(ex[1]) for ex in examples]
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad)
        return (inputs, targets)

def get_loader(dataset, batch_size, shuffle = True):
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn,shuffle=shuffle)
    return data_loader

#FeedForwardNNLM模型
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLM,self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
        # self.activate =  F.relu
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden, _ = self.rnn(embeds)
        output = self.output(hidden)
        log_probs = F.log_softmax(output, dim=2)
        return log_probs
    
#训练
embedding_dim = 128
# embedding_dim = 64

hidden_dim = 256
# hidden_dim = 128

batch_size = 1024
# context_size = 3
num_epoch = 10

corpus, vocab = load_reuters()
dataset = RnnlmDataset(corpus, vocab)
data_loader = get_loader(dataset, batch_size)

#忽略pad的损失
nll_loss = nn.NLLLoss(ignore_index=dataset.pad)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNLM(len(vocab), embedding_dim, hidden_dim)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
total_losses = []
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(data_loader,desc=f"Training Epoch{epoch}"):
        inputs , targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs.view(-1, log_probs.shape[-1]),targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Loss:{total_loss:.2f}")
    total_losses.append(total_loss)

# 保存词向量

def save_pretrained(vocab, embeds, save_path):
    """
    Save pretrained token vectors in a unified format, where the first line
    specifies the `number_of_tokens` and `embedding_dim` followed with all
    token vectors, one token per line.
    """
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx_to_token):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")
    print(f"Pretrained embeddings saved to: {save_path}")

save_pretrained(vocab,model.embeddings.weight.data,"./rnnlm.vec")


