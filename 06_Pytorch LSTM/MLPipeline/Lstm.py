import torch
import torch.nn as nn

class LSTM(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):  #
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # self.drop  = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_dim, 64, batch_first=True)
        self.linear = nn.Linear(64, 5)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embeddings(x)
        # x = self.drop(x)
        out_pack, (ht, ct) = self.lstm(x)
        out_pack1, (ht, ct) = self.lstm1(out_pack)
        out = self.linear(ht[-1])
        # out = self.softmax(out)
        return out
