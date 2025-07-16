import torch
from torch import nn
from torch.nn.functional import softmax

class Attention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        if self.hidden_dim % self.n_heads != 0:
            raise TypeError("hidden_dim should be divisible by n_heads")
        self.head_dim = self.hidden_dim // self.n_heads

        self.wq = nn.Linear(self.head_dim, self.head_dim)
        self.wk = nn.Linear(self.head_dim, self.head_dim)

    def forward(self, x):
        Q = self.wq(x) # [1, 1, 12, 512]
        K = self.wk(x) # [1, 1, 512, 12]
        scores = Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        return scores

class AttentionContext(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.attn_score = Attention(hidden_dim, n_heads)
        self.wv = nn.Linear(self.head_dim, self.head_dim)
        self.wo = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, self.n_heads, seq_len, self.head_dim)
        V = self.wv(x) 
        attn_weights = self.attn_score(x)
        attn_weights = softmax(attn_weights, dim=-1) 
        x = attn_weights @ V 
        x = x.view(batch_size, seq_len, self.head_dim * self.n_heads)
        x = self.wo(x)
        return x

if __name__ == '__main__':
    attn = AttentionContext(512, 4)
    x = torch.randn(1, 12, 512)
    output = attn(x)
    print(output.shape)  # [1, 4, 512]
