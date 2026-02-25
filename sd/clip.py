import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

# 将文本 token 转为向量表示
class CLIPEmbedding(nn.Module):

    # n_vocab: 词汇表大小，典型值 49408
    # n_embd: 嵌入维度，典型值 768
    # n_tokens: 最大序列长度，典型值 77
    def __init__(self, n_vocab: int, n_embd: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)

        # Position Embedding: 添加位置信息(可学习参数,不是正弦编码)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x

# 标准 Transformer 编码器层,包含两个子层
class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embd : int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, Seq_Len, Dim)

        ## SELF ATTENTIOIN

        residue = x

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask = True)

        x += residue

        ## FEEDFORWARD LAYER

        residue = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # Quick GLU activation function 

        x = self.linear_2(x)

        x += residue

        return x

class CLIP(nn.Module): # Transformer

    def __init__(self):
        super().__init__()

        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)


        # (Batch_Size, Sequence_Len) -> (Batch_Size, Sequence_Len, Dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        return output