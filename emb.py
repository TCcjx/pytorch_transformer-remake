"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: emb.py.py
 @DateTime: 2025-05-14 15:26
 @SoftWare: PyCharm
 输入词序列，然后将词序列 id 向量化，再给id附加位置信息
"""
import math
from torch import nn
import torch
from dataset import de_vocab, de_preprocess, train_dataset


# 将词Id转化为 嵌入向量表示 并 添加位置编码信息
class EmbeddingWithPosition(nn.Module):
    """
    输入带有批次信息 句子 ID列表
    返回 带有位置信息的嵌入表示
    """
    def __init__(self, vocab_size, emb_size, dropout=0.1, seq_max_len=5000):
        super().__init__()

        # 将词ID转化为嵌入向量表示
        self.seq_emb = nn.Embedding(vocab_size, emb_size)

        # 为序列中的每个位置准备一个位置向量， 也就是emb_size宽
        position_idx = torch.arange(0, seq_max_len, dtype=torch.float).unsqueeze(-1)
        position_emb_fill = position_idx * torch.exp(-torch.arange(0,emb_size,2) * math.log(10000.0) / emb_size)
        position_encoding = torch.zeros(seq_max_len, emb_size, dtype = torch.float)
        position_encoding[:,0::2] = torch.sin(position_emb_fill)
        position_encoding[:,1::2] = torch.cos(position_emb_fill)
        self.register_buffer('position_encoding', position_encoding) # 固定参数，不需要train

        # 防过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.seq_emb(x)
        x = x + self.position_encoding.unsqueeze(0)[:, :x.size()[1], :]
        return self.dropout(x)

if __name__ == '__main__':
    emb = EmbeddingWithPosition(len(de_vocab), 128)

    de_tokens, de_ids = de_preprocess(train_dataset[0][0])
    de_ids_tensor = torch.tensor(de_ids, dtype=torch.long)

    emb_result = emb(de_ids_tensor.unsqueeze(0))
    print('de_ids_tensor:', de_ids_tensor.size(), 'emb_result:', emb_result.size())
