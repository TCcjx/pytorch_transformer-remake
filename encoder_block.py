"""
encoder block支持堆叠, 每个block都输入emb序列并输出emb序列(1:1对应)
因为要支持encoder block进行堆叠，所以输入输出的尺寸应该保持一致
"""
from torch import nn
import torch
from multihead_attn import MultiHeadAttention
from emb import EmbeddingWithPosition
from dataset import de_preprocess, train_dataset, de_vocab # 德语词表


# 编码器模块
class EncoderBlock(nn.Module):
    def __init__(self, emb_size, q_k_size, v_size, f_size, head): # f_size feedforward模块中间维度

        super(EncoderBlock, self).__init__()

        self.multihead_attn = MultiHeadAttention(emb_size, q_k_size, v_size, head) # 多头注意力模块
        self.z_linear=nn.Linear(head*v_size,emb_size) # 调整多头注意力的输出尺寸大小，恢复成输入的大小
        self.addnorm1=nn.LayerNorm(emb_size) # 按last dim做norm

        # feed-forward模块
        self.feedforward = nn.Sequential(
            nn.Linear(emb_size,f_size),
            nn.ReLU(),
            nn.Linear(f_size,emb_size)
        )
        self.addnorm2=nn.LayerNorm(emb_size)

    def forward(self,x,attn_mask): # x: (batch_size, seq_len, emb_size)
        z = self.multihead_attn(x,x,attn_mask) # z: (batch_size, seq_len, head*v_size)
        z = self.z_linear(z) # z: (batch_size, seq_len, emb_size)
        output = self.addnorm1(z + x)

        z = self.feedforward(output) # z: (batch_size, seq_len, emb_size)
        return self.addnorm2(z + output) # (batch_size, seq_len, emb_size)

if __name__ == '__main__':
    emb = EmbeddingWithPosition(vocab_size=len(de_vocab), emb_size=256) # 带位置编码信息的嵌入编码模块
    de_tokens,de_ids = de_preprocess(train_dataset[0][0]) # 取德语句子的ID序列
    de_ids_tensor = torch.tensor(de_ids,dtype=torch.long)
    emb_result = emb(de_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    print('emb_result:', emb_result.size()) # (1, 16, 128)

    attn_mask = torch.zeros((1, de_ids_tensor.size()[0], de_ids_tensor.size()[0])) # (1,seq_len,seq_len)

    # 5个Encoder block堆叠
    encoder_blocks = []
    for i in range(5):
        encoder_blocks.append(EncoderBlock(emb_size=256, q_k_size=256, v_size=512, f_size=1024, head=8))

    # 前向forward
    encoder_outputs = emb_result
    for i in range(5):
        encoder_outputs= encoder_blocks[i](encoder_outputs,attn_mask)
    print('encoder_outputs:',encoder_outputs.shape) # (1, 16, 128)
