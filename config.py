"""
 @Author: TCcjx
 @Email: tcc2025@163.com
 @FileName: config.py.py
 @DateTime: 2025-05-14 14:41
 @SoftWare: PyCharm
"""
import torch

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输入的最长序列长度 （受限于 position emb）
SEQ_MAX_LEN = 5000

if __name__ == '__main__':
    print(DEVICE)
    print(type('DEVICE'))