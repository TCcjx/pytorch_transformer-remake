# pytorch_transformer
复现transformer架构，用于德语->英文的翻译任务

# multi-Attention的实现
在本项目中multi-Attention的实现，方便了编码器和解码器的多头注意力机制的使用
这里的实现代码比较简洁，主要是为了方便解码器模块的第二个多头注意力机制的传参，提高代码的复用率

# 整体文件架构
decoder_blcok -> decoder

encoder_block -> encoder

emb -> 嵌入表示 + 位置信息

dataset -> 构建词表 + de/en分词器

config -> 全局配置信息，DEVICE的调用
