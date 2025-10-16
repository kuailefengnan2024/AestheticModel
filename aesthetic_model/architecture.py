# -*- coding: utf-8 -*-
"""
[对应于 README 2.1 架构定义]

定义了整个审美模型的神经网络结构。

主要包含两个部分：
1.  SharedEncoder: 一个共享的编码器，用于从输入的图片和文本中提取特征。
    - 视觉部分: 使用预训练的 OpenCLIP 模型。
    - 文本部分: 使用预训练的 BERT 或其他语言模型。
2.  MultiHeadMLP: 多个独立的、轻量级的MLP预测头。
    - 每个头对应一个审美维度（例如总分、构图、色彩等）。
    - 接收SharedEncoder输出的特征，并预测该维度的得分。
"""
import torch
import torch.nn as nn
import open_clip

class AestheticModel(nn.Module):
    def __init__(self, config):
        """
        初始化模型。

        Args:
            config (object): 包含所有模型超参数的配置对象。
                             例如: config.vision_model_name, config.pretrained
        """
        super().__init__()
        
        # 1. 加载预训练的CLIP模型 (视觉和文本编码器会一并加载)
        #    open_clip 会自动处理下载和缓存
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.vision_model_name, 
            pretrained=config.pretrained
        )
        
        self.visual = model.visual
        self.text = model.text # 使用CLIP自带的文本编码器
        self.tokenizer = open_clip.get_tokenizer(config.vision_model_name)

        # TODO: 2. 定义共享编码器的融合策略（如果需要）
        #    - 例如，获取 embedding_dim
        #    - embedding_dim = self.visual.output_dim 

        # TODO: 3. 根据config中的定义，动态创建多个MLP预测头
        pass

    def forward(self, image, text):
        """
        模型的前向传播。

        Args:
            image (torch.Tensor): 经过预处理的图片Tensor。
            text (torch.Tensor): 经过tokenizer处理的文本Tensor。

        Returns:
            dict: 一个字典，key是每个预测头的名称，value是对应的预测得分。
                  e.g., {'total_score': tensor, 'composition': tensor, ...}
        """
        # TODO: 1. 从图片和文本中提取特征
        # TODO: 2. 将特征输入到每一个MLP头中，得到所有维度的预测分数
        # TODO: 3. 返回包含所有分数的字典
        pass
