# -*- coding: utf-8 -*-
"""
[对应于 README 2.1 架构定义]

定义了整个审美模型的神经网络结构。
采用双塔结构 (Two-Tower Architecture) + 多头 MLP (Multi-Head MLP)。

架构细节:
1. Vision Encoder: OpenCLIP (ViT-L-14)
   - 提取图像的高维特征。
2. Text Encoder: XLM-RoBERTa
   - 提取 Prompt 的语义特征，支持中英文混合。
3. Fusion Layer:
   - 由于 Vision 和 Text 来自不同的预训练空间，我们需要一个投影层(Projection)
     将它们映射到同一个维度，然后进行融合 (Concatenation)。
4. Regression Heads:
   - 多个独立的 MLP 用于预测不同的审美维度。
"""
import torch
import torch.nn as nn
import open_clip
from transformers import AutoModel, AutoConfig

class AestheticModel(nn.Module):
    def __init__(self, config):
        """
        初始化模型。

        Args:
            config (object): 配置对象 (Namespace 或 dict)，包含:
                             - vision_model_name, vision_pretrained
                             - text_model_name
                             - projection_dim, mlp_hidden_dim
                             - heads (dict)
        """
        super().__init__()
        self.config = config
        
        # ======================================================================
        # 1. Vision Encoder (OpenCLIP)
        # ======================================================================
        print(f"Loading Vision Model: {config.vision_model_name} ({config.vision_pretrained})...")
        # create_model_and_transforms 返回 (model, train_transform, val_transform)
        # 我们只需要 model。在训练脚本中会单独处理 transform。
        self.vision_backbone, _, _ = open_clip.create_model_and_transforms(
            config.vision_model_name, 
            pretrained=config.vision_pretrained
        )
        # 冻结视觉部分的大部分参数? 通常在微调初期可以冻结，或者使用很小的 LR。
        # 这里我们默认开启梯度，允许微调。
        self.vision_hidden_dim = self.vision_backbone.visual.output_dim
        
        # ======================================================================
        # 2. Text Encoder (HuggingFace Transformers)
        # ======================================================================
        print(f"Loading Text Model: {config.text_model_name}...")
        self.text_backbone = AutoModel.from_pretrained(config.text_model_name)
        self.text_hidden_dim = self.text_backbone.config.hidden_size

        # ======================================================================
        # 3. Fusion Layer (特征融合)
        # ======================================================================
        # 我们将 vision 和 text 特征都投影到 projection_dim
        self.projection_dim = config.projection_dim
        
        self.vision_projector = nn.Sequential(
            nn.Linear(self.vision_hidden_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU()
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_hidden_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.GELU()
        )
        
        # 融合后的维度 (Concat)
        self.combined_dim = self.projection_dim * 2

        # ======================================================================
        # 4. Multi-Head Prediction Heads (多头预测)
        # ======================================================================
        self.heads = nn.ModuleDict()
        
        for head_name in config.heads.keys():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(self.combined_dim, config.mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.mlp_hidden_dim, 1) # 输出一个标量分数
            )
            
        print("Model initialized successfully.")

    def forward(self, image, input_ids, attention_mask=None):
        """
        模型前向传播。

        Args:
            image (torch.Tensor): 图片张量 [batch_size, channels, height, width]
            input_ids (torch.Tensor): 文本Token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): 文本Mask [batch_size, seq_len]

        Returns:
            dict: {head_name: score_tensor (batch_size, 1)}
        """
        # 1. Extract Vision Features
        # OpenCLIP 的 encode_image 通常返回归一化的特征，但我们也可以用未归一化的
        vision_features = self.vision_backbone.encode_image(image) # [batch, vision_dim]
        vision_features = vision_features.float() # 确保是 float32 (如果用了混合精度)
        
        # 2. Extract Text Features
        # HuggingFace 模型输出通常是一个 object，last_hidden_state 是 [batch, seq, dim]
        # 我们取 [CLS] token (index 0) 或者做 mean pooling
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 取 CLS token 对应的向量作为句子表示
        # 对于 Roberta 类模型，通常取第一个 token
        text_features = text_outputs.last_hidden_state[:, 0, :] # [batch, text_dim]
        
        # 3. Projection & Fusion
        v_proj = self.vision_projector(vision_features) # [batch, proj_dim]
        t_proj = self.text_projector(text_features)     # [batch, proj_dim]
        
        # 拼接特征
        combined_features = torch.cat((v_proj, t_proj), dim=1) # [batch, proj_dim * 2]
        
        # 4. Multi-Head Prediction
        outputs = {}
        for head_name, head_layer in self.heads.items():
            outputs[head_name] = head_layer(combined_features)
            
        return outputs
