# -*- coding: utf-8 -*-
"""
核心模型架构 (AestheticModel - Dual Encoder Version)

功能:
1.  **Vision Backbone**: 使用 Transformers CLIPModel (ViT-L-14) 提取图像特征。
2.  **Text Backbone**: 使用 Transformers CLIPModel 提取 Prompt 特征。
3.  **Feature Fusion**: 
    - 纯视觉任务 (构图/色彩): 只使用 Vision Features。
    - 图文任务 (一致性/总分): 使用 Vision + Text Features 拼接。
4.  **Multi-Head MLP**: 独立的预测头。
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPConfig

class AestheticScorer(nn.Module):
    def __init__(self, config):
        """
        初始化模型。
        config: 需包含 vision_model_name (path or name), mlp_hidden_dim, heads
        """
        super().__init__()
        self.config = config
        
        # 1. Load CLIP (Vision + Text)
        print(f"Loading CLIP Model (Transformers): {config.vision_model_name}...")
        
        # 从预训练加载 (支持本地路径或HF Hub ID)
        self.clip = CLIPModel.from_pretrained(config.vision_model_name)
        
        # 自动获取维度
        self.vision_dim = self.clip.config.projection_dim
        self.text_dim = self.clip.config.projection_dim
        
        # Freezing Backbone (Optional)
        if getattr(config, "freeze_backbone", False):
            print("❄️ Freezing CLIP Backbone (Vision + Text)...")
            for param in self.clip.parameters():
                param.requires_grad = False
            # Ensure it stays in eval mode (disable dropout/batchnorm updates in backbone)
            self.clip.eval()
        
        # 2. Heads (Multi-Task)
        self.heads = nn.ModuleDict()
        
        # 定义哪些 Head 需要 Text 特征
        self.multimodal_heads = ["total", "text_alignment"]
        
        # Dropout rate
        dropout_rate = getattr(config, "dropout", 0.1)
        
        for head_name in config.heads:
            # 决定输入维度
            if head_name in self.multimodal_heads:
                input_dim = self.vision_dim + self.text_dim
            else:
                input_dim = self.vision_dim
                
            self.heads[head_name] = nn.Sequential(
                nn.Linear(input_dim, config.mlp_hidden_dim),
                nn.LayerNorm(config.mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(config.mlp_hidden_dim, 1) # Scalar Score
            )
            
        # Log
        print(f"Model Initialized. Vision Dim: {self.vision_dim}, Text Dim: {self.text_dim}")
        print(f"Dropout Rate: {dropout_rate}, Backbone Frozen: {getattr(config, 'freeze_backbone', False)}")

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        """
        Args:
            pixel_values: [B, 3, H, W]
            input_ids:    [B, SeqLen] (Optional, for text alignment)
            attention_mask: (Optional, for text encoder)
        """
        # 1. Vision Features
        # get_image_features 返回 projection 后的特征 (normalized)
        v_emb = self.clip.get_image_features(pixel_values=pixel_values) # [B, proj_dim]
        
        # 2. Text Features (Optional)
        t_emb = None
        if input_ids is not None:
            # get_text_features 返回 projection 后的特征 (normalized)
            # 注意: transformers 的 text encoder 需要 attention_mask (padding mask)
            # 如果 Dataset 没传 mask，通常 0 是 padding
            t_emb = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask) # [B, proj_dim]
        
        # 3. Multi-Head Prediction
        outputs = {}
        for name, head in self.heads.items():
            if name in self.multimodal_heads:
                if t_emb is None:
                    # 如果没有提供 text，暂时报错
                    raise ValueError(f"Head '{name}' requires text input.")
                
                # Concat
                combined = torch.cat([v_emb, t_emb], dim=1) # [B, V+T]
                outputs[name] = head(combined).squeeze(-1)
            else:
                # 纯视觉
                outputs[name] = head(v_emb).squeeze(-1)
            
        return outputs

# 配置类
class ModelConfig:
    def __init__(self, **kwargs):
        # 默认指向 HF Hub ID，也可以改为本地路径 "models/clip-vit-large-patch14"
        self.vision_model_name = "openai/clip-vit-large-patch14"
        self.mlp_hidden_dim = 1024
        self.heads = ["total", "composition", "color", "lighting"]
        self.freeze_backbone = False
        self.dropout = 0.1
        self.__dict__.update(kwargs)
