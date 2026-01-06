# -*- coding: utf-8 -*-
"""
核心模型架构 (AestheticModel - Dual Encoder Version)

功能:
1.  **Vision Backbone**: 使用 OpenCLIP (ViT-L-14) 提取图像特征。
2.  **Text Backbone**: 使用 OpenCLIP 对应的 Text Transformer 提取 Prompt 特征。
3.  **Feature Fusion**: 针对不同任务进行特征选择性融合。
    - 纯视觉任务 (构图/色彩): 只使用 Vision Features。
    - 图文任务 (一致性/总分): 使用 Vision + Text Features 拼接。
4.  **Multi-Head MLP**: 独立的预测头。
"""
import torch
import torch.nn as nn
import open_clip

class AestheticScorer(nn.Module):
    def __init__(self, config):
        """
        初始化模型。
        config: 需包含 vision_model_name, vision_pretrained, mlp_hidden_dim, heads
        """
        super().__init__()
        self.config = config
        
        # 1. Load CLIP (Vision + Text)
        print(f"Loading CLIP Model: {config.vision_model_name} (pretrained={config.vision_pretrained})...")
        clip_model, _, _ = open_clip.create_model_and_transforms(
            config.vision_model_name, 
            pretrained=config.vision_pretrained
        )
        self.visual = clip_model.visual
        self.text = clip_model.text
        
        # 自动获取维度
        # OpenCLIP 的 visual 和 text 模块通常都有 output_dim 属性
        # 注意: OpenCLIP 的 forward 可能会返回 projected features (e.g. 768)
        self.vision_dim = self.visual.output_dim
        self.text_dim = self.text.output_dim
        
        # 2. Heads (Multi-Task)
        self.heads = nn.ModuleDict()
        
        # 定义哪些 Head 需要 Text 特征
        self.multimodal_heads = ["total", "text_alignment"]
        
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
                nn.Dropout(0.1),
                nn.Linear(config.mlp_hidden_dim, 1) # Scalar Score
            )
            
        # Log
        print(f"Model Initialized. Vision Dim: {self.vision_dim}, Text Dim: {self.text_dim}")

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        """
        Args:
            pixel_values: [B, 3, H, W]
            input_ids:    [B, SeqLen] (Optional, for text alignment)
            attention_mask: (Optional, for text encoder)
        """
        # 1. Vision Features
        # visual(x) returns projected features by default
        v_emb = self.visual(pixel_values) # [B, vision_dim]
        
        # 2. Text Features (Optional)
        t_emb = None
        if input_ids is not None:
            # text(x) returns projected features
            t_emb = self.text(input_ids) # [B, text_dim]
            
        # 3. Multi-Head Prediction
        outputs = {}
        for name, head in self.heads.items():
            if name in self.multimodal_heads:
                if t_emb is None:
                    # 如果没有提供 text，但 head 需要 text，这就尴尬了
                    # 临时方案：用 0 填充 text 部分，或者报错
                    # 这里假设训练时一定有 text
                    raise ValueError(f"Head '{name}' requires text input, but input_ids is None.")
                
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
        self.vision_model_name = "ViT-L-14"
        self.vision_pretrained = "openai"
        self.mlp_hidden_dim = 1024
        # 默认包含 text_alignment
        self.heads = ["total", "composition", "color", "lighting", "text_alignment"]
        self.__dict__.update(kwargs)
