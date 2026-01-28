# -*- coding: utf-8 -*-
"""
核心数据集模块 (Core Dataset & Preprocessing)

功能:
1.  **AestheticDataset**: 
    - 负责加载 JSONL 格式的成对偏好数据 (Winner/Loser)。
    - 执行数据预检 (Pre-check)，自动过滤掉图片缺失的样本，确保训练鲁棒性。
    - 将原始数据转换为 PyTorch Tensor，并提取多维度评分 Label。

2.  **DynamicResizePad (关键预处理)**:
    - 实现了本项目核心的 "动态尺寸支持" 逻辑。
    - **Resize**: 保持长宽比缩放图片，避免拉伸变形。
    - **Padding**: 将缩放后的图片填充至标准正方形 (如 224x224)。
    - **Masking**: 生成 pixel-level 的 Attention Mask (1=有效, 0=填充)，指导模型忽略黑边。

注意:
    本模块生成的 Tensor 是内存对象，直接供给 DataLoader 使用，不会保存为物理文件。
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from transformers import CLIPTokenizer, AltCLIPProcessor

from utils.logger import logger

class DynamicResizePad:
    """
    [预处理核心] 动态缩放与填充逻辑。
    将任意比例图片缩放并填充至正方形，同时生成 Attention Mask。
    """
    def __init__(self, target_size: int = 224, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        执行转换。
        Returns:
            {
                "pixel_values": [3, H, W]  (Normalized Tensor),
                "pixel_mask":   [1, H, W]  (0/1 Mask, 1=Image, 0=Padding)
            }
        """
        # 1. 保持比例缩放 (Resize)
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 使用 BICUBIC 保证质量
        resized_image = image.resize((new_w, new_h), Image.BICUBIC)
        
        # 2. 转为 Tensor 并归一化
        # F.to_tensor 会归一化到 [0, 1]
        img_tensor = F.to_tensor(resized_image) 
        # Normalize (使用 CLIP 默认均值方差)
        img_tensor = F.normalize(img_tensor, self.mean, self.std)
        
        # 3. 创建画布并填充 (Padding)
        # 初始化一个全0 (或全黑) 的画布 [3, target_size, target_size]
        # 注意：这里填充的是 0，但因为已经 Normalize 过了，0 并不代表黑色。
        # 更好的做法是用 0 填充，然后依靠 Mask 让模型忽略它。
        padded_image = torch.zeros((3, self.target_size, self.target_size), dtype=img_tensor.dtype)
        
        # 贴在左上角 (0, 0)
        # 也可以做 Center Padding，但左上角计算最简单
        padded_image[:, :new_h, :new_w] = img_tensor
        
        # 4. 生成 Mask (Attention Mask)
        # 1.0 = 有效区域, 0.0 = 填充区域
        # 形状 [1, H, W] 方便后续卷积或 Patch 处理
        pixel_mask = torch.zeros((1, self.target_size, self.target_size), dtype=torch.float32)
        pixel_mask[:, :new_h, :new_w] = 1.0
        
        return {
            "pixel_values": padded_image,
            "pixel_mask": pixel_mask
        }

class AestheticDataset(Dataset):
    def __init__(self, data_path: str, target_size: int = 224, model_name: str = "BAAI/AltCLIP", image_dir: Optional[str] = None):
        """
        初始化数据集。

        Args:
            data_path: jsonl 文件路径
            target_size: 图片输入尺寸 (默认 224 for CLIP)
            model_name: 模型名称 (默认为 BAAI/AltCLIP)
            image_dir: (可选) 图片根目录覆盖。
        """
        self.data_path = Path(data_path)
        self.target_size = target_size
        self.image_dir_override = Path(image_dir) if image_dir else None
        
        # 初始化 Processor (AltCLIP 使用 Processor 同时处理文本和图片预处理逻辑)
        # 但我们这里自定义了图片预处理 (DynamicResizePad)，所以主要用它来处理文本
        logger.info(f"Loading Processor: {model_name}...")
        try:
            self.processor = AltCLIPProcessor.from_pretrained(model_name)
            self.use_altclip = True
        except Exception:
            logger.warning(f"Failed to load AltCLIPProcessor for {model_name}, falling back to CLIPTokenizer.")
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.use_altclip = False
        
        # 初始化预处理器
        self.preprocessor = DynamicResizePad(target_size=target_size)
        
        if self.image_dir_override:
            logger.info(f"Using Image Directory Override: {self.image_dir_override}")
        
        # 加载并过滤数据
        self.samples = self._load_and_filter_data()
        logger.info(f"Dataset 初始化完成: 加载了 {len(self.samples)} 条有效样本。")

    def _load_and_filter_data(self) -> List[Dict]:
        """加载数据并预检图片是否存在"""
        valid_samples = []
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
            
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    item = json.loads(line)
                    
                    # 检查路径
                    p_winner = Path(item["image_winner_path"])
                    p_loser = Path(item["image_loser_path"])
                    
                    # 如果有 override，检查 override 路径下的文件是否存在
                    if self.image_dir_override:
                        p_winner = self.image_dir_override / p_winner.name
                        p_loser = self.image_dir_override / p_loser.name

                    if p_winner.exists() and p_loser.exists():
                        valid_samples.append(item)
                    else:
                        pass
                except Exception as e:
                    logger.warning(f"解析样本 {line_idx} 失败: {e}")
                    
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. 读取图片
        # 路径处理逻辑：如果有 override，使用 override 目录 + 文件名
        if self.image_dir_override:
            winner_path = self.image_dir_override / Path(item["image_winner_path"]).name
            loser_path = self.image_dir_override / Path(item["image_loser_path"]).name
        else:
            winner_path = item["image_winner_path"]
            loser_path = item["image_loser_path"]

        try:
            img_winner = Image.open(winner_path).convert("RGB")
            img_loser = Image.open(loser_path).convert("RGB")
        except Exception as e:
            logger.error(f"读取图片失败: {winner_path} / {loser_path} - {e}")
            return self.__getitem__(np.random.randint(len(self)))

        # 2. 预处理 (Dynamic Padding)
        processed_winner = self.preprocessor(img_winner)
        processed_loser = self.preprocessor(img_loser)
        
        # 3. 处理分数 (Labels)
        scores = item["scores"]
        dims = ["total", "composition", "color", "atmosphere", "text_alignment", "coherence"]
        
        winner_scores = []
        loser_scores = []
        
        for d in dims:
            # 兼容性：如果旧数据没有新维度，返回 -1.0 (mask 标记)
            # 注意：JSONL读取后 scores[d] 可能不存在，或者存在但为 None
            s_win = scores.get(d, {}).get("winner", -1.0)
            s_lose = scores.get(d, {}).get("loser", -1.0)
            
            # 双重保险：如果是 None 也转为 -1.0
            if s_win is None: s_win = -1.0
            if s_lose is None: s_lose = -1.0
                
            winner_scores.append(float(s_win))
            loser_scores.append(float(s_lose))
        
        # 4. 处理文本 (Tokenize)
        # AltCLIP Processor handles text tokenization (padding, truncation, etc.)
        if self.use_altclip:
            text_inputs = self.processor(
                text=item["prompt"], 
                padding="max_length", 
                truncation=True, 
                max_length=77, 
                return_tensors="pt"
            )
        else:
            text_inputs = self.tokenizer(
                item["prompt"], 
                padding="max_length", 
                truncation=True, 
                max_length=77, 
                return_tensors="pt"
            )
        
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        return {
            "winner": processed_winner, # {pixel_values, pixel_mask}
            "loser": processed_loser,   # {pixel_values, pixel_mask}
            "scores_winner": torch.tensor(winner_scores, dtype=torch.float32),
            "scores_loser": torch.tensor(loser_scores, dtype=torch.float32),
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

def collate_fn(batch):
    """
    自定义 Collate 函数，处理嵌套字典。
    """
    pixel_values_winner = torch.stack([item["winner"]["pixel_values"] for item in batch])
    pixel_mask_winner = torch.stack([item["winner"]["pixel_mask"] for item in batch])
    
    pixel_values_loser = torch.stack([item["loser"]["pixel_values"] for item in batch])
    pixel_mask_loser = torch.stack([item["loser"]["pixel_mask"] for item in batch])
    
    scores_winner = torch.stack([item["scores_winner"] for item in batch])
    scores_loser = torch.stack([item["scores_loser"] for item in batch])
    
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    return {
        "winner_pixel_values": pixel_values_winner,
        "winner_pixel_mask": pixel_mask_winner,
        "loser_pixel_values": pixel_values_loser,
        "loser_pixel_mask": pixel_mask_loser,
        "scores_winner": scores_winner,
        "scores_loser": scores_loser,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
