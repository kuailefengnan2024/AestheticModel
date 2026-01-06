"""
数据流水线协议定义模块 (Data Pipeline Schemas)

角色: 协议层 (Protocol Layer)
功能: 定义数据流水线中各阶段的标准数据结构 (Pydantic Models)，确保上下游数据一致性。
核心模型:
  - GeneratedPair: [中间态] 记录生图阶段产出的图片对元数据。
  - TrainingData:  [终态] 记录打分阶段产出的完整训练样本 (Prompt + Images + Scores)。
"""
from typing import Dict, Optional, List, Union
from pydantic import BaseModel, Field

# ==============================================================================
# 1. 评分相关结构 (Scoring Structures)
# ==============================================================================

class ScoreDetail(BaseModel):
    """
    单个维度在两张图上的得分。
    """
    winner: float = Field(..., description="胜者图片在该维度的得分")
    loser: float = Field(..., description="败者图片在该维度的得分")

class Scores(BaseModel):
    """
    所有维度的评分集合。
    支持的维度应与 VLM 的裁判 Prompt 保持一致。
    """
    total: ScoreDetail
    composition: ScoreDetail
    color: ScoreDetail
    lighting: ScoreDetail
    # 可根据需要添加更多维度，如 'creativity', 'text_match' 等

# ==============================================================================
# 2. 中间过程结构 (Intermediate Structures)
# ==============================================================================

class GeneratedPair(BaseModel):
    """
    Step 1 生成的图片对记录。
    存储在 data/intermediate/generated_pairs.jsonl 中。
    """
    prompt_id: str
    prompt: str
    width: int
    height: int
    image_a_path: str
    image_b_path: str
    model_name: str
    # 可选：记录生成的 seed，虽然 Seedream 4.5 可能不支持指定 seed，但 3.0 支持
    seed_a: Optional[str] = None
    seed_b: Optional[str] = None
    timestamp: float

# ==============================================================================
# 3. 最终训练数据结构 (Final Training Data Structure)
# ==============================================================================

class TrainingData(BaseModel):
    """
    Step 2 打分完成后的最终训练数据。
    存储在 data/preferences_train.jsonl 中。
    """
    prompt: str
    
    # 明确指出文件的物理路径
    image_winner_path: str
    image_loser_path: str
    
    # 具体的 Logits (分数)，用于计算 Loss
    # 注意：winner_score['total'] 必须 > loser_score['total']
    scores: Scores
    
    # VLM 给出的判词/理由，用于可解释性分析或后续优化
    reasoning: Optional[str] = Field(None, description="VLM对评分优劣的核心理由")
    
    # 元数据
    source: str = "synthetic_vlm_judge"
    prompt_id: Optional[str] = None

