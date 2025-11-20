# -*- coding: utf-8 -*-
"""
[对应于 README 2.2 损失函数]

实现了自定义的组合排序损失函数 (Combined Ranking Loss)。

核心思想:
排序损失 (Ranking Loss) 的目标是让模型对 "Winner" 的打分高于 "Loser"。
一个简单的实现是 `max(0, margin - (score_winner - score_loser))`。

组合排序损失:
由于我们的模型是多头输出（每个审美维度一个头），我们需要将所有头的排序损失
综合起来，形成一个总的损失函数，以实现多任务学习。

这个文件中的 `CombinedRankingLoss` 类将：
1.  接收模型对 (Winner, Loser) 数据对的批量预测结果。
2.  对每一个预测头（审美维度），计算其对应的排序损失。
3.  根据预设的权重，将所有头的损失加权求和，得到最终的 total_loss。
"""
import torch
import torch.nn as nn

class CombinedRankingLoss(nn.Module):
    def __init__(self, head_weights, margin=1.0):
        """
        初始化组合排序损失。

        Args:
            head_weights (dict): 一个字典，定义了每个预测头的损失权重。
                                 e.g., {'total_score': 0.5, 'composition': 0.25, 'color': 0.25}
            margin (float): 排序损失中的间隔参数。
        """
        super().__init__()
        self.head_weights = head_weights
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, winner_scores, loser_scores):
        """
        计算总损失。

        Args:
            winner_scores (dict): 模型对 "Winner" 样本的预测得分字典。
                                  e.g., {'total_score': tensor, 'composition': tensor, ...}
            loser_scores (dict): 模型对 "Loser" 样本的预测得分字典。

        Returns:
            torch.Tensor: 加权求和后的总损失。
            dict: 一个包含总损失和每个头单独损失的字典，用于日志记录。
        """
        total_loss = 0.0
        loss_dict = {}

        # 目标是让 winner_score > loser_score，所以 target tensor 全是1
        target = torch.ones_like(next(iter(winner_scores.values())))

        for head_name, weight in self.head_weights.items():
            if head_name in winner_scores and head_name in loser_scores:
                head_loss = self.ranking_loss(winner_scores[head_name], loser_scores[head_name], target)
                weighted_loss = weight * head_loss
                total_loss += weighted_loss
                loss_dict[f'loss_{head_name}'] = head_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
