# -*- coding: utf-8 -*-
"""
[对应于 README 2.2 损失函数]

实现了自定义的组合排序损失函数 (Combined Ranking Loss)。

核心思想 (Industrial SOTA):
1.  **Ranking Loss**: 使用 Bradley-Terry 模型 (BCEWithLogitsLoss) 来优化成对排序。
    这是 RLHF 中最标准的做法，比 MarginRankingLoss 更稳定，且不需要调 margin 参数。
2.  **Regression Loss (Auxiliary)**: 辅助的 MSE Loss。
    纯 Ranking Loss 只能学到 "A > B"，但会导致分数漂移 (scale drift)。
    我们引入回归损失，强迫模型预测的分数尽可能接近 VLM 给出的绝对分值 (Ground Truth)，
    从而把分数 "锚定" 在 0-10 分的合理区间内。

公式:
    Total Loss = w1 * Rank_Loss + w2 * Reg_Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedRankingLoss(nn.Module):
    def __init__(self, rank_weight=1.0, reg_weight=0.1):
        """
        初始化组合损失函数。

        Args:
            rank_weight (float): 排序损失的权重。
            reg_weight (float): 回归损失的权重。通常给一个小权重，作为正则化项。
        """
        super().__init__()
        self.rank_weight = rank_weight
        self.reg_weight = reg_weight
        
        # 排序损失: BCEWithLogitsLoss
        # 目标是预测 P(A > B) = 1 (如果 A 是 winner)
        # 输入是 logits_diff = score_A - score_B
        self.ranking_loss = nn.BCEWithLogitsLoss()
        
        # 回归损失: MSELoss
        self.regression_loss = nn.MSELoss()

    def forward(self, pred_scores_winner, pred_scores_loser, gt_scores_winner=None, gt_scores_loser=None):
        """
        计算总损失。支持多维度输入 (dict of tensors)。

        Args:
            pred_scores_winner (dict): 模型对 Winner 的预测 logits。 {dim: [B]}
            pred_scores_loser (dict): 模型对 Loser 的预测 logits。 {dim: [B]}
            gt_scores_winner (Tensor): [B, num_dims] Winner 的真实分数 (0-10)。
            gt_scores_loser (Tensor): [B, num_dims] Loser 的真实分数 (0-10)。

        Returns:
            torch.Tensor: 加权求和后的总损失 scalar。
            dict: 用于日志的详细 loss 字典。
        """
        total_loss = 0.0
        loss_dict = {}

        # 假设所有 dim 的权重均等，直接求和
        # 如果需要不同维度不同权重，可以扩展 head_weights
        
        # 1. Ranking Loss (Pairwise)
        rank_loss_sum = 0.0
        dims_count = 0
        
        for dim_name in pred_scores_winner.keys():
            s_win = pred_scores_winner[dim_name]
            s_lose = pred_scores_loser[dim_name]
            
            # 预测分差
            diff = s_win - s_lose
            
            # Target: 1.0 (代表 Winner 应该赢)
            target = torch.ones_like(diff)
            
            # Loss = -log(sigmoid(s_win - s_lose))
            l_rank = self.ranking_loss(diff, target)
            
            rank_loss_sum += l_rank
            loss_dict[f'rank_loss/{dim_name}'] = l_rank.item()
            dims_count += 1
            
        avg_rank_loss = rank_loss_sum / max(dims_count, 1)
        loss_dict['avg_rank_loss'] = avg_rank_loss.item()
        
        # 2. Regression Loss (Pointwise, Optional)
        reg_loss_sum = 0.0
        
        if gt_scores_winner is not None and gt_scores_loser is not None:
            # gt_scores 顺序: [total, composition, color, lighting, text_alignment]
            # 我们需要确保顺序对应。这里简化处理：假设调用者已经把 GT 整理好了
            # 实际上，最好传入 dict，但为了 Tensor 效率，我们假设 dims 顺序一致
            
            # 为了简单，我们只计算 Total Score 的回归损失，或者所有维度的
            # 假设 dim_names 列表是固定的
            ordered_dims = ["total", "composition", "color", "lighting"] 
            # 注意: gt_scores 的列顺序必须和这个对应
            
            # 展平预测值为 Tensor: [B, num_dims]
            # 过滤掉不在 ordered_dims 里的 (比如 text_alignment 如果 GT 没有)
            preds_win_list = []
            preds_lose_list = []
            valid_gt_indices = []
            
            for i, dim in enumerate(ordered_dims):
                if dim in pred_scores_winner:
                    preds_win_list.append(pred_scores_winner[dim])
                    preds_lose_list.append(pred_scores_loser[dim])
                    valid_gt_indices.append(i)
            
            if preds_win_list:
                preds_win_tensor = torch.stack(preds_win_list, dim=1)
                preds_lose_tensor = torch.stack(preds_lose_list, dim=1)
                
                # 取出对应的 GT
                gt_win_tensor = gt_scores_winner[:, valid_gt_indices]
                gt_lose_tensor = gt_scores_loser[:, valid_gt_indices]
                
                # MSE
                l_reg_win = self.regression_loss(preds_win_tensor, gt_win_tensor)
                l_reg_lose = self.regression_loss(preds_lose_tensor, gt_lose_tensor)
                
                reg_loss_sum = (l_reg_win + l_reg_lose) / 2
                loss_dict['avg_reg_loss'] = reg_loss_sum.item()
        
        # 3. Combine
        total_loss = self.rank_weight * avg_rank_loss + self.reg_weight * reg_loss_sum
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict
