# -*- coding: utf-8 -*-
"""
[对应于 README 3.1 训练流程]

启动模型训练的主脚本。

工作流程:
1.  **加载配置:**
    - 从 `configs/training_config.yaml` 文件中加载所有训练参数，
      包括学习率、批大小、训练轮次、模型超参、数据路径等。

2.  **初始化组件:**
    - 根据配置，初始化模型 (`AestheticModel`)、损失函数 (`CombinedRankingLoss`)。
    - 初始化数据预处理器和分词器 (e.g., from CLIP, BERT)。
    - 创建训练和验证用的 `PreferenceDataset` 和 `DataLoader`。
    - 初始化优化器 (e.g., AdamW) 和学习率调度器。

3.  **初始化日志:**
    - [对应于 README 3.3] 初始化 `wandb` 或 `TensorBoard`，用于监控训练过程中的
      各项指标（总损失、各维度子损失、验证集准确率等）。

4.  **训练循环:**
    - 对每个 epoch:
        - 对每个 batch 的数据:
            - 将数据送入模型，分别得到 Winner 和 Loser 的多维度评分。
            - 使用评分计算组合排序损失。
            - 执行反向传播和优化器步骤。
            - 记录当前 batch 的损失。
        - (可选) 在每个 epoch 结束后，在验证集上进行评估，计算准确率。
        - 保存模型 checkpoint。

执行命令示例:
`python scripts/train.py --config configs/training_config.yaml`
"""
import argparse
import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW

# 从 aesthetic_model 包中导入核心组件
from core.architecture import AestheticModel
from core.dataset import PreferenceDataset
from core.loss import CombinedRankingLoss

def main(config_path):
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # TODO: 2. 初始化所有组件 (模型, 数据集, Dataloader, 损失函数, 优化器等)
    #    - model = AestheticModel(config['model'])
    #    - loss_fn = CombinedRankingLoss(config['loss']['head_weights'])
    #    - train_dataset = PreferenceDataset(...)
    #    - train_loader = DataLoader(...)
    #    - optimizer = AdamW(...)

    # TODO: 3. 初始化日志工具 (wandb)

    print("开始训练...")
    # 4. 训练循环
    for epoch in range(config['training']['num_epochs']):
        print(f"--- Epoch {epoch+1}/{config['training']['num_epochs']} ---")
        
        # for batch in train_loader:
            # TODO: a. 将数据移动到GPU
            # TODO: b. 模型前向传播，得到 winner_scores 和 loser_scores
            # TODO: c. 计算损失: total_loss, loss_dict = loss_fn(winner_scores, loser_scores)
            # TODO: d. 反向传播和优化
            # TODO: e. 记录日志 (wandb.log(loss_dict))
        
        # TODO: (可选) 运行验证集评估
        # TODO: 保存模型 checkpoint

    print("训练完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练审美评估模型")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="指向训练配置文件的路径")
    args = parser.parse_args()
    main(args.config)
