# -*- coding: utf-8 -*-
"""
[对应于 README 3.2 评估指标]

在独立的测试集上评估模型性能的脚本。

核心指标:
**准确率 (Accuracy):** 在模型从未见过的人类标注数据上，
模型判断 `Winner > Loser` 的正确率有多高。
这是衡量模型是否学习到与人类一致的偏好判断能力的关键。

工作流程:
1.  **加载模型:**
    - 加载一个已经训练好的模型 checkpoint。
    - 将模型设置为评估模式 (`model.eval()`)。

2.  **加载测试数据:**
    - 创建一个用于测试的 `PreferenceDataset` 和 `DataLoader`。
    - 测试集应该是独立的、最好由人类专家标注的数据，以提供最可靠的评估。

3.  **执行评估:**
    - 禁用梯度计算 (`torch.no_grad()`)。
    - 遍历测试集中的每一个数据对。
    - 将数据送入模型，得到 Winner 和 Loser 的总分 (`total_score`)。
    - 如果 `model_score(Winner) > model_score(Loser)`，则记为一次正确预测。

4.  **计算并报告结果:**
    - 计算总的准确率: `(正确预测的数量 / 总样本数) * 100%`。
    - (可选) 可以进一步分析在不同类别或场景下的准确率。

执行命令示例:
`python scripts/evaluate.py --checkpoint /path/to/model.pth --test_data data/human_annotated_test.jsonl`
"""
import argparse
import torch
from torch.utils.data import DataLoader

from aesthetic_model.architecture import AestheticModel
from aesthetic_model.dataset import PreferenceDataset

def main(args):
    # TODO: 1. 加载模型
    #    - device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #    - model = AestheticModel(...) # 需要一个方法来从config构建模型
    #    - model.load_state_dict(torch.load(args.checkpoint))
    #    - model.to(device)
    #    - model.eval()

    # TODO: 2. 加载测试数据
    #    - test_dataset = PreferenceDataset(data_path=args.test_data, ...)
    #    - test_loader = DataLoader(test_dataset, ...)

    correct_predictions = 0
    total_samples = 0

    print("开始评估...")
    with torch.no_grad():
        # for batch in test_loader:
            # TODO: a. 将数据移动到GPU
            # TODO: b. 模型前向传播，得到 winner_scores 和 loser_scores
            #    - winner_total_score = winner_scores['total_score']
            #    - loser_total_score = loser_scores['total_score']
            
            # TODO: c. 比较分数，计算正确预测的数量
            #    - correct_predictions += (winner_total_score > loser_total_score).sum().item()
            #    - total_samples += len(batch['winner_image'])
            pass

    # 4. 计算并报告结果
    # accuracy = (correct_predictions / total_samples) * 100
    # print(f"评估完成。")
    # print(f"测试集样本总数: {total_samples}")
    # print(f"准确率: {accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估已训练的模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="已训练模型的checkpoint文件路径")
    parser.add_argument("--test_data", type=str, required=True, help="用于评估的测试数据文件路径")
    # TODO: 可能需要添加 --config 参数来帮助构建模型
    args = parser.parse_args()
    main(args)
