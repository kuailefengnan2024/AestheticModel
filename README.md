

### **项目：多维度审美评估模型 (AestheticModel)**

#### **1. 项目目标 (Objective)**

*   **核心目标:** 构建一个独立的AI模型，该模型能够模拟人类专家的多维度审美和商业判断力。
*   **输入:** kv图片 + 客观元素描述Prompt
*   **输出:** 结构化的、可解释的评分报告，包含总分以及多个预设维度的分项得分（如构图、色彩、风格契合度等）。
*   **商业价值:** 赋能大规模、自动化、低成本的创意筛选、评估与优化。

---

### **2. 技术战略 (Strategy)**

*   **核心范式:** **RLAIF (AI反馈强化学习)**。利用大型VLM（教师模型）的先进判断力，通过**知识蒸馏**，将其能力迁移到一个轻量、高效的专属模型（本模型）中。
*   **训练方法:** **多任务学习 (Multi-Task Learning)**。通过一个统一的**组合排序损失函数 (Combined Ranking Loss)**，并行地训练模型在所有维度上的排序和打分能力。

---

### **3. 核心TODO清单 (Core TODOs)**

#### **Phase 1: 数据生产 (Data Engine)**

*   **[ ] 1.1 VLM校准:** 设计并验证一个能引导公司VLM API稳定输出**多维度JSON评分**的“裁判Prompt”。**(这是成功的基石)**
*   **[ ] 1.2 数据生成:** 编写脚本，自动化地生成海量的**成对偏好数据** `(Prompt, Winner, Loser)`，并附带VLM给出的完整多维度评分。

#### **Phase 2: 模型构建 (Modeling)**

*   **[ ] 2.1 架构定义:** 定义一个**多头输出 (Multi-Head)** 的模型。
    *   **共享编码器 (Body):** 加载预训练的OpenCLIP（视觉）和BERT（文本）。
    *   **独立输出头 (Heads):** 为“总分”和每一个评分维度分别创建一个独立的MLP预测头。
*   **[ ] 2.2 损失函数:** 实现**组合排序损失**，将所有头的排序损失加权求和。

#### **Phase 3: 训练与评估 (Training & Evaluation)**

*   **[ ] 3.1 训练流程:** 编写训练脚本，实现端到端的多任务训练循环。
*   **[ ] 3.2 评估指标:** 核心是**准确率**——在独立的人类标注测试集上，模型判断 `Winner > Loser` 的正确率有多高。
*   **[ ] 3.3 日志监控:** 使用`wandb`或类似工具，监控**总损失**和**各维度子损失**的收敛情况。

---

### **4. 项目代码结构说明 (Project Structure)**

```
AestheticModel/
├── aesthetic_model/           # 核心模型Python包
│   ├── __init__.py            # -> 包初始化文件，方便模块导入
│   ├── architecture.py        # -> 定义模型的神经网络结构 (对应TODO 2.1)
│   ├── dataset.py             # -> 定义PyTorch Dataset，用于加载成对偏好数据
│   └── loss.py                # -> 实现组合排序损失函数 (对应TODO 2.2)
│
├── configs/                   # 存放所有配置文件
│   └── training_config.yaml   # -> 包含所有训练参数、模型超参数和文件路径等
│
├── data/                      # 存放最终生成的数据集 (被.gitignore忽略)
│
├── data_engine/               # 负责数据生产的独立模块 (对应Phase 1)
│   ├── __init__.py            # -> 使data_engine成为一个可导入的Python包
│   ├── generate_data.py       # -> 自动化生成成对偏好数据的核心脚本 (对应TODO 1.2)
│   └── prompts/               # -> 存放用于引导VLM的Prompt模板
│       └── judge_prompt.txt   # -> "裁判Prompt"，引导VLM输出结构化评分 (对应TODO 1.1)
│
├── scripts/                   # 存放模型训练与评估相关的可执行脚本 (对应Phase 3)
│   ├── train.py               # -> 启动模型训练的入口脚本 (对应TODO 3.1 & 3.3)
│   └── evaluate.py            # -> 在独立测试集上评估模型性能的脚本 (对应TODO 3.2)
│
├── .gitignore                 # -> Git忽略文件配置
├── README.md                  # -> 本文档
└── requirements.txt           # -> 项目所需的Python依赖库
```