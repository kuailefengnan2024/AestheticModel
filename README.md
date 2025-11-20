
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
*   **数据处理策略：支持可变尺寸输入 (Data Strategy: Supporting Variable Image Sizes)**
    *   **原则:** 为确保模型能准确评估`构图`并具备强大的泛化能力，必须处理不同长宽比的图片，避免采用会破坏构图信息的中心裁剪或拉伸。
    *   **实现:** 采用**动态填充 (Dynamic Padding)** 与 **注意力掩码 (Attention Mask)** 的先进方案。在数据加载流程 (`core/dataset.py`) 中：
        1.  保持图片的原始长宽比进行缩放。
        2.  在组合批次（batching）时，将批内图片动态填充至统一尺寸。
        3.  生成注意力掩码，引导模型在计算时**完全忽略**填充区域，只关注真实的图像内容。

---

### **3. 核心TODO清单 (Core TODOs)**

#### **Phase 1: 数据生产 (Data Engine)**

*   **[ ] 1.1 VLM校准:** 设计并验证一个能引导公司VLM API稳定输出**多维度JSON评分**的“裁判Prompt”。**(这是成功的基石)**
*   **[ ] 1.2 数据生成:** 编写脚本，自动化地生成海量的**成对偏好数据** `(Prompt, Winner, Loser)`，并附带VLM给出的完整多维度评分。

#### **Phase 2: 模型构建 (Modeling)**

*   **[x] 2.1 架构定义:** 定义一个**多头输出 (Multi-Head)** 的模型。
    *   **共享编码器 (Body):** 加载预训练的OpenCLIP (ViT-L-14) 和 XLM-RoBERTa (多语言支持)。
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
├── api/                   # [通用工具层] 存放所有与外部AI服务(LLM, Vision, Image Gen)交互的客户端代码
│   ├── vision/            # -> 视觉理解 API
│   ├── llm/               # -> 语言模型 API
│   ├── image/             # -> 图片生成 API
│   └── ...                # -> 其他辅助 API 模块
│
├── config/                # [配置中心]
│   ├── settings.py        # -> 基础设施配置 (API Keys, 路径, 环境变量)
│   └── training_config.yaml # -> 实验配置 (模型架构参数, 训练超参数, Loss权重)
│
├── core/                  # [核心模型层] (原 aesthetic_model) 包含模型的核心算法实现
│   ├── __init__.py
│   ├── architecture.py    # -> 定义神经网络结构 (OpenCLIP + XLM-R + MultiHead MLP)
│   ├── dataset.py         # -> PyTorch Dataset, 处理成对数据加载与动态Padding
│   └── loss.py            # -> 组合排序损失函数 (CombinedRankingLoss)
│
├── data_pipeline/         # [数据流水线] (原 data_engine) 负责数据生产与预处理
│   ├── generate_data.py   # -> 自动化生成成对偏好数据 (Prompt -> Gen -> VLM Judge)
│   ├── generate_prompts.py# -> 批量生成高质量 Prompt
│   └── prompts/           # -> 存放引导 AI 的 Prompt 模板 (裁判提示词, 生成提示词)
│
├── scripts/               # [执行脚本层] 模型的训练、评估与推理入口
│   ├── train.py           # -> 启动模型训练
│   └── evaluate.py        # -> 模型效果评估
│
├── data/                  # [数据存储] 存放数据集 (通常被 .gitignore 忽略)
├── outputs/               # [产出物] 存放模型权重、日志、生成的图片样本
├── README.md              # -> 本文档
└── requirements.txt       # -> Python 依赖列表
```

### **5. API服务结构说明 (API Service Structure)**

*(本目录主要包含封装好的第三方服务客户端，供 `data_pipeline` 调用)*

```
api/
├── vision/                    # -> 存放与核心视觉理解相关的API路由和逻辑
├── llm/                       # -> 存放与大型语言模型交互相关的API路由和逻辑
├── image/                     # -> 可能用于处理图片生成、获取等任务的API
├── edit/                      # -> 可能用于处理图片编辑、修改等任务的API
├── audit/                     # -> 存放用于内容审核、安全过滤相关的API和逻辑
├── base.py                    # -> 定义API服务共享的基类、数据模型(Pydantic)或通用工具函数
└── factory.py                 # -> 应用工厂，负责创建和配置Web应用实例
```
