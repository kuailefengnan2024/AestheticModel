
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
    *   **尺寸策略 (Aspect Ratios):** 为了训练模型对不同构图的适应性，生图时应随机覆盖以下常用比例：
        *   **Square:** 1:1
        *   **Portrait:** 3:4, 9:16
        *   **Landscape:** 4:3, 16:9
        *   *(模型将通过动态Padding机制统一处理这些输入)*

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
│   ├── schemas.py         # -> [新] 定义统一的数据结构 (Pydantic Models)
│   ├── generate_prompts.py# -> 批量生成高质量 Prompt
│   ├── generate_images.py # -> [Step 1] 批量生成成对图片 (支持多尺寸)
│   ├── judge_pairs.py     # -> [Step 2] 调用 VLM 对图片对进行打分和标注
│   └── prompts/           # -> 存放引导 AI 的 Prompt 模板 (裁判提示词, 生成提示词)
│
├── crawlers/                  # [新增] 爬虫专用文件夹
│   ├── __init__.py
│   ├── base_crawler.py        # -> 爬虫基类 (处理通用的请求、重试、代理等)
│   └── byteartist_crawler.py  # -> Byteartist 专用爬虫
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

### **6. 核心Q&A与设计哲学 (Core Design Philosophy)**

#### **Q1: 什么是回归任务 (Regression Task)?**
*   **定义**: 让 AI 预测一个**连续数值**的任务（例如：预测明天的气温是 25.3度）。
*   **对比**: 分类任务是预测类别（猫 vs 狗），排序任务是预测顺序（A > B）。
*   **应用**: 在本项目中，如果让模型直接输出“这张图是 8.5 分”，这就是回归。但我们选择不这样做。

#### **Q2: 为什么要用 Winner vs Loser (对比学习) 而不是直接训练打分 (回归)?**
*   **Ground Truth 不可靠**: 人类专家对“8分”的主观定义差异很大（有人手松，有人手紧），导致绝对分数充满噪声，模型难以收敛。
*   **相对关系更鲁棒**: 虽然专家打分不同，但他们通常都能一致认同“图A 比 图B 好”。利用这种相对的偏好（Ranking）进行训练，模型能学到更本质的审美特征。
*   **数据更准**: VLM（如 GPT-4o, Gemini）在判断两张图谁好谁坏时，比直接给一张图打分要准确得多。

#### **Q3: 最终模型的输入输出是什么?**
*   **输入**: 一张图片 (Image) + 对应的提示词 (Prompt)。(Prompt 用于让模型理解“图文一致性”)
*   **输出**: 一个字典，包含所有维度的**Logits (相对分数)**。
    *   `e.g., {'total_score': 2.1, 'composition': 0.5, ...}`
*   **用法**:
    *   **筛选**: 给 4 张图，模型输出 4 个分，直接选分最高的。
    *   **注意**: 输出的分数不是 0-10 的绝对值，而是表示相对优劣的数值（越大越好）。

#### **Q4: 为什么 MLP 只有 1-2 层？能承担复杂的审美评估吗?**
*   **站在巨人的肩膀上**: 本模型的**核心特征提取能力**来自于强大的预训练模型 `OpenCLIP (ViT-L-14)` 和 `XLM-RoBERTa`。
*   **CLIP 已经“懂”了**: CLIP 已经在 4 亿对图文数据上见过各种构图、色彩和风格，它的输出向量（Embeddings）已经包含了高度浓缩的审美信息。
*   **MLP 只是翻译**: MLP 不需要从头学习“什么是美”，它只需要学习**如何将 CLIP 提取好的特征映射到我们的评分标准上**。
*   **微调的魔力**: 训练时我们会**解冻** CLIP 的部分层进行微调，让它变得更适应我们的特定任务，大部分复杂的非线性变换是在 Transformer 层完成的。

#### **Q5: MLP 头的具体形状结构是怎样的?**
*   这是一个经典的 **"Bottleneck" (瓶颈)** 结构，负责将高维特征压缩为单一的评分。 
*   **结构图解**:
    ```text
    [ Input: 1024 ]  <-- 融合特征 (CLIP 512 + RoBERTa 512)
          |
    [ Hidden: 512 ]  <-- 隐藏层 (Linear + ReLU + Dropout)
          |
    [ Output: 1 ]    <-- 最终评分 (Scalar Logits)
    ```
*   **数学表达**: $Score = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$

### **7. 数据集样例 (Dataset Sample)**

```json
{
  "prompt": "一张暖色调的3D数字海报...",
  
  // 1. 明确指出谁是总分赢家，方便人类查阅
  "image_winner_path": "outputs/raw/001_seed888_winner.png",
  "image_loser_path":  "outputs/raw/001_seed222_loser.png",
  
  // 2. 具体的 Logits (分数)，用于训练
  // 注意：这里隐含了 winner 的 total 分数一定 > loser 的 total 分数
  "scores": {
    "total":       {"winner": 8.5, "loser": 6.0}, 
    
    // 但是在子维度上，Loser 依然可以逆袭 (Feature Decoupling)
    "composition": {"winner": 9.0, "loser": 5.0}, 
    "color":       {"winner": 7.0, "loser": 8.0}  // <--- 看这里，Loser 色彩更好
  }
}
```
