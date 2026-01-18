# 核心技术与算法逻辑笔记

## 1. 本项目本质
- **定位**: Aesthetic Reward Model (审美奖励模型) / 裁判 (Judge)。
- **架构**: CLIP (Vision/Text Backbone) + Multi-Head MLP (多任务预测)。
- **算法**: **Pairwise Ranking (成对排序)**。
  - 损失函数: Bradley-Terry Loss (BCEWithLogitsLoss) + Auxiliary Regression (MSE)。
  - 核心逻辑: 学习区分 $P(\text{Winner} > \text{Loser})$，模拟 VLM/人类的审美偏好。

## 2. 算法对比 (生态位)

| 算法 | 角色 | 逻辑 (通俗) | 核心输入 | 核心输出 | 对 RM 的依赖 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Aesthetic Model** (本项目) | **裁判 (Judge)** | **鉴赏家**: 学会给好图打高分，坏图打低分。 | 图片 | 分数 (Score) | - |
| **PPO** (Proximal Policy Optimization) | **运动员 (Actor)** | **被私教盯着练**: 每画一张，裁判实时打分，根据分数调整参数。 | Prompt | 图片 | **强依赖** (必须在线参与训练) |
| **DPO** (Direct Preference Optimization) | **运动员 (Actor)** | **看教科书自学**: 直接学习数据集里的 (Win, Lose) 对，不需要裁判实时在场。 | Prompt | 图片 | **弱依赖** (训练时不需要，但**数据构建**需要 RM 自动标注) |
| **GRPO** (Group Relative Policy Optimization) | **运动员 (Actor)** | **小组赛选拔**: PPO 的变体。一次生成一组 (Group) 结果，组内比较优劣，不需要额外的 Value Network (Critic)。 | Prompt | 图片 | **强依赖** (需要 RM 对组内结果进行打分排序) |

## 3. 生产环境现状 (2025/2026)

- **微调主流 (Training)**: **DPO** (及变体 IPO/KTO)。
  - 优势: 稳定、省显存、流程短。
  - 适用: 绝大多数企业 (Llama, Mistral, SDXL, Flux)。
- **顶尖与科研 (SOTA)**: **PPO / GRPO**。
  - 适用: DeepSeek (R1/V3), OpenAI, Anthropic。冲击推理/数学/代码等复杂逻辑任务的上限。
  - 特点: GRPO 去掉了 Critic 网络，显存更友好，是 DeepSeek R1 的核心算法。
- **基础设施**: **Reward Model (本项目)**。
  - 必不可少: 用于 **1) 构建 DPO 数据集** (自动标注); **2) 推理时 Best-of-N 筛选** (提升用户体感质量)。

## 4. 竞品/模型推测

- **Seedream 4.5 (字节/即梦)**: **DiT + DPO**。
  - 依据: 字节偏向 DiT 架构，且 DPO 是目前生图微调的标准，利用海量用户点击数据构建偏好。
- **NanoBanana (Google)**: **Diffusion + RLHF (RAIFT)**。
  - 依据: Google 倾向于用自家强大的 Gemini VLM 给生成模型打分，走经典的强化学习路线。

## 5. 进阶学习路线 (DPO 实践)

- **目标**: 实现 **Aesthetic Fine-Tuner** (让 SDXL/Flux 懂审美)。
- **步骤**:
  1.  **数据合成**: 使用 **本项目 (RM)** 对 SDXL 生成的图打分，构建 `(Prompt, Win, Lose)` 数据集。
  2.  **DPO 训练**: 使用 `diffusers` + `TRL` 库，加载 LoRA，使用 `DPOLoss` 进行微调。
  3.  **验证**: 对比微调前后的审美差异。

