"""
批量打分与标注脚本 (Batch Judging & Labeling)

角色: 标注者 (Labeler) - RLAIF 流水线第二阶段
功能: 利用 VLM (视觉大模型) 对生成的图片对进行多维度审美评估，构建偏好数据集。
输入: data/intermediate/generated_pairs.jsonl (生图记录)
输出: data/preferences_train.jsonl (最终训练数据集)
逻辑: 读取图片对 -> 构造 VLM 请求 (多图列表) -> 解析 JSON 评分 -> 计算 Winner/Loser -> 生成结构化训练数据。
"""
import asyncio
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# 引入项目模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.factory import ApiClientFactory
from config import settings
from data_pipeline.schemas import GeneratedPair, TrainingData, Scores, ScoreDetail
from utils.logger import logger

load_dotenv()

# ==============================================================================
# 配置区域
# ==============================================================================

# 输入: 之前生图脚本生成的中间文件
INPUT_PAIRS_FILE = settings.BASE_DIR / "data" / "intermediate" / "generated_pairs.jsonl"

# 输出: 最终的训练数据
OUTPUT_TRAIN_FILE = settings.BASE_DIR / "data" / "preferences_train.jsonl"
OUTPUT_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

# 失败记录 (方便 Debug)
FAILED_LOG_FILE = settings.BASE_DIR / "data" / "intermediate" / "failed_judgments.jsonl"

CONCURRENCY_LIMIT = 10 # 打分可以比生图稍微快一点

# VLM 裁判 Prompt 模板
# 这里的关键是让 VLM 返回纯 JSON，并且对 A/B 进行明确区分
JUDGE_PROMPT_TEMPLATE = """
你是一位拥有15年经验的顶尖创意总监，你的审美判断既精准又深刻，并且非常熟悉当前市场的商业标准。你的任务是对我提供的两张图片，进行多维度的、专业的审美评估。

这些图片是根据同一个提示词生成的："{prompt}"。

The first image in the input list is Image A.
The second image in the input list is Image B.

**评估维度:**

请根据以下维度，对每一张图片进行独立打分（1-10分，1分最低，10分最高）：

1.  **total (总分):** 对图片整体质量和吸引力的综合评价。
2.  **composition (构图):** 元素的布局、平衡感、视觉引导线是否和谐且有冲击力。
3.  **color (色彩):** 色彩搭配是否协调、有美感，是否能有效传达情感或主题。
4.  **lighting (光影):** 光线的运用是否为画面增添了层次、氛围和质感。

**输出格式要求:**

请严格按照以下JSON格式返回你的评估结果，不要包含任何额外的解释或说明文字。

{{
    "image_a": {{
        "total": <score>,
        "composition": <score>,
        "color": <score>,
        "lighting": <score>
    }},
    "image_b": {{
        "total": <score>,
        "composition": <score>,
        "color": <score>,
        "lighting": <score>
    }},
    "reasoning": "请在这里用一句话简要说明你判定优劣的核心理由。"
}}
"""

# ==============================================================================
# 核心逻辑
# ==============================================================================

def parse_vlm_response(content: str) -> dict | None:
    """
    尝试从 VLM 的回复中提取 JSON。
    VLM 有时候会废话，比如 "Here is the JSON: ```json ... ```"，需要清洗。
    """
    try:
        # 1. 尝试直接解析
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # 2. 尝试提取 ```json ... ``` 代码块
    match = re.search(r"```json(.*?)```", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    # 3. 尝试提取 { ... }
    match = re.search(r"(\{.*\})", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    return None

async def judge_single_pair(semaphore, client, pair_record: GeneratedPair, pbar):
    """
    处理单条数据：调用 VLM -> 解析 -> 写入结果。
    """
    async with semaphore:
        prompt_text = JUDGE_PROMPT_TEMPLATE.format(prompt=pair_record.prompt)
        
        # 准备图片路径列表 [Image A, Image B]
        img_a = Path(pair_record.image_a_path)
        img_b = Path(pair_record.image_b_path)
        
        if not img_a.exists() or not img_b.exists():
            logger.error(f"找不到图片文件: {img_a} 或 {img_b}")
            pbar.update(1)
            return

        try:
            # 调用 Vision API (原生多图列表支持)
            response_content, error = await client.call_api(
                prompt_text=prompt_text,
                image_paths=[str(img_a), str(img_b)]
            )
                
            if error:
                logger.error(f"VLM 调用失败: {error}")
                # 记录到失败日志
                with open(FAILED_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"pair_id": pair_record.prompt_id, "error": str(error)}) + "\n")
                pbar.update(1)
                return
                
            # 解析 JSON
            result_json = parse_vlm_response(response_content)
            if not result_json or "image_a" not in result_json or "image_b" not in result_json:
                logger.error(f"VLM 返回格式错误: {response_content[:100]}...")
                # 记录到失败日志
                with open(FAILED_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"pair_id": pair_record.prompt_id, "raw_response": response_content}) + "\n")
                pbar.update(1)
                return
                
            # 构造最终 Training Data
            scores_a = result_json["image_a"]
            scores_b = result_json["image_b"]
            
            # 谁是总分 Winner?
            # 注意：如果分数相等，我们倾向于认为 A 是 Winner (或者随机，这里为了确定性选 A)
            if float(scores_a.get("total", 0)) >= float(scores_b.get("total", 0)):
                winner_path = pair_record.image_a_path
                loser_path = pair_record.image_b_path
                is_a_winner = True
            else:
                winner_path = pair_record.image_b_path
                loser_path = pair_record.image_a_path
                is_a_winner = False
                
            # 组装 Scores 对象
            # 这里的 winner/loser 指的是 "image_winner_path" 和 "image_loser_path" 对应的实体
            # 例如：如果 B 是 Winner (Total分高)，那么 TrainingData.image_winner_path = B
            # 此时 scores.color.winner 应该填 B 的色彩分，scores.color.loser 填 A 的色彩分
            
            final_scores = {}
            dimensions = ["total", "composition", "color", "lighting"]
            
            for dim in dimensions:
                val_a = float(scores_a.get(dim, 0))
                val_b = float(scores_b.get(dim, 0))
                
                final_scores[dim] = ScoreDetail(
                    winner=val_a if is_a_winner else val_b,
                    loser=val_b if is_a_winner else val_a
                )
                
            training_record = TrainingData(
                prompt=pair_record.prompt,
                prompt_id=pair_record.prompt_id,
                image_winner_path=winner_path,
                image_loser_path=loser_path,
                scores=Scores(**final_scores),
                reasoning=result_json.get("reasoning", "")
            )
            
            # 写入文件
            with open(OUTPUT_TRAIN_FILE, "a", encoding="utf-8") as f:
                f.write(training_record.json() + "\n")
                
            pbar.update(1)

        except Exception as e:
            logger.error(f"处理 Pair 异常 [{pair_record.prompt_id}]: {e}")
            pbar.update(1)

async def main():
    print(f"开始批量打分任务 (VLM Judge)...")
    print(f"输入: {INPUT_PAIRS_FILE}")
    print(f"输出: {OUTPUT_TRAIN_FILE}")
    
    # 1. 初始化 Vision Client
    # 注意：确保 .env 中配置了 VISION_API_PROVIDER=doubao_seed_1_6_vision
    client = ApiClientFactory.create_vision_client()
    if not client:
        logger.error("无法初始化 Vision 客户端，请检查配置。")
        return

    # 2. 读取 Pairs
    if not INPUT_PAIRS_FILE.exists():
        logger.error("未找到输入文件。请先运行 generate_images.py")
        return
        
    all_pairs = []
    with open(INPUT_PAIRS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_pairs.append(GeneratedPair.parse_raw(line))
                except:
                    continue
                    
    print(f"总共有 {len(all_pairs)} 对图片记录。")
    
    # 3. 检查断点 (通过 prompt_id)
    processed_ids = set()
    if OUTPUT_TRAIN_FILE.exists():
        with open(OUTPUT_TRAIN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "prompt_id" in data:
                        processed_ids.add(data["prompt_id"])
                except:
                    pass
                    
    # 过滤掉已经打过分的
    tasks_to_run = [p for p in all_pairs if p.prompt_id not in processed_ids]
    
    # 还需要过滤掉本地图片文件不存在的 (比如被用户删了)
    # 这一步会稍微花点时间 IO，但为了稳健性是值得的
    # 也可以在运行时检查
    
    print(f"剩余 {len(tasks_to_run)} 对需要打分。")
    
    if not tasks_to_run:
        print("所有任务已完成！")
        return

    # 4. 执行并发
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    pbar = tqdm(total=len(tasks_to_run), desc="Judging Pairs")
    
    tasks = [
        judge_single_pair(semaphore, client, p, pbar)
        for p in tasks_to_run
    ]
    
    await asyncio.gather(*tasks)
    
    if hasattr(client, "close"):
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
