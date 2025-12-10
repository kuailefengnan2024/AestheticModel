"""
批量打分与标注脚本 (Batch Judging & Labeling)

角色: 标注者 (Labeler) - RLAIF 流水线第二阶段
功能: 利用 VLM (视觉大模型) 对生成的图片对进行多维度审美评估，构建偏好数据集。
输入: data/intermediate/generated_pairs.jsonl (生图记录)
输出: data/preferences_train.jsonl (最终训练数据集)
逻辑: 读取图片对 -> 拼接/构造 VLM 请求 -> 解析 JSON 评分 -> 计算 Winner/Loser -> 生成结构化训练数据。
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
You are an expert aesthetic critic and photographer.
I will show you two images generated from the same prompt: "{prompt}".

Image A is the first image.
Image B is the second image.

Please evaluate both images on the following dimensions:
1. Total Quality (Overall aesthetic appeal)
2. Composition (Balance, framing, visual flow)
3. Color (Harmony, palette choice, mood)
4. Lighting (Contrast, shadows, atmosphere)

For each dimension, give a score from 0.0 to 10.0 (allows one decimal place).
Be critical and strict. A score of 8.0+ should only be given to professional-level works.

You MUST return the result in the following JSON format ONLY, no other text:

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
    }}
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
        # 注意：这里我们不需要在代码里做 resize，直接传原图，Doubao/GPT 会自己处理
        # 只要确保路径存在
        img_a = Path(pair_record.image_a_path)
        img_b = Path(pair_record.image_b_path)
        
        if not img_a.exists() or not img_b.exists():
            logger.error(f"找不到图片文件: {img_a} 或 {img_b}")
            pbar.update(1)
            return

        try:
            # 这里的 call_api 需要支持多图输入
            # 我们假设 Vision Client 的 call_api 签名支持 images=[path1, path2] 或者多次调用 add_image
            # 由于目前的 DoubaoSeedVisionProvider 可能只实现了单图，我们需要检查一下
            # 即使只支持单图，我们也可以把两张图拼起来 (Horizontal Stack) 发给它，或者发两次请求
            # **最佳实践**: VLM API 通常支持多图列表。
            # 如果我们用的是 api/vision/doubao_seed_1_6_vision.py，它目前只接收 `image_path` (单数)。
            # 为了不改动底层 API 太大，我们这里采用 **“分别打分”** 或者 **“拼图打分”** 策略。
            # 考虑到 Pairwise 比较需要同时看到两张图才能对比出细微差别，
            # 如果底层 API 不支持多图列表，建议把两张图在本地拼成一张宽图发过去。
            
            # --- 拼图逻辑 ---
            # 为了简单，我们暂时用 python 脚本把两张图横向拼在一起，生成一个临时文件
            # 这样 VLM 只需要处理一张图，而且能直观对比
            # 缺点：VLM 需要知道左边是A，右边是B
            
            # TODO: 更好的做法是修改 Vision Provider 支持多图列表。
            # 这里先用临时拼图方案，因为这是最快能跑通且不破坏现有架构的方法。
            
            from PIL import Image
            
            im_a = Image.open(img_a)
            im_b = Image.open(img_b)
            
            # 统一高度拼接
            # 假设两张图尺寸接近（我们生图时是随机但也受控的）
            # 直接横向拼接
            dst = Image.new('RGB', (im_a.width + im_b.width + 20, max(im_a.height, im_b.height)), (0,0,0))
            dst.paste(im_a, (0, 0))
            dst.paste(im_b, (im_a.width + 20, 0)) # 中间留20px黑缝
            
            temp_combined_path = settings.BASE_DIR / "data" / "intermediate" / "temp" / f"combined_{pair_record.prompt_id}.jpg"
            temp_combined_path.parent.mkdir(parents=True, exist_ok=True)
            dst.save(temp_combined_path, quality=85)
            
            # 修改 Prompt 告诉模型左边是A，右边是B
            combined_prompt = prompt_text.replace(
                "Image A is the first image.", "Image A is on the LEFT side."
            ).replace(
                "Image B is the second image.", "Image B is on the RIGHT side."
            )
            
            response_content, error = await client.call_api(
                prompt_text=combined_prompt,
                image_path=str(temp_combined_path)
            )
            
            # 删除临时文件
            try:
                os.remove(temp_combined_path)
            except:
                pass
                
            if error:
                logger.error(f"VLM 调用失败: {error}")
                raise Exception(error)
                
            # 解析 JSON
            result_json = parse_vlm_response(response_content)
            if not result_json or "image_a" not in result_json or "image_b" not in result_json:
                logger.error(f"VLM 返回格式错误: {response_content[:100]}...")
                # 记录到失败日志
                with open(FAILED_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"pair": pair_record.dict(), "response": response_content}) + "\n")
                pbar.update(1)
                return
                
            # 构造最终 Training Data
            scores_a = result_json["image_a"]
            scores_b = result_json["image_b"]
            
            # 谁是总分 Winner?
            if scores_a.get("total", 0) >= scores_b.get("total", 0):
                winner_path = pair_record.image_a_path
                loser_path = pair_record.image_b_path
                is_a_winner = True
            else:
                winner_path = pair_record.image_b_path
                loser_path = pair_record.image_a_path
                is_a_winner = False
                
            # 组装 Scores 对象
            # 注意：schemas.Scores 里的每个字段都是 ScoreDetail(winner=..., loser=...)
            # 所以我们要根据谁赢了总分，来填对应的坑。
            # 如果 A 赢了 Total，那么 scores.composition.winner 就填 A 的构图分
            # 哪怕 A 的构图分其实比 B 低（这就是 Feature Decoupling 还没做完的地方，
            # 但 schemas.ScoreDetail 的 winner/loser 是指“整张图的赢家/输家”，而不是“该维度的赢家”）
            # 等等！让我们回看 README 的定义：
            # "scores": {"composition": {"winner": 9.0, "loser": 5.0}}
            # 这里的 winner 指的是 image_winner_path 对应的那张图。
            # 所以逻辑是：
            # 1. 先确定 Total Winner 是哪张图（比如 A）。
            # 2. 那么 scores 里的所有 "winner" 字段都填 A 的分，所有 "loser" 字段都填 B 的分。
            # 3. 如果 B 的色彩分更高，那么 color.loser > color.winner，这是允许的，也是模型要学的。
            
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
                scores=Scores(**final_scores)
            )
            
            # 写入文件
            with open(OUTPUT_TRAIN_FILE, "a", encoding="utf-8") as f:
                f.write(training_record.json() + "\n")
                
            pbar.update(1)

        except Exception as e:
            logger.error(f"处理 Pair 异常: {e}")
            pbar.update(1)

async def main():
    print(f"开始批量打分任务...")
    print(f"输入: {INPUT_PAIRS_FILE}")
    print(f"输出: {OUTPUT_TRAIN_FILE}")
    
    # 1. 初始化 Vision Client
    client = ApiClientFactory.create_vision_client()
    if not client:
        logger.error("无法初始化 Vision 客户端。")
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
                    
    print(f"总共有 {len(all_pairs)} 对图片待打分。")
    
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
                    
    tasks_to_run = [p for p in all_pairs if p.prompt_id not in processed_ids]
    print(f"剩余 {len(tasks_to_run)} 对需要处理。")
    
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

