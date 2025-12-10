"""
批量生图脚本 (Batch Image Generation)

角色: 生产者 (Producer) - RLAIF 流水线第一阶段
功能: 基于 Prompt 批量生成成对图片 (Pairwise Images)，支持多尺寸随机采样与断点续传。
输入: data/image_prompts_expanded.jsonl (Prompts)
输出: 
  1. outputs/raw/pairs/*.png (原始图片文件)
  2. data/intermediate/generated_pairs.jsonl (图片元数据记录)
逻辑: 读取 Prompt -> 随机选择宽高比 -> 调用生图 API 生成 A/B 两图 -> 保存文件与元数据。
"""
import asyncio
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# 引入项目模块
import sys
# 确保可以导入项目根目录的模块
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.factory import ApiClientFactory
from config import settings
from data_pipeline.schemas import GeneratedPair
from utils.logger import logger

# 加载环境变量
load_dotenv()

# ==============================================================================
# 配置区域
# ==============================================================================

# 输入 Prompt 文件
INPUT_PROMPTS_FILE = settings.BASE_DIR / "data" / "image_prompts_expanded.jsonl"

# 输出目录
RAW_IMAGES_DIR = settings.OUTPUTS_DIR / "raw" / "pairs"
RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# 中间结果文件 (记录已生成的 Pair)
OUTPUT_METADATA_FILE = settings.BASE_DIR / "data" / "intermediate" / "generated_pairs.jsonl"
OUTPUT_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

# 随机宽高比策略 (Width, Height)
# 涵盖: 1:1, 3:4, 4:3, 9:16, 16:9
ASPECT_RATIOS = [
    (1024, 1024), # Square 1:1
    (896, 1152),  # Portrait 3:4 (Approx)
    (1152, 896),  # Landscape 4:3 (Approx)
    (768, 1344),  # Portrait 9:16 (Approx)
    (1344, 768)   # Landscape 16:9 (Approx)
]

# 并发限制 (防止 API 限流)
CONCURRENCY_LIMIT = 5 

# ==============================================================================
# 核心逻辑
# ==============================================================================

async def generate_single_image(client, prompt: str, width: int, height: int, filename: Path) -> bool:
    """
    调用 API 生成单张图片并保存。
    如果成功返回 True，失败返回 False。
    """
    try:
        if filename.exists():
            return True
            
        # 构造尺寸字符串 "WxH"
        size_str = f"{width}x{height}"
        
        # 调用 API (这里假设 Seedream 接口支持 size 参数)
        # 注意：不同的 Provider 对 size 参数的处理可能不同，需确保 Provider 适配器已正确实现
        image_bytes, error = await client.call_api(
            prompt=prompt, 
            size=size_str
        )
        
        if error or not image_bytes:
            logger.error(f"生图失败: {error} | Prompt: {prompt[:30]}...")
            return False
            
        # 保存图片
        with open(filename, "wb") as f:
            f.write(image_bytes)
            
        return True
    except Exception as e:
        logger.error(f"生图异常: {e}")
        return False

async def process_prompt(
    semaphore: asyncio.Semaphore, 
    client, 
    prompt_data: dict, 
    pbar
):
    """
    处理单个 Prompt 的完整流程：生成 A 和 B。
    """
    async with semaphore:
        prompt = prompt_data.get("prompt")
        # 如果原始数据里没有ID，生成一个UUID
        prompt_id = prompt_data.get("id", str(uuid.uuid4())[:8])
        
        # 1. 随机决定本组图片的尺寸 (A和B保持一致)
        width, height = random.choice(ASPECT_RATIOS)
        
        # 2. 定义文件名
        # 命名格式: {prompt_id}_{width}x{height}_{variant}.png
        filename_a = RAW_IMAGES_DIR / f"{prompt_id}_{width}x{height}_A.png"
        filename_b = RAW_IMAGES_DIR / f"{prompt_id}_{width}x{height}_B.png"
        
        # 3. 生成 Image A
        success_a = await generate_single_image(client, prompt, width, height, filename_a)
        if not success_a:
            pbar.update(1)
            return

        # 4. 生成 Image B (希望是不同的 Seed)
        # Seedream API 默认是随机 Seed，所以多次调用自然会不同
        success_b = await generate_single_image(client, prompt, width, height, filename_b)
        if not success_b:
            pbar.update(1)
            return
            
        # 5. 记录成功结果
        pair_record = GeneratedPair(
            prompt_id=prompt_id,
            prompt=prompt,
            width=width,
            height=height,
            image_a_path=str(filename_a),
            image_b_path=str(filename_b),
            model_name=client.model,
            timestamp=time.time()
        )
        
        # 原子写入 (append mode)
        with open(OUTPUT_METADATA_FILE, "a", encoding="utf-8") as f:
            f.write(pair_record.model_dump_json() + "\n")
            
        pbar.update(1)

async def main():
    print(f"开始批量生图任务...")
    print(f"输入: {INPUT_PROMPTS_FILE}")
    print(f"输出目录: {RAW_IMAGES_DIR}")
    print(f"记录文件: {OUTPUT_METADATA_FILE}")
    
    # 1. 初始化生图客户端 (使用 settings 中配置的 IMAGE_API_PROVIDER)
    # 默认可能是 'seedream' 或 'gpt_image_1'
    client = ApiClientFactory.create_image_client()
    if not client:
        logger.error("无法初始化生图客户端，请检查 settings.py 配置。")
        return

    # 2. 读取 Prompts
    if not INPUT_PROMPTS_FILE.exists():
        logger.error(f"未找到 Prompt 文件: {INPUT_PROMPTS_FILE}")
        return
        
    all_prompts = []
    with open(INPUT_PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_prompts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    print(f"总共有 {len(all_prompts)} 条 Prompts。")

    # 3. 检查断点 (已完成的)
    completed_ids = set()
    if OUTPUT_METADATA_FILE.exists():
        with open(OUTPUT_METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed_ids.add(data.get("prompt_id"))
                    except:
                        pass
    
    print(f"已完成 {len(completed_ids)} 条，剩余 {len(all_prompts) - len(completed_ids)} 条。")
    
    # 过滤待处理任务
    # 注意：这里假设 JSONL 里的每行最好有个 id 字段。如果原数据没有 id，我们可能需要依赖 prompt 内容本身来去重。
    # 为了简化，我们暂时通过 prompt 文本哈希或者假设原数据在内存里按顺序生成了 id 来匹配。
    # 这里我们做一个简单的补丁：如果原数据没有 id，我们在读取时就给它临时分配一个，但这样断点续传可能不准。
    # 更好的方式是依赖 prompt 字符串本身去重（如果 prompt 是唯一的）。
    
    tasks_to_run = []
    for p in all_prompts:
        # 尝试获取 id，如果没有则用 prompt 文本做 key
        p_id = p.get("id")
        if not p_id:
            # 临时补救：如果没有 id，我们这里不做强去重，或者我们应该在 generate_prompts.py 阶段就生成好 id。
            # 这里简单起见，如果原文件没有 id，我们生成一个基于内容的 hash id
            import hashlib
            p_id = hashlib.md5(p["prompt"].encode("utf-8")).hexdigest()[:8]
            p["id"] = p_id
            
        if p_id not in completed_ids:
            tasks_to_run.append(p)
            
    if not tasks_to_run:
        print("所有任务已完成！")
        return

    # 4. 执行并发任务
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    pbar = tqdm(total=len(tasks_to_run), desc="Generating Pairs")
    tasks = [
        process_prompt(semaphore, client, p, pbar)
        for p in tasks_to_run
    ]
    
    await asyncio.gather(*tasks)
    
    # 关闭客户端
    if hasattr(client, "close"):
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())

