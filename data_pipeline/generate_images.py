"""
批量生图脚本 (Batch Image Generation)

角色: 生产者 (Producer) - RLAIF 流水线第一阶段
功能: 基于 Prompt 批量生成成对图片 (Pairwise Images)，支持多尺寸随机采样与断点续传。
特性: 
  - 混合模型策略: Seedream 4.5 (50%), Seedream 3.0 (30%), Gemini 3 Pro (20%)
  - 全尺寸覆盖: Square, Portrait, Landscape, Ultra Wide
输入: data/image_prompts_expanded.jsonl (Prompts)
输出: 
  1. outputs/raw/pairs/*.png (原始图片文件)
  2. data/intermediate/generated_pairs.jsonl (图片元数据记录)
逻辑: 读取 Prompt -> 随机选择模型 -> 随机选择宽高比 -> 调用生图 API 生成 A/B 两图 -> 保存文件与元数据。
"""
import asyncio
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import List, Tuple, Dict

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

# ------------------------------------------------------------------------------
# 尺寸策略 (Aspect Ratio Strategy)
# ------------------------------------------------------------------------------
# 覆盖从 1:1 到 21:9 的主流画幅
# Key: Gemini 需要的 aspectRatio 字符串
# Value: Seedream 需要的 (Width, Height) 像素值 (基准边长 1024)
ASPECT_RATIO_MAP = {
    # Square
    "1:1":   (2048, 2048),
    
    # Portrait
    "9:16":  (1536, 2688),
    "3:4":   (1792, 2304),
    
    # Landscape
    "16:9":  (2688, 1536),
    "4:3":   (2304, 1792)
}

# Gemini 对非常规尺寸支持不佳，容易回退到 1:1，因此仅限制在核心尺寸
GEMINI_SAFE_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4"]

# ------------------------------------------------------------------------------
# 模型路由策略 (Model Routing Strategy)
# ------------------------------------------------------------------------------
# 格式: (Provider Name, Probability Weight)
MODEL_ROUTING_WEIGHTS = [
    ("seedream_4_5",        0.5), # 50%
    ("seedream",            0.3), # 30% (Seedream 3.0)
    ("gemini_3_pro_image",  0.2)  # 20%
]

# 并发限制 (Provider-Specific Concurrency Limits)
# 针对不同模型的性能特点设置独立的并发上限
PROVIDER_CONCURRENCY = {
    "seedream_4_5": 32,      # 高吞吐 (IPM 500)，可以设置得比较高
    "seedream": 16,          # Seedream 3.0
    "gemini_3_pro_image": 4  # 限制较严 (QPM 100, TPM限制)，且Token消耗大，需保守
}
DEFAULT_CONCURRENCY = 8      # 未知模型的默认并发数 

# ==============================================================================
# 核心逻辑
# ==============================================================================

async def generate_single_image(
    client, 
    provider_name: str,
    prompt: str, 
    ratio_key: str, 
    width: int, 
    height: int, 
    filename: Path
) -> bool:
    """
    调用 API 生成单张图片并保存。
    自动适配不同 Provider 的参数格式。
    """
    try:
        if filename.exists():
            return True
            
        # --- 参数适配 (Parameter Adaptation) ---
        api_kwargs = {}
        
        if "gemini" in provider_name:
            # Gemini 使用 aspectRatio
            api_kwargs["aspectRatio"] = ratio_key
            # Gemini 可能还需要 imageSize (可选)
            api_kwargs["imageSize"] = "1K" 
        else:
            # Seedream 使用 size="WxH"
            api_kwargs["size"] = f"{width}x{height}"

        # 调用 API
        image_bytes, error = await client.call_api(
            prompt=prompt, 
            **api_kwargs
        )
        
        if error or not image_bytes:
            logger.error(f"生图失败 [{provider_name}]: {error} | Prompt: {prompt[:30]}...")
            return False
            
        # 保存图片
        with open(filename, "wb") as f:
            f.write(image_bytes)
            
        return True
    except Exception as e:
        logger.error(f"生图异常 [{provider_name}]: {e}")
        return False

async def process_prompt(
    semaphores: Dict[str, asyncio.Semaphore], 
    clients: Dict,
    prompt_data: dict, 
    pbar
):
    """
    处理单个 Prompt 的完整流程：生成 A 和 B。
    """
    prompt = prompt_data.get("prompt")
    prompt_id = prompt_data.get("id", str(uuid.uuid4())[:8])
    
    # 1. 随机选择本组的模型
    provider_name = random.choices(
        [w[0] for w in MODEL_ROUTING_WEIGHTS], 
        weights=[w[1] for w in MODEL_ROUTING_WEIGHTS]
    )[0]
    
    # 获取对应的信号量，如果没有则使用默认的（这里假设有个 default key 或者临时创建一个）
    # 注意：为了简单，我们在 main 里会确保所有 keys 都有 semaphore
    # 如果 provider_name 不在配置里，回退到 seedream 的配置或默认值
    semaphore = semaphores.get(provider_name)
    if not semaphore:
        # Fallback (理论上不应该发生，除非配置漏了)
        semaphore = semaphores.get("default", asyncio.Semaphore(1))

    async with semaphore:
        client = clients.get(provider_name)
        if not client:
            logger.error(f"找不到模型客户端: {provider_name}")
            pbar.update(1)
            return

        # 2. 随机选择本组的尺寸 (A和B保持一致)
        # 差异化策略: Seedream 全开，Gemini 保守
        if "gemini" in provider_name:
            valid_ratios = GEMINI_SAFE_RATIOS
        else:
            valid_ratios = list(ASPECT_RATIO_MAP.keys())
            
        ratio_key = random.choice(valid_ratios)
        
        width, height = ASPECT_RATIO_MAP[ratio_key]
        
        # --- 针对 Gemini 的尺寸减半优化 (Save Tokens) ---
        # Seedream 使用 2K 分辨率，Gemini 使用 1K 分辨率
        if "gemini" in provider_name:
            width = width // 2
            height = height // 2
        
        # 3. 定义文件名
        # 命名包含模型名和尺寸，方便后续分析
        # 格式: {prompt_id}_{provider}_{ratio}_{variant}.png
        # 注意文件名中不要包含冒号等非法字符，ratio_key 如 "16:9" 需替换
        safe_ratio = ratio_key.replace(":", "-")
        filename_a = RAW_IMAGES_DIR / f"{prompt_id}_{provider_name}_{safe_ratio}_A.png"
        filename_b = RAW_IMAGES_DIR / f"{prompt_id}_{provider_name}_{safe_ratio}_B.png"
        
        # 4. 生成 Image A
        success_a = await generate_single_image(
            client, provider_name, prompt, ratio_key, width, height, filename_a
        )
        if not success_a:
            pbar.update(1)
            return

        # 5. 生成 Image B (同源对抗，使用同一个 client)
        success_b = await generate_single_image(
            client, provider_name, prompt, ratio_key, width, height, filename_b
        )
        if not success_b:
            pbar.update(1)
            return
            
        # 6. 记录成功结果
        pair_record = GeneratedPair(
            prompt_id=prompt_id,
            prompt=prompt,
            width=width,
            height=height,
            image_a_path=str(filename_a),
            image_b_path=str(filename_b),
            model_name=provider_name,
            timestamp=time.time()
        )
        
        with open(OUTPUT_METADATA_FILE, "a", encoding="utf-8") as f:
            f.write(pair_record.model_dump_json() + "\n")
            
        pbar.update(1)

async def main():
    print(f"开始批量生图任务 (混合模型策略)...")
    print(f"输入: {INPUT_PROMPTS_FILE}")
    print(f"输出目录: {RAW_IMAGES_DIR}")
    
    # 1. 初始化模型池 (Client Pool)
    # 预加载所有需要的 Provider 客户端
    print("正在初始化生图模型池...")
    clients = {}
    
    try:
        # 遍历所有配置的权重模型
        required_providers = set(w[0] for w in MODEL_ROUTING_WEIGHTS)
        
        for p_name in required_providers:
            print(f"  - 初始化: {p_name}")
            # 使用 Factory 的新参数功能
            client = ApiClientFactory.create_image_client(provider_name=p_name)
            if client:
                clients[p_name] = client
            else:
                print(f"    [ERROR] 初始化失败: {p_name}")
                
        if not clients:
            logger.error("没有可用的生图客户端，任务终止。")
            return
            
    except Exception as e:
        logger.error(f"初始化模型池异常: {e}")
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

    # 3. 检查断点
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
    
    # 过滤任务
    tasks_to_run = []
    for p in all_prompts:
        p_id = p.get("id")
        if not p_id:
            import hashlib
            p_id = hashlib.md5(p["prompt"].encode("utf-8")).hexdigest()[:8]
            p["id"] = p_id
            
        if p_id not in completed_ids:
            tasks_to_run.append(p)
            
    print(f"剩余 {len(tasks_to_run)} 条任务待处理。")
    if not tasks_to_run:
        print("所有任务已完成！")
        return

    # 4. 执行并发任务
    # 初始化不同 Provider 的信号量
    semaphores = {}
    for p_name in clients.keys():
        limit = PROVIDER_CONCURRENCY.get(p_name, DEFAULT_CONCURRENCY)
        semaphores[p_name] = asyncio.Semaphore(limit)
        print(f"  - [{p_name}] 并发限制: {limit}")
    
    # 添加默认 fallback
    semaphores["default"] = asyncio.Semaphore(DEFAULT_CONCURRENCY)

    pbar = tqdm(total=len(tasks_to_run), desc="Generating Pairs")
    
    tasks = [
        process_prompt(semaphores, clients, p, pbar)
        for p in tasks_to_run
    ]
    
    await asyncio.gather(*tasks)
    
    # 关闭所有客户端
    for client in clients.values():
        if hasattr(client, "close"):
            await client.close()

if __name__ == "__main__":
    asyncio.run(main())
