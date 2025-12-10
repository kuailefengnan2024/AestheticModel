# -*- coding: utf-8 -*-
"""
[数据生产流程的第一步]

使用大型语言模型 (LLM, 例如 Gemini) 批量生成用于文生图的提示词 (Prompt)。

工作流程:
1.  **加载 "种子提示词" (Seed Prompts):**
    - 读取 `data/raw_prompts/byteartist_prompts.jsonl`。
    - 提取其中的高质量 Prompt 作为风格参考。

2.  **调用LLM API (RAG Style Generation):**
    - 随机抽取几个种子 Prompt。
    - 让 LLM 参考这些种子的风格，生成新的、多样化的 Prompt。

3.  **保存:**
    - 将生成的新 Prompt 追加到 `data/image_prompts.txt`。

执行命令示例:
`python data_pipeline/generate_prompts.py`
"""
import argparse
import os
import asyncio
import re
import sys
import json
import random
from typing import List

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.llm.gemini_3_pro import Gemini3ProProvider
from config.settings import LLM_API_CONFIGS

# ================= 配置区 =================
SEED_FILE_PATH = "data/raw_prompts/byteartist_prompts.jsonl"
OUTPUT_FILE_PATH = "data/image_prompts_expanded.jsonl"
TARGET_COUNT = 2000      # 目标总数
BATCH_SIZE = 10         # 每次生成的数量
SEED_SAMPLE_SIZE = 16   # 每次参考的种子数量 (KV Context)
MAX_CONCURRENT_REQUESTS = 64 # 最大并发请求数 (防止 Rate Limit)

# 定义生成策略 (Personas)
STRATEGIES = {
    "kv_structure_clone": {
        "weight": 1.0,
        "desc": "You are a Senior Art Director for Commercial Ads. Your job is to take existing high-performing Key Visual (KV) prompts and 'reskin' them for new products/games, while STRICTLY preserving their commercial composition structure."
    }
}
# =========================================

async def load_seeds_from_jsonl(filepath: str) -> List[str]:
    """从 JSONL 文件加载种子 Prompt"""
    seeds = []
    if not os.path.exists(filepath):
        print(f"Warning: Seed file not found at {filepath}")
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 兼容 byteartist_crawler 的输出格式
                    p = data.get('prompt')
                    if p and len(p) > 10: # 忽略太短的
                        seeds.append(p)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading seed file: {e}")
        
    print(f"Loaded {len(seeds)} seed prompts from {filepath}")
    return seeds

async def generate_batch(llm_client, strategy_name: str, seeds: List[str], count: int) -> List[str]:
    """核心生成逻辑"""
    strategy_desc = STRATEGIES[strategy_name]["desc"]
    
    # 随机抽取种子 (Style Reference)
    sample_size = min(len(seeds), SEED_SAMPLE_SIZE) # 使用顶部配置的参数
    if sample_size == 0:
        return []
    selected_seeds = random.sample(seeds, sample_size)
    
    # 打印选中的种子
    print(f"\n[Selected Seeds (KV Templates) - Top 5 of {sample_size}]")
    for i, s in enumerate(selected_seeds[:5]): # 只打印前5个避免刷屏
        print(f"  {i+1}. {s[:100]}...")
    print("-" * 50)

    ref_seeds_text = "\n".join([f"Template {i+1}: {s}" for i, s in enumerate(selected_seeds)])

    system_prompt = f"""
    Task: Generate {count} Commercial Key Visual (KV) prompts.
    Role: {strategy_desc}
    
    [REFERENCE KV TEMPLATES] (Study these for Composition, Lighting, and Text Layout):
    {ref_seeds_text}
    
    [INSTRUCTIONS]
    1. Analyze the References (Composition, Lighting, Text Layout).
    2. Generate {count} NEW prompts inspired by these styles.
    3. **SUBJECT SWAP**: Keep the "Commercial Vibe" but change the subject/product.
    4. **LANGUAGE**: Write in **CHINESE**, but keep all technical/rendering terms in ENGLISH.
    5. **NO META-COMMENTARY**: Do NOT include prefixes like "Reskin of Template X", "Subject: ...", or "Style: ...". Just output the prompt text itself.
    
    [OUTPUT FORMAT - JSON ONLY]
    You MUST return a valid JSON object with a single key "prompts" containing a list of strings.
    Example:
    {{
      "prompts": [
        "xxxxxxx",
        "xxxxxxxxxxxxx"
      ]
    }}
    Do NOT output any markdown code blocks (like ```json), just the raw JSON string.
    """
    
    user_prompt = f"Generate {count} KV prompts in JSON format now."

    try:
        # 调用 LLM
        messages = [
            {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
        ]
        
        content, error = await llm_client.call_api(messages)
        
        if error:
            print(f"LLM Error: {error}")
            return []
            
        prompts = []
        if content:
            raw_list = []
            try:
                # 尝试清洗 markdown 标记 (有些模型喜欢包一层 ```json ... ```)
                clean_content = content.strip()
                if clean_content.startswith("```"):
                    clean_content = clean_content.strip("`").replace("json", "").strip()
                
                # 核心解析逻辑：直接解析 JSON
                data = json.loads(clean_content)
                
                # 提取列表
                if isinstance(data, dict) and "prompts" in data:
                    raw_list = data["prompts"]
                elif isinstance(data, list):
                    raw_list = data
                    
            except json.JSONDecodeError:
                print("JSON Parse Failed. Fallback to line split.")
                # 如果 JSON 解析失败，退回到按行分割，尝试提取每一行作为 prompt
                raw_list = [line.strip() for line in content.split('\n') if len(line.strip()) > 10]

            # 统一清洗逻辑 (无论来源是 JSON 还是 Fallback 都需要清洗)
            for item in raw_list:
                if not isinstance(item, str) or len(item) < 10:
                    continue

                # 清洗 1: 去除首尾的引号、逗号 (针对 Fallback 提取出的 raw json line)
                clean_item = item.strip('", \t\r\n')

                # 过滤: 如果剩下来的像是 JSON key 或结构 (e.g. "prompts": [) 则跳过
                if clean_item.startswith("prompts") or clean_item.endswith("[") or clean_item.endswith("{") or clean_item == "]" or clean_item == "}":
                    continue

                # 清洗 2: 去除序号前缀 (e.g. "1. ", "- ")
                clean_item = re.sub(r'^[\d\-\.\*\s]+', '', clean_item)

                # 清洗 3: 去除模板引用前缀 (e.g. "Reskin of Template 3:", "仿照[Template 8]结构：", "(Reskin of Template 6)")
                # 3.1 去除括号包裹的引用说明 (e.g. (Reskin of Template 1))
                clean_item = re.sub(r'^\s*[\(（]\s*(?:Reskin of|仿照|模仿|Ref).*?(?:Template|模板).*?[\)）]\s*', '', clean_item, flags=re.IGNORECASE)
                # 3.2 去除冒号结尾的引用说明 (e.g. Reskin of Template 1: )
                clean_item = re.sub(r'^\s*(?:Reskin of|仿照|模仿|Ref).*?(?:Template|模板).*?[:：]\s*', '', clean_item, flags=re.IGNORECASE)

                if len(clean_item) > 10:
                    prompts.append(clean_item)

        return prompts[:count]
        
    except Exception as e:
        print(f"Generation failed: {e}")
        return []

async def main():
    # 1. 初始化 LLM
    llm_config = LLM_API_CONFIGS.get("gemini_3_pro")
    if not llm_config:
        print("Error: Gemini config not found.")
        return
    
    llm_client = Gemini3ProProvider(**llm_config)
    
    # 2. 加载种子
    seeds = await load_seeds_from_jsonl(SEED_FILE_PATH)
    if not seeds:
        print("No seeds found. Using dummy seeds.")
        seeds = ["A futuristic city, neon lights, 8k", "A fantasy landscape with a castle"]

    # 3. 准备输出文件
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    total_generated = 0
    if os.path.exists(OUTPUT_FILE_PATH):
        with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            total_generated = sum(1 for _ in f)
    
    print(f"Starting generation loop. Target: {TARGET_COUNT}. Current: {total_generated}")
    print(f"Concurrency Level: {MAX_CONCURRENT_REQUESTS}")

    # 信号量用于控制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def worker(f_out):
        async with semaphore:
            # 随机选择策略
            strategy = random.choices(
                list(STRATEGIES.keys()),
                weights=[s["weight"] for s in STRATEGIES.values()]
            )[0]
            
            print(f"Generating batch ({strategy})...")
            new_prompts = await generate_batch(llm_client, strategy, seeds, BATCH_SIZE)
            
            if new_prompts:
                unique_in_batch = 0
                for p in new_prompts:
                    # 构造 JSON 对象
                    record = {
                        "source": "synthetic_brainstorm",
                        "prompt": p
                    }
                    # 写入 JSONL (注意：写入文件通常不是线程安全的，但在 asyncio 单线程模型下一般没事)
                    # 为了绝对安全，这里只是简单写入，如果以后用多线程需要加锁
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    unique_in_batch += 1
                f_out.flush()
                return unique_in_batch
            else:
                print("  -> Empty batch.")
                return 0

    # 4. 并发循环生成
    # 我们需要计算还需要生成多少次 batch
    remaining_count = TARGET_COUNT - total_generated
    if remaining_count <= 0:
        print("Target already reached.")
        return

    needed_batches = (remaining_count + BATCH_SIZE - 1) // BATCH_SIZE # 向上取整
    
    with open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as f_out:
        tasks = []
        for _ in range(needed_batches):
            tasks.append(worker(f_out))
        
        # 等待所有任务完成，并统计结果
        results = await asyncio.gather(*tasks)
        total_added = sum(results)
        total_generated += total_added
        
        print(f"Batch processing complete. Added {total_added} prompts. Final Total: {total_generated}")

    print("Done!")
    await llm_client.close()

if __name__ == '__main__':
    asyncio.run(main())
