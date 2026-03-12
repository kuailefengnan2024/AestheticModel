# -*- coding: utf-8 -*-
"""
脚本名称: refine_to_kv_format.py
功能描述: 
    此脚本用于将现有的原始 prompt (来自 data/raw_prompts/byteartist_prompts.jsonl) 
    优化、扩充为高质量的 KV (Key Visual) 格式 prompt。
    
    主要任务包括：
    1.  读取 byteartist_prompts.jsonl 中的原始 prompt。
    2.  利用 Gemini 3.0 Pro 模型对每个 prompt 进行分析和润色。
    3.  补充细节：增加光影、材质、构图、色彩、氛围等专业描述。
    4.  优化结构：确保 prompt 结构清晰，便于文生图模型理解（如：主体 + 环境 + 风格 + 技术参数）。
    5.  保持原意：在扩充的同时，保留原始 prompt 的核心创意和意图。
    6.  **新增**：生成一个随机的主标题文案（5-8个字），与原 Prompt 内容契合。
    7.  输出优化后的 prompt 到新的 JSONL 文件 (data/image_prompts_expanded.jsonl)。
"""
import os
import sys
import json
import asyncio
import warnings

# 过滤无关紧要的 Pydantic 警告
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.llm.gemini_3_pro import Gemini3ProProvider
from config.settings import LLM_API_CONFIGS

# ================= 配置区 =================
INPUT_FILE_PATH = "data/raw_prompts/byteartist_prompts.jsonl"
# 修改为新的输出文件名，避免覆盖原始扩充文件
OUTPUT_FILE_PATH = "data/raw_prompts/byteartist_prompts_refined.jsonl"
# 并发控制
MAX_CONCURRENT_REQUESTS = 50 # 同时进行的请求数

# 系统提示词模板 (中文版 + 标题融合)
SYSTEM_PROMPT_TEMPLATE = """
角色：你是一位资深的商业广告创意总监和 Prompt 工程师，专注于制作高质量的 Key Visual (KV) 主视觉。

任务：
你的任务是将一个输入的原始提示词（Raw Prompt）优化为一个**高质量、商业级的 KV 提示词**。

你需要严格遵守以下要求：

1.  **Prompt 优化要求**：
    *   **忠实还原**：必须严格保留原始提示词的核心创意、主体、风格和意图，不得随意更改主体（如：不要把“熊猫”改成“老虎”）。不要自我发挥太多，尽量保持与原作的一致性。
    *   **融入标题**：请根据画面内容，构思一个**5-8个字的中文主标题文案**，并选择一个合适的**字体风格**（如：书法体、黑体、手写体、3D立体字等），将它们自然地融入到 Prompt 的描述中（例如：“画面中央是主标题‘XXXX’，采用金色的书法字体...”）。
    *   **结构清晰**：风格和元提示词保持一致 只扩充缺失细节 不要太长 不要出现kv相关字眼 描述画面内容即可

2.  **输出格式（JSON）**：
    你必须**仅**返回一个合法的 JSON 对象，不要包含任何 markdown 代码块标记（如 ```json），格式如下：
    {
        "refined_prompt": "优化后的完整 Prompt 文本..."
    }


"""

async def refine_single_prompt(llm_client, semaphore, raw_prompt: str, source: str, output_file: str):
    """
    使用 LLM 优化单个 Prompt，完成后立即写入文件
    """
    if not raw_prompt or len(raw_prompt) < 2:
        return

    async with semaphore:
        user_prompt = f"原始提示词：{raw_prompt}"
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # 调用 Gemini 3.0 Pro
            content, error = await llm_client.call_api(messages)
            
            if error:
                print(f"LLM Error: {error}")
                return
            
            if content:
                # 清洗结果 (去除可能的 Markdown 标记)
                cleaned_content = content.strip()
                if cleaned_content.startswith("```"):
                    cleaned_content = cleaned_content.strip("`").replace("json", "").strip()
                
                result = None
                try:
                    result = json.loads(cleaned_content)
                except json.JSONDecodeError:
                    print(f"JSON parse error: {cleaned_content[:50]}...")
                    return

                if result and "refined_prompt" in result:
                    # 立即写入文件 (流式)
                    record = {
                        "source": source,
                        "prompt": result["refined_prompt"] # 核心 KV Prompt (包含标题描述)
                    }
                    # 简单追加写入，单线程 Event Loop 下通常安全，或者用 aiofiles 更好，这里为了简单直接 open
                    # 注意：如果多线程需要锁，但 asyncio 是单线程并发
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    # 打印一个小点表示进度
                    print(".", end="", flush=True)
                else:
                    print(f"JSON format error (missing keys): {cleaned_content[:50]}...")
                
        except Exception as e:
            print(f"Refinement failed for prompt '{raw_prompt[:20]}...': {e}")
            return

async def main():
    # 1. 初始化 LLM
    llm_config = LLM_API_CONFIGS.get("gemini_3_pro")
    if not llm_config:
        print("Error: Gemini 3.0 Pro config not found in settings.")
        return
    
    llm_client = Gemini3ProProvider(**llm_config)
    
    print(f"Loading raw prompts from {INPUT_FILE_PATH}...")
    
    # 2. 读取原始数据
    raw_data = []
    if os.path.exists(INPUT_FILE_PATH):
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    raw_data.append(data)
                except json.JSONDecodeError:
                    continue
    else:
        print(f"Error: Input file {INPUT_FILE_PATH} not found.")
        return

    print(f"Found {len(raw_data)} prompts. Starting refinement...")
    print(f"Output file: {os.path.abspath(OUTPUT_FILE_PATH)}")
    
    # 3. 准备输出文件 (覆盖模式)
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        pass # 清空文件

    # 4. 并发处理 (流式)
    # 不再分批等待，而是创建一个大任务列表，由信号量控制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tasks = []
    for item in raw_data:
        p = item.get("prompt")
        s = item.get("source", "unknown")
        if p:
            tasks.append(refine_single_prompt(llm_client, semaphore, p, s, OUTPUT_FILE_PATH))
    
    # 使用 tqdm 显示进度条 (如果可用)
    try:
        from tqdm.asyncio import tqdm
        await tqdm.gather(*tasks, desc="Refining Prompts")
    except ImportError:
        print(f"Processing {len(tasks)} tasks with concurrency {MAX_CONCURRENT_REQUESTS}...")
        await asyncio.gather(*tasks)
        print("\nRefinement complete.")

    await llm_client.close()
    print(f"Done! All refined prompts saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
