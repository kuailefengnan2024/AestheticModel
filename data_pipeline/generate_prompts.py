# -*- coding: utf-8 -*-
"""
[数据生产流程的第一步]

使用大型语言模型 (LLM, 例如 Gemini) 批量生成用于文生图的提示词 (Prompt)。

工作流程:
1.  **加载 "元提示词":**
    - 读取 `data_engine/prompts/image_prompt_generator.txt` 文件，
      这个文件指导LLM如何生成我们需要的提示词。

2.  **调用LLM API:**
    - 连接到LLM的API (例如 Google's Gemini API)。
    - 将 "元提示词" 发送给LLM，请求生成指定数量的提示词。

3.  **解析和保存:**
    - 解析LLM返回的结果。
    - 将生成的一系列提示词逐行写入到指定的输出文件中 (例如 `data/image_prompts.txt`)，
      以供后续的文生图步骤使用。

执行命令示例:
`python data_engine/generate_prompts.py --num_prompts 1000 --output_file data/image_prompts.txt`
"""
import argparse
import os
import asyncio
import re
import sys

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.llm.gemini_2_5_pro import Gemini25ProProvider
from config.settings import LLM_API_CONFIGS

# ==============================================================================
#                                可配置参数
# ==============================================================================
# 在这里直接修改希望生成的提示词数量
DEFAULT_NUM_PROMPTS = 100
# ==============================================================================

async def call_llm_api(meta_prompt, num_prompts):
    """
    调用LLM API来生成提示词。
    """
    print(f"正在调用 Gemini 2.5 Pro API 生成 {num_prompts} 条提示词...")
    
    provider = None
    try:
        # 1. 从 settings.py 中获取 Gemini 的配置
        llm_config = LLM_API_CONFIGS.get("gemini_2_5_pro")
        if not llm_config:
            raise ValueError("在 config/settings.py 中未找到 'gemini_2_5_pro' 的配置")

        # 2. 初始化 Provider
        provider = Gemini25ProProvider(**llm_config)

        # 3. 构造符合Gemini API格式的messages列表
        messages = [{"role": "user", "content": meta_prompt}]
        
        # 4. 发送请求并获取响应
        content, error = await provider.call_api(messages)

        if error:
            print(f"API 调用出错: {error}")
            return []
        
        # 5. 从响应中解析出生成的提示词列表
        # 假设返回的 content 是一个由换行符分隔的字符串
        if content:
            # 使用正则表达式或字符串分割来处理可能存在的编号 (e.g., "1. prompt...")
            prompts = re.split(r'\n\d+\.\s*', content)
            # 过滤掉空的字符串
            return [p.strip() for p in prompts if p.strip()]
        else:
            return []

    finally:
        if provider:
            await provider.close()


async def main(args):
    # 1. 加载 "元提示词" 模板
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            # 将请求生成的数量placeholder替换为具体数字
            meta_prompt = f.read().replace("[数量]", str(DEFAULT_NUM_PROMPTS))
    except FileNotFoundError:
        print(f"错误: 找不到元提示词文件 {args.prompt_file}")
        return

    # 2. 调用 LLM API
    try:
        generated_prompts = await call_llm_api(meta_prompt, DEFAULT_NUM_PROMPTS)
        if not generated_prompts:
            print("API未能返回任何提示词。")
            return
    except Exception as e:
        print(f"调用API时发生错误: {e}")
        return

    # 3. 保存到输出文件
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for prompt in generated_prompts:
            f.write(prompt.strip() + '\n')
            
    print(f"成功生成 {len(generated_prompts)} 条提示词，并保存到 {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用LLM批量生成文生图提示词")
    parser.add_argument("--prompt_file", type=str, default="data_engine/prompts/image_prompt_generator.txt", help="用于生成提示词的'元提示词'文件路径")
    parser.add_argument("--output_file", type=str, default="data/image_prompts.txt", help="保存生成提示词的输出文件路径")
    
    args = parser.parse_args()
    asyncio.run(main(args))
