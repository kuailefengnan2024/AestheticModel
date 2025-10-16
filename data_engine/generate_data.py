# -*- coding: utf-8 -*-
"""
[对应于 README 1.2 数据生成]

自动化生成成对偏好数据集的核心脚本。

工作流程:
1.  **读取输入:**
    - 从一个源目录或列表文件中，获取所有待处理的图片和对应的prompt。
    - 加载 `prompts/judge_prompt.txt` 中的 "裁判Prompt" 模板。

2.  **随机配对:**
    - 从输入中随机抽取两张图片（Image A, Image B）及其关联的文本，组成一个比较对。

3.  **调用VLM API:**
    - 将 "裁判Prompt" 和两张图片的数据，一起发送给外部的VLM API（例如GPT-4V）。
    - 这是一个成本较高的步骤，需要处理API key、请求速率限制和错误重试。

4.  **解析和处理结果:**
    - 解析VLM返回的JSON评分结果。
    - 根据 "total_score" 来判断哪一张是 Winner，哪一张是 Loser。
    - 如果分数相同或返回格式错误，则丢弃该数据对。

5.  **保存数据:**
    - 将有效的数据对 `(Prompt, Winner_Image_Path, Loser_Image_Path)` 及其
      完整的VLM多维度评分，以结构化的格式（如.jsonl）追加写入到输出文件中。

执行命令示例:
`python data_engine/generate_data.py --input_dir /path/to/images --output_file data/preference_data.jsonl --api_key YOUR_API_KEY`
"""
import argparse
import json
import os
import random

def call_vlm_api(prompt, image_1_path, image_2_path):
    """
    一个调用外部VLM API的伪代码函数。
    实际实现需要替换为对特定API（如OpenAI, Google, Anthropic）的调用。
    """
    print(f"正在调用VLM API比较 {image_1_path} 和 {image_2_path}...")
    # TODO: 在这里实现真实的API调用逻辑
    # 1. 读取图片文件
    # 2. 构造API请求体 (包含prompt和图片数据)
    # 3. 发送请求并获取响应
    # 4. 处理可能的API错误

    # 以下是模拟的返回数据
    mock_response = {
      "image_1_scores": {"total_score": 8, "composition": 7, "color": 9},
      "image_2_scores": {"total_score": 6, "composition": 5, "color": 7},
      "reasoning": "图片1的色彩更具吸引力。"
    }
    return mock_response


def main(args):
    # TODO: 1. 加载裁判Prompt模板
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        judge_prompt = f.read()

    # TODO: 2. 获取所有输入图片及其描述
    # 这里的实现取决于你的数据组织方式，可能需要读取一个metadata文件
    image_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg'))]
    
    # 打开输出文件，准备以追加模式写入
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for _ in range(args.num_pairs): # 生成指定数量的数据对
            # TODO: 3. 随机抽取一对不重复的图片进行比较
            if len(image_files) < 2:
                print("图片数量不足，无法配对。")
                break
            
            img1_path, img2_path = random.sample(image_files, 2)
            
            # TODO: 4. 调用VLM API
            try:
                vlm_result = call_vlm_api(judge_prompt, img1_path, img2_path)
                
                # TODO: 5. 解析结果，判断Winner/Loser
                score1 = vlm_result['image_1_scores']['total_score']
                score2 = vlm_result['image_2_scores']['total_score']

                if score1 == score2:
                    print(f"分数相同，跳过: {img1_path} vs {img2_path}")
                    continue
                
                winner_path = img1_path if score1 > score2 else img2_path
                loser_path = img2_path if score1 > score2 else img1_path
                
                # TODO: 6. 构造并写入数据记录
                record = {
                    'winner': winner_path,
                    'loser': loser_path,
                    'vlm_scores': vlm_result
                }
                f.write(json.dumps(record) + '\n')
                print(f"成功生成一条数据: Winner is {os.path.basename(winner_path)}")

            except Exception as e:
                print(f"处理失败: {img1_path} vs {img2_path}, 错误: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用VLM生成成对偏好数据")
    parser.add_argument("--input_dir", type=str, required=True, help="包含源图片的目录")
    parser.add_argument("--output_file", type=str, default="data/preferences.jsonl", help="输出的jsonl数据文件路径")
    parser.add_argument("--prompt_file", type=str, default="data_engine/prompts/judge_prompt.txt", help="裁判Prompt模板文件路径")
    parser.add_argument("--num_pairs", type=int, default=100, help="希望生成的数据对数量")
    # TODO: 可能需要添加 --api_key, --model_name 等参数
    
    args = parser.parse_args()
    main(args)
