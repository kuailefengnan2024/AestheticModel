import asyncio
import os
import sys
import base64
from pathlib import Path
from dotenv import load_dotenv

# 将项目根目录添加到 sys.path
# 获取当前文件的父目录的父目录 (即项目根目录 AestheticModel)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 加载 .env 环境变量
load_dotenv()

# 确保在运行此测试前，您已经正确配置了 .env 文件中的 ARK_API_KEY
# 并且安装了必要的依赖

from api.factory import ApiClientFactory
from api.vision.doubao_seed_1_6_vision import DoubaoSeedVisionProvider
from config import settings

async def test_doubao_vision_api():
    """
    测试 Doubao Seed-1.6 Vision API 的连通性和基本功能。
    """
    
    print("\n" + "="*50)
    print("开始测试 Doubao Seed-1.6 Vision API")
    print("="*50)

    # 1. 准备测试图片
    # 我们需要一张本地图片来测试。
    # 这里我们尝试查找 outputs 目录下是否有之前生成的测试图片，如果没有，就让用户手动提供路径。
    
    # 尝试寻找一些常见的测试图片路径
    potential_images = [
        Path("outputs/test_seedream_v3.png"),
        Path("outputs/test_seedream_v4_5.png"),
        Path("outputs/test_gpt_image_output.png")
    ]
    
    image_path = None
    for p in potential_images:
        if p.exists():
            image_path = str(p)
            print(f"[INFO] 找到测试图片: {image_path}")
            break
            
    if not image_path:
        print("[WARN] 未在 outputs/ 目录下找到自动生成的测试图片。")
        # 创建一个简单的空白/纯色图片用于测试（如果没有找到现有图片）
        # 为了不依赖 PIL，这里我们建议用户先运行生图测试
        print("请先运行生图测试 (如 python tests/test_seedream_v3.py) 生成一张图片，或者手动修改本脚本指定图片路径。")
        return

    # 2. 初始化客户端
    # 我们使用 Factory 来创建，模拟真实环境
    # 确保 settings.py 里 VISION_API_PROVIDER = "doubao_seed_1_6_vision"
    
    # 强制指定 factory 使用 doubao vision (为了测试目的，忽略 settings 的默认值)
    # 但由于 factory 是读取 settings 的，我们直接实例化类或者临时修改 settings
    # 这里直接实例化类更直接
    
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 ARK_API_KEY 环境变量。")
        return

    # 从 settings 获取配置，或者使用硬编码的默认值
    vision_config = settings.VISION_API_CONFIGS.get("doubao_seed_1_6_vision")
    
    print(f"[INFO] 初始化客户端，使用模型: {vision_config['model']}")
    
    client = DoubaoSeedVisionProvider(
        model=vision_config['model'],
        api_key=api_key,
        base_url=vision_config['base_url']
    )

    # 3. 构造 Prompt
    # 模拟我们项目中需要的打分场景
    prompt = """
    请作为一位专业的摄影与审美专家，对这张图片进行评估。
    请返回一个JSON格式的评分，包含以下字段：
    - total_score (0-10分)
    - composition (0-10分，构图)
    - color (0-10分，色彩)
    - lighting (0-10分，光影)
    - brief_comment (简短评价)
    """

    print(f"\n[INFO] 发送请求中... Prompt: {prompt.strip()[:50]}...")
    
    try:
        # 4. 调用 API
        response_content, error = await client.call_api(
            prompt_text=prompt,
            image_path=image_path
        )

        if error:
            print(f"[ERROR] API 调用失败: {error}")
        else:
            print("\n" + "-"*30)
            print("API 响应成功:")
            print("-"*30)
            print(response_content)
            print("-"*30)
            
            # 简单的验证
            if "total_score" in response_content or "json" in response_content.lower():
                 print("[PASS] 响应看起来包含评分信息。")
            else:
                 print("[WARN] 响应内容可能不符合预期格式，请检查输出。")

    except Exception as e:
        print(f"[ERROR] 测试过程中发生异常: {e}")
    finally:
        # 5. 关闭客户端
        await client.close()
        print("\n测试结束。")

if __name__ == "__main__":
    asyncio.run(test_doubao_vision_api())

