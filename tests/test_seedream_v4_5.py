import asyncio
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from api.image.seedream_4_5 import Seedream45Provider
from config.settings import IMAGE_API_CONFIGS, OUTPUTS_DIR
from utils.logger import logger

# 从 .env 文件加载环境变量
load_dotenv()

async def test_seedream_v4_5_generation():
    """
    测试 Seedream v4.5 图片生成
    """
    logger.info("开始 Seedream v4.5 生成测试...")
    provider = None
    try:
        # 1. 获取配置
        config = IMAGE_API_CONFIGS.get("seedream_4_5")
        if not config:
            logger.error("在 config/settings.py 中未找到配置 'seedream_4_5'")
            return

        api_key = config.get("api_key")
        if not api_key:
            logger.error("未能加载 API Key (ARK_API_KEY)。")
            return
        
        logger.info(f"使用模型: {config.get('model')}")

        # 2. 初始化 provider
        provider = Seedream45Provider(**config)
        logger.info("Seedream45Provider 初始化成功。")

        # 3. 定义测试提示词
        # 从 tests/prompt 文件读取提示词
        prompt_path = Path(__file__).parent / "prompt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            logger.info(f"已从文件读取提示词: {prompt[:50]}...")
        except Exception as e:
            logger.error(f"无法读取提示词文件 {prompt_path}: {e}")
            return
        
        # 4. 调用 API
        # v4.5 应该忽略 seed 和 guidance_scale，我们故意传入看看是否报错
        logger.info(f"正在调用 API 生成图片 (尝试传入 seed 和 guidance_scale 以测试兼容性)...")
        image_data, error = await provider.call_api(
            prompt, 
            size="2048x2048",
            guidance_scale=7.0, # 应该被忽略
            seed=42             # 应该被忽略
        )

        if error:
            logger.error(f"API 调用失败: {error}")
        elif image_data:
            logger.success("API 调用成功！收到图片数据。")
            
            # 5. 保存图片
            output_path = OUTPUTS_DIR / "test_seedream_v4_5.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(image_data)
            
            logger.success(f"图片已保存至: {output_path}")
        else:
            logger.warning("无数据返回且无错误。")

    except Exception as e:
        logger.error(f"测试过程中发生意外错误: {e}", exc_info=True)
    finally:
        if provider:
            await provider.close()

if __name__ == "__main__":
    asyncio.run(test_seedream_v4_5_generation())

