import asyncio
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from api.image.seedream import SeedreamProvider
from config.settings import IMAGE_API_CONFIGS, OUTPUTS_DIR
from utils.logger import logger

# 从 .env 文件加载环境变量
load_dotenv()

async def test_seedream_v3_generation():
    """
    测试 Seedream v3.0 图片生成
    """
    logger.info("开始 Seedream v3.0 生成测试...")
    provider = None
    try:
        # 1. 获取配置
        config = IMAGE_API_CONFIGS.get("seedream")
        if not config:
            logger.error("在 config/settings.py 中未找到配置 'seedream'")
            return

        api_key = config.get("api_key")
        if not api_key:
            logger.error("未能加载 API Key (ARK_API_KEY)。")
            return
        
        logger.info(f"使用模型: {config.get('model')}")

        # 2. 初始化 provider
        provider = SeedreamProvider(**config)
        logger.info("SeedreamProvider (v3) 初始化成功。")

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
        # v3 支持 seed, guidance_scale
        logger.info(f"正在调用 API 生成图片...")
        image_data, error = await provider.call_api(
            prompt, 
            size="1024x1024",
            guidance_scale=7.0,
            seed=42
        )

        if error:
            logger.error(f"API 调用失败: {error}")
        elif image_data:
            logger.success("API 调用成功！收到图片数据。")
            
            # 5. 保存图片
            output_path = OUTPUTS_DIR / "test_seedream_v3.png"
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
    asyncio.run(test_seedream_v3_generation())

