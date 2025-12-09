import asyncio
import os
import sys
from dotenv import load_dotenv

# 确保项目根目录在 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 .env 文件加载环境变量
load_dotenv()

from api.llm.gemini_2_5_pro import Gemini25ProProvider
from config.settings import LLM_API_CONFIGS
from utils.logger import logger

async def test_gemini_connection():
    """
    一个简单的函数，用于测试与 Gemini API 的连接。
    """
    logger.info("开始 Gemini API 连接测试...")
    provider = None
    try:
        # 1. 从 settings.py 获取 Gemini 的配置
        gemini_config = LLM_API_CONFIGS.get("gemini_2_5_pro")
        if not gemini_config:
            logger.error("在 config/settings.py 中未找到 Gemini 的配置 'gemini_2_5_pro'")
            return

        # 检查 API 密钥是否已加载
        api_key = gemini_config.get("api_key")
        if not api_key:
            logger.error("未能加载 AZURE_OPENAI_API_KEY。请检查您的 .env 文件和 config/settings.py。")
            return
        
        logger.info(f"找到 Gemini API Key: {api_key[:4]}...{api_key[-4:]}")

        # 2. 初始化 provider
        logger.info("正在初始化 Gemini25ProProvider...")
        provider = Gemini25ProProvider(**gemini_config)
        logger.info("Provider 初始化成功。")

        # 3. 构造一个简单的测试消息
        messages = [{"role": "user", "content": "你好，这是一个连接测试。请问 1+1 等于几？"}]
        
        # 4. 调用 API
        logger.info("正在调用 Gemini API...")
        content, error = await provider.call_api(messages)

        if error:
            logger.error(f"Gemini API 调用失败，错误信息: {error}")
        else:
            logger.success(f"Gemini API 调用成功！返回内容: {content}")

    except Exception as e:
        logger.error(f"测试过程中发生意外错误: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
    finally:
        if provider:
            await provider.close()
            logger.info("Provider client 已关闭。")

if __name__ == "__main__":
    # 使用 asyncio.run() 来运行异步的主函数
    asyncio.run(test_gemini_connection())
