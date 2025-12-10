import asyncio
import os
import sys
from dotenv import load_dotenv

# 确保项目根目录在 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 .env 文件加载环境变量
load_dotenv()

from api.llm.gemini_3_pro import Gemini3ProProvider
from config.settings import LLM_API_CONFIGS
from utils.logger import logger

async def test_gemini_3_pro_functionality():
    """
    测试 Gemini 3.0 Pro 的功能，包括 thinking 参数。
    """
    logger.info("开始 Gemini 3.0 Pro API 功能测试...")
    provider = None
    try:
        # 1. 从 settings.py 获取 Gemini 3.0 Pro 的配置
        config = LLM_API_CONFIGS.get("gemini_3_pro")
        if not config:
            logger.error("在 config/settings.py 中未找到 'gemini_3_pro' 的配置")
            return

        # 检查 API 密钥
        api_key = config.get("api_key")
        if not api_key:
            logger.error("未能加载 AZURE_OPENAI_API_KEY。请检查您的 .env 文件。")
            # 尝试从环境变量直接获取（作为备选）
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            if not api_key:
                logger.warning("无法找到 API Key，测试将无法进行实际调用。")
                return
            config["api_key"] = api_key
        
        logger.info(f"找到 API Key: {api_key[:4]}...{api_key[-4:]}")

        # 2. 初始化 provider
        logger.info("正在初始化 Gemini3ProProvider...")
        provider = Gemini3ProProvider(**config)
        logger.info(f"Provider 初始化成功，模型: {provider.model}")

        # ----------------------------------------------------------------------
        # 测试场景 1: 默认 Thinking (budget_tokens=2000)
        # ----------------------------------------------------------------------
        logger.info("\n--- 测试 1: 默认 Thinking (budget=2000) ---")
        messages_1 = [{"role": "user", "content": "1+1等于几？请简短回答。"}]
        logger.info(f"发送消息: {messages_1[0]['content']}")
        
        content_1, error_1 = await provider.call_api(messages_1)
        if error_1:
            logger.error(f"调用失败: {error_1}")
        else:
            logger.success(f"调用成功！响应: {content_1}")

        # ----------------------------------------------------------------------
        # 测试场景 2: 自定义 Thinking (budget_tokens=4000)
        # ----------------------------------------------------------------------
        logger.info("\n--- 测试 2: 自定义 Thinking (budget=4000) ---")
        logger.info("发送消息: 解释量子纠缠（简短）")
        content_2, error_2 = await provider.call_api(
            [{"role": "user", "content": "解释量子纠缠，一句话。"}],
            budget_tokens=4000
        )
        if error_2:
            logger.error(f"调用失败: {error_2}")
        else:
            logger.success(f"调用成功！响应: {content_2}")

        # ----------------------------------------------------------------------
        # 测试场景 3: 关闭 Thinking (budget_tokens=0)
        # ----------------------------------------------------------------------
        logger.info("\n--- 测试 3: 关闭 Thinking (budget=0) ---")
        logger.info("发送消息: 你好")
        content_3, error_3 = await provider.call_api(
            [{"role": "user", "content": "你好"}],
            budget_tokens=0
        )
        if error_3:
            logger.error(f"调用失败: {error_3}")
        else:
            logger.success(f"调用成功！响应: {content_3}")

    except Exception as e:
        logger.error(f"测试过程中发生意外错误: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
    finally:
        if provider:
            await provider.close()
            logger.info("\nProvider client 已关闭。")

if __name__ == "__main__":
    asyncio.run(test_gemini_3_pro_functionality())

