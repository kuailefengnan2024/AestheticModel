"""
豆包Ark API客户端

提供豆包API的单次调用能力，专注于API通信
"""

import os
import time
import asyncio
import httpx
import sys
from pathlib import Path
from utils.logger import logger
from api.base import BaseLlmProvider
from volcenginesdkarkruntime import AsyncArk

# API默认配置
DEFAULT_API_TIMEOUT = 60  # 默认超时时间（秒）
DEFAULT_MAX_RETRIES = 3   # 默认最大重试次数
DEFAULT_RETRY_DELAY = 2   # 默认重试延迟（秒）


class Doubao15ThinkproProvider(BaseLlmProvider):
    """
    一个适配器，用于调用火山方舟（Ark）的豆包系列大语言模型。
    """
    def __init__(self, model: str, api_key: str, base_url: str, fallback_prompt: str, **kwargs):
        """
        初始化异步客户端。
        """
        super().__init__()  # 调用父类构造函数以初始化重试等参数
        if not api_key or api_key == "your_ark_api_key_here":
            raise ValueError(
                "ARK_API_KEY无效或未在.env文件中配置。请提供一个有效的火山方舟API密钥。"
            )
        
        self.client = AsyncArk(base_url=base_url, api_key=api_key)
        self.model = model
        self.fallback_prompt = fallback_prompt
        logger.debug(f"Doubao15ThinkproProvider已初始化，使用模型: {self.model}")

    async def call_api(self, prompt_text: str, **kwargs) -> tuple[str | None, str | None]:
        """
        调用火山方舟API。

        :param prompt_text: 发送给模型的用户提示词
        :param kwargs: predefined_messages - 预定义的对话消息
        :return: 一个元组 (content, error_message)。成功时 content 是API返回内容，error_message 为 None；
                 失败时 content 为 None，error_message 包含错误信息。
        """
        messages = kwargs.get("predefined_messages", [
            {"role": "system", "content": "You are a helpful assistant."}
        ])
        messages.append({"role": "user", "content": prompt_text})
        
        # 为了避免传递重复的关键字参数，我们从kwargs中移除已经处理过的键
        api_kwargs = kwargs.copy()
        api_kwargs.pop("predefined_messages", None)
        api_kwargs.pop("messages", None)
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                logger.debug(f"第 {attempt + 1}/{self.max_retries} 次尝试调用API...")

                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **api_kwargs
                )

                duration = time.time() - start_time
                logger.debug(f"API调用成功，耗时: {duration:.2f}s。模型: {completion.model}")

                content = completion.choices[0].message.content
                return content, None

            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                logger.error(f"API调用HTTP状态错误: {e.response.status_code}, Body: {error_body}")
                error_message = f"HTTP Status {e.response.status_code}: {error_body}"
                if attempt >= self.max_retries - 1:
                    return None, error_message
            except Exception as e:
                logger.error(f"API调用异常: {e}")
                error_message = str(e)
                if attempt >= self.max_retries - 1:
                    return None, error_message

            if attempt < self.max_retries - 1:
                retry_delay_with_backoff = self.retry_delay * (2 ** attempt)
                logger.debug(f"等待 {retry_delay_with_backoff:.1f} 秒后重试...")
                await asyncio.sleep(retry_delay_with_backoff)

        return None, "达到最大重试次数后依然失败"
    
    async def close(self):
        """关闭 aiohttp 客户端会话。"""
        if self.client:
            await self.client.close()
            logger.info("Doubao15ThinkproProvider client closed.")