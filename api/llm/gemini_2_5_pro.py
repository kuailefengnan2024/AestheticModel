"""
Gemini 2.5 Pro LLM API 适配器

功能：为 Gemini 2.5 Pro 模型提供统一适配。
角色：作为Gemini服务的适配器，将API格式转换为项目统一接口。
架构：实现BaseLlmProvider抽象基类，由ApiClientFactory进行创建。
"""
import asyncio
import os
import httpx
from openai import AsyncOpenAI
from typing import Tuple

from api.base import BaseLlmProvider
from utils.logger import logger

class Gemini25ProProvider(BaseLlmProvider):
    """
    一个适配器，用于通过一个特殊的、兼容OpenAI的端点调用 Gemini 2.5 Pro LLM 服务。
    
    核心兼容性改动:
    1. 使用通用的 `AsyncOpenAI` 客户端，而非 `AsyncAzureOpenAI`。
    2. 传入自定义的 `httpx.AsyncClient` 来禁用SSL证书验证 (verify=False)，以适应特殊的网络环境。
    3. 添加了非标准的 `Api-Key` 请求头。
    4. 在 `call_api` 中添加了非标准的 `X-TT-LOGID` 请求头。
    5. 对传入的 `messages` 中的 'user' 内容格式进行了转换，以符合目标端点的特殊要求。
    """
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        """
        初始化一个兼容性的 OpenAI 客户端。
        参数从 config/settings.py 的 LLM_API_CONFIGS 中读取，并进行了特殊适配。
        """
        if not api_key:
            raise ValueError("API key is required for this provider, but was not found.")
        if not base_url:
            raise ValueError("base_url is required.")
        
        self.model = model
        # 关键兼容性改动：创建一个禁用SSL验证的 httpx 客户端
        # 这对于连接到使用自签名证书的公司内部网络端点通常是必需的
        http_client = httpx.AsyncClient(verify=False)

        # 使用通用的 AsyncOpenAI 客户端，并注入我们的自定义设置
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            # 关键兼容性改动：添加一个非标准的 'Api-Key' 头
            default_headers={"Api-Key": api_key},
            http_client=http_client
        )
        logger.info(f"Gemini25ProProvider initialized for model: {self.model} with custom endpoint and SSL verification disabled.")

    async def call_api(self, messages: list, **kwargs) -> Tuple[str | None, str | None]:
        """
        调用兼容OpenAI的chat completions接口生成文本。
        """
        try:
            # 关键兼容性改动：转换 'messages' 结构
            # 将 "content": "..." 转换为 "content": [{"type": "text", "text": "..."}]
            transformed_messages = []
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    transformed_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                else:
                    # 对于其他角色或格式，直接保留
                    transformed_messages.append(msg)

            response = await self.client.chat.completions.create(
                model=self.model, 
                messages=transformed_messages,
                # 关键兼容性改动：添加一个额外的请求头
                extra_headers={"X-TT-LOGID": ""},
                **kwargs
            )
            content = response.choices[0].message.content
            return content, None
        except Exception as e:
            logger.error(f"Error calling Gemini 2.5 Pro LLM API: {e}")
            return None, str(e)

    async def close(self):
        """关闭 aiohttp 客户端会话。"""
        if self.client:
            await self.client.close()
            logger.info("Gemini25ProProvider client closed.")
