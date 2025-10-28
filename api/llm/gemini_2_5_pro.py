"""
Gemini 2.5 Pro LLM API 适配器

功能：为 Gemini 2.5 Pro 模型提供统一适配。
角色：作为Gemini服务的适配器，将API格式转换为项目统一接口。
架构：实现BaseLlmProvider抽象基类，由ApiClientFactory进行创建。
"""
import asyncio
import os
from openai import AsyncAzureOpenAI
from typing import Tuple

from api.base import BaseLlmProvider
from utils.logger import logger

class Gemini25ProProvider(BaseLlmProvider):
    """
    一个适配器，用于调用 Gemini 2.5 Pro LLM 服务。
    """
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, model: str, deployment_name: str | None = None, **kwargs):
        """
        初始化AzureOpenAI客户端。
        参数从 config/settings.py 的 LLM_API_CONFIGS 中读取。
        """
        if not api_key:
            raise ValueError("API key is required for this provider, but was not found.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required.")
        
        self.model = model
        # 如果在配置中指定了部署名称(deployment_name)，则使用它；否则，回退到使用模型名称(model)。
        self.deployment_name = deployment_name or model

        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        logger.info(f"Gemini25ProProvider initialized for model: {self.model} (deployment: {self.deployment_name})")

    async def call_api(self, messages: list, **kwargs) -> Tuple[str | None, str | None]:
        """
        调用Azure OpenAI的chat completions接口生成文本。
        """
        try:
            # 关键修正：直接使用传入的 'messages' 列表
            response = await self.client.chat.completions.create(
                model=self.deployment_name, 
                messages=messages,
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
