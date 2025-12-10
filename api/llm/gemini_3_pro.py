"""
Gemini 3.0 Pro LLM API 适配器

功能：为 Gemini 3.0 Pro 模型提供统一适配。
角色：作为Gemini服务的适配器，将API格式转换为项目统一接口。
架构：实现BaseLlmProvider抽象基类，由ApiClientFactory进行创建。
"""
import httpx
from openai import AsyncOpenAI
from typing import Tuple

from api.base import BaseLlmProvider
from utils.logger import logger

class Gemini3ProProvider(BaseLlmProvider):
    """
    一个适配器，用于通过一个特殊的、兼容OpenAI的端点调用 Gemini 3.0 Pro LLM 服务。
    
    参考 CURL:
    curl --location 'https://search-va.byteintl.net/gpt/openapi/online/v2/crawl?ak=...' \
    --header 'Content-Type: application/json' \
    --header 'X-TT-LOGID: ${your_logid}' \
    --data '{    
    "stream": false,    
    "model": "gemini-3-pro-preview-new",    
    "max_tokens": 4096,    
    "messages": [...],    
    "thinking": {        
    "include_thoughts": true,        
    "budget_tokens": 2000    
    }
    }'
    """
    def __init__(self, api_key: str, base_url: str, model: str, budget_tokens: int = 2000, **kwargs):
        """
        初始化一个兼容性的 OpenAI 客户端。
        """
        if not api_key:
            raise ValueError("API key is required for this provider, but was not found.")
        if not base_url:
            raise ValueError("base_url is required.")
        
        self.model = model
        self.default_budget_tokens = budget_tokens
        # 禁用SSL验证
        # 增加默认超时时间，因为 thinking 模型可能响应较慢
        http_client = httpx.AsyncClient(verify=False, timeout=120.0)

        # 使用 AsyncOpenAI 客户端
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            # 添加 'Api-Key' 头，仿照 2.5 实现
            default_headers={"Api-Key": api_key},
            http_client=http_client
        )
        logger.info(f"Gemini3ProProvider initialized for model: {self.model} with custom endpoint.")

    async def call_api(self, messages: list, **kwargs) -> Tuple[str | None, str | None]:
        """
        调用兼容OpenAI的chat completions接口生成文本。
        """
        try:
            # 关键兼容性改动：转换 'messages' 结构 (仿照 2.5 Pro)
            # 将 "content": "..." 转换为 "content": [{"type": "text", "text": "..."}]
            transformed_messages = []
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    transformed_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                else:
                    transformed_messages.append(msg)

            # 准备 thinking 参数
            # 默认 budget_tokens 为 self.default_budget_tokens，如果 kwargs 中有传则覆盖
            # 注意：必须从 kwargs 中 pop 出 budget_tokens，否则传给 OpenAI SDK 会报错
            budget = kwargs.pop("budget_tokens", self.default_budget_tokens)
            thinking_config = {
                "include_thoughts": True,
                "budget_tokens": budget
            }
            
            # 如果 budget_tokens 为 0，则关闭 thinking (参考用户提供的说明)
            if budget == 0:
                thinking_config = None
            
            # 构造 extra_body
            extra_body = {}
            if thinking_config:
                extra_body["thinking"] = thinking_config

            response = await self.client.chat.completions.create(
                model=self.model, 
                messages=transformed_messages,
                # 添加 X-TT-LOGID 头
                extra_headers={"X-TT-LOGID": ""},
                extra_body=extra_body,
                **kwargs
            )
            content = response.choices[0].message.content
            return content, None
        except Exception as e:
            logger.error(f"Error calling Gemini 3.0 Pro LLM API: {e}")
            return None, str(e)

    async def close(self):
        """关闭客户端会话。"""
        if self.client:
            await self.client.close()
            logger.info("Gemini3ProProvider client closed.")

