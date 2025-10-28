"""
Gemini 2.5 Pro Vision API 适配器

功能：为 Gemini 2.5 Pro 模型提供视觉理解能力。
角色：将图文多模态输入转换为Gemini兼容的API格式。
架构：实现BaseAuditProvider抽象基类，由ApiClientFactory进行创建。
"""
import asyncio
import base64
import httpx
import mimetypes
from openai import AsyncAzureOpenAI
from typing import Tuple, List, Dict, Any

from api.base import BaseAuditProvider
from utils.logger import logger

async def _get_image_uri(image_path_or_url: str) -> str | None:
    """根据输入是URL还是本地路径，获取图片的 Base64 Data URI。"""
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        # 处理URL
        return await _fetch_and_encode_image_from_url(image_path_or_url)
    else:
        # 处理本地文件路径
        return _encode_local_image(image_path_or_url)

def _encode_local_image(image_path: str) -> str | None:
    """将本地图片文件编码为 Base64 Data URI。"""
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg" # 备选

        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Failed to read or encode local image from path {image_path}: {e}")
        return None

async def _fetch_and_encode_image_from_url(image_url: str) -> str | None:
    """下载图片并将其编码为 Base64 Data URI。"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, follow_redirects=True)
            response.raise_for_status()
        
        image_bytes = response.content
        mime_type = response.headers.get("content-type")
        if not mime_type or not mime_type.startswith("image/"):
            # 如果响应头没有提供，则根据URL猜测
            mime_type, _ = mimetypes.guess_type(image_url)
            if not mime_type:
                mime_type = "image/jpeg" # 作为最后的备选

        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Failed to fetch or encode image from URL {image_url}: {e}")
        return None

class Gemini25ProVisionProvider(BaseAuditProvider):
    """
    一个适配器，用于调用 Gemini 2.5 Pro Vision 服务。
    """
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, model: str, deployment_name: str | None = None, **kwargs):
        """
        初始化AzureOpenAI客户端。
        参数从 config/settings.py 的 VISION_API_CONFIGS 中读取。
        """
        if not api_key:
            raise ValueError("Azure OpenAI API key is required.")
        if not azure_endpoint:
            raise ValueError("Azure OpenAI endpoint is required.")
        
        self.model = model
        # 如果在配置中指定了部署名称(deployment_name)，则使用它；否则，回退到使用模型名称(model)。
        # 这增加了灵活性，以应对模型名称和部署名称不一致的情况。
        self.deployment_name = deployment_name or model

        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        logger.info(f"Gemini25ProVisionProvider initialized for model: {self.model} (deployment: {self.deployment_name})")

    async def call_api(self, prompt_text: str, **kwargs) -> Tuple[str | None, str | None]:
        """
        调用Azure OpenAI的chat completions接口处理图文输入。
        
        Args:
            prompt_text (str): 用户的文本提示。
            **kwargs: 必须包含 'image_url' (str) 或 'image_path' (str) 来指定图片。
        """
        image_url = kwargs.get("image_url")
        image_path = kwargs.get("image_path")

        if not image_url and not image_path:
            return None, "Image URL or local path is required for Vision API call."
        
        image_source = image_url or image_path
        
        # 将URL或本地路径统一转换为Base64 Data URI
        encoded_image_uri = await _get_image_uri(image_source)
        if not encoded_image_uri:
            return None, f"Could not process image from source: {image_source}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": encoded_image_uri}},
                ],
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name, # 使用 deployment_name 而不是 model
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 2048) 
            )
            content = response.choices[0].message.content
            return content, None
        except Exception as e:
            logger.error(f"Error calling Gemini 2.5 Pro Vision API: {e}")
            return None, str(e)

    async def close(self):
        """关闭 aiohttp 客户端会话。"""
        if self.client:
            await self.client.close()
            logger.info("Gemini25ProVisionProvider client closed.")
