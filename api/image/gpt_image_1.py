import base64
import uuid
from typing import Tuple

import httpx
from api.base import BaseImageProvider
from config import settings
from utils.logger import logger

class GptImage1Provider(BaseImageProvider):
    """
    一个适配器，用于调用 gpt-image-1 模型生成图片。
    该实现使用 httpx 直接调用，因为其API不完全兼容OpenAI SDK。
    """
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        if not api_key:
            raise ValueError("API key for gpt-image-1 is required.")
        
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient()
        self.output_dir = settings.GENERATED_IMAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"GptImage1Provider (httpx) initialized for model: {self.model}")

    async def call_api(self, prompt: str, **kwargs) -> Tuple[bytes | None, str | None]:
        """
        使用 httpx 直接调用 API 生成图片，并返回图片的原始字节流。
        """
        logger.info(f"Generating image with gpt-image-1 (httpx) for prompt: {prompt[:50]}...")
        
        # 1. 构造请求URL（带ak查询参数）
        request_url = f"{self.base_url}?ak={self.api_key}"
        
        # 2. 构造请求JSON体
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": kwargs.get("n", 1),
            "size": kwargs.get("size", "1024x1024"),
            "quality": kwargs.get("quality", "standard"),
        }

        try:
            # 3. 发送POST请求
            response = await self.client.post(request_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()

            if not result.get("data") or not result["data"][0].get("b64_json"):
                error_msg = result.get("error", {}).get("message", "API did not return image data.")
                raise ValueError(error_msg)

            # 4. 解码Base64数据
            image_base64 = result["data"][0]["b64_json"]
            image_data = base64.b64decode(image_base64)
            
            logger.info(f"Image data successfully received from gpt-image-1 API.")
            return image_data, None

        except httpx.HTTPStatusError as e:
            error_details = e.response.text
            logger.error(f"Error calling gpt-image-1 API (HTTP {e.response.status_code}): {error_details}")
            return None, f"HTTP {e.response.status_code} - {error_details}"
        except Exception as e:
            logger.error(f"Error processing gpt-image-1 response: {e}")
            return None, str(e)

    async def close(self):
        """关闭 httpx 客户端。"""
        if self.client:
            await self.client.aclose()
            logger.info("GptImage1Provider (httpx) client closed.")
