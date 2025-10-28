import base64
import uuid
import json
from typing import Tuple

import httpx
from api.base import BaseImageEditorProvider
from config import settings
from utils.logger import logger

class GptImage1EditorProvider(BaseImageEditorProvider):
    """
    一个适配器，用于调用 gpt-image-1 模型来编辑图片。
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
        logger.info(f"GptImage1EditorProvider (httpx) initialized for model: {self.model}")

    async def call_api(self, prompt: str, input_image_bytes: bytes, **kwargs) -> Tuple[bytes | None, str | None]:
        """
        使用 httpx 以 multipart/form-data 格式调用 API 编辑图片，并返回字节流。
        """
        logger.info(f"Editing image with gpt-image-1 (multipart) for prompt: {prompt[:50]}...")
        
        # 1. 构造正确的图片编辑专用URL
        # 这是解决问题的关键：编辑功能需要一个特定的URL路径
        request_url = f"{self.base_url.rstrip('/')}/openai/images/edits?ak={self.api_key}"
        
        # 2. 准备 multipart/form-data
        # 修正：将所有参数都放入 files 中，以更精确地模拟表单提交
        # 这可以解决某些挑剔的API后端无法正确解析混合 data 和 files 的问题
        files = {
            'prompt': (None, prompt),
            'model': (None, self.model),
            'response_format': (None, 'b64_json'),
            'n': (None, str(kwargs.get("n", 1))),
            'size': (None, kwargs.get("size", "1024x1024")),
            'image[]': ("input_image.png", input_image_bytes, "image/png"),
        }

        try:
            # 3. 发送 multipart/form-data 请求
            response = await self.client.post(request_url, files=files, timeout=180)
            response.raise_for_status()
            
            # 关键调试步骤：尝试解析JSON，如果失败，则记录原始响应文本
            try:
                result = response.json()
            except json.JSONDecodeError:
                logger.error(f"API响应不是有效的JSON。状态码: {response.status_code}, 响应文本: {response.text}")
                return None, f"API response was not valid JSON. Raw text: {response.text}"

            if not result.get("data") or not result["data"][0].get("b64_json"):
                error_msg = result.get("error", {}).get("message", "API did not return image data.")
                raise ValueError(error_msg)

            # 4. 解码返回的Base64数据
            output_image_base64 = result["data"][0]["b64_json"]
            output_image_bytes = base64.b64decode(output_image_base64)
            
            logger.info("Image edit data successfully received from gpt-image-1 API.")
            return output_image_bytes, None

        except httpx.HTTPStatusError as e:
            error_details = e.response.text
            logger.error(f"Error calling gpt-image-1 edit API (HTTP {e.response.status_code}): {error_details}")
            return None, f"HTTP {e.response.status_code} - {error_details}"
        except Exception as e:
            logger.error(f"Error processing gpt-image-1 edit response: {e}")
            return None, str(e)

    async def close(self):
        """关闭 httpx 客户端。"""
        if self.client:
            await self.client.aclose()
            logger.info("GptImage1EditorProvider (httpx) client closed.")
