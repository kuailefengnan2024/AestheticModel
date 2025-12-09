import base64
import httpx
from typing import Tuple, Any

from openai import AsyncOpenAI
from api.base import BaseImageProvider
from utils.logger import logger

class GeminiImageProvider(BaseImageProvider):
    """
    适配器，用于调用 gemini-3-pro-image-preview 模型生成图片。
    该模型通过 Chat Completion 接口的多模态能力返回图片数据。
    """
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        if not api_key:
            raise ValueError("API key for gemini-3-pro-image is required.")
        
        self.model = model
        
        # 创建禁用SSL验证的 httpx 客户端，适应内部网络环境
        http_client = httpx.AsyncClient(verify=False)
        
        self.client = AsyncOpenAI(
            api_key=api_key, 
            base_url=base_url,
            http_client=http_client,
            default_headers={"Api-Key": api_key} # 借鉴 Gemini25ProProvider 的配置
        )
        
        self.image_config = {
            "aspectRatio": kwargs.get("aspectRatio", "1:1"),
            "imageSize": kwargs.get("imageSize", "1K"),
            "imageOutputOptions": {
                "mimeType": kwargs.get("mimeType", "image/png")
            }
        }
        logger.info(f"GeminiImageProvider initialized for model: {self.model}")

    async def call_api(self, prompt: str, **kwargs) -> Tuple[bytes | None, str | None]:
        """
        调用 Gemini Chat API 生成图片。
        """
        logger.info(f"Generating image with {self.model} for prompt: {prompt[:50]}...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # 准备 extra_body
        # 允许 kwargs 覆盖默认配置
        current_image_config = self.image_config.copy()
        
        # 简单的尺寸映射
        if "size" in kwargs:
             size = kwargs["size"]
             # 假设 size 格式如 "1024x1024", 这里 gemini 可能只接受 aspectRatio
             # 暂时不自动转换，保留默认或 kwargs 中的设置
             pass

        # 合并可能的额外参数
        # 注意：这里我们构造 extra_body 传递给 OpenAI SDK
        extra_body = {
            "response_modalities": ["TEXT", "IMAGE"],
            "image_config": current_image_config
        }

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                stream=False,
                max_tokens=20000,
                messages=messages,
                extra_body=extra_body
            )
            
            # 解析响应
            # 使用 model_dump() 获取原始字典结构，因为 multimodal_contents 不是标准字段
            response_dict = response.model_dump()
            
            choices = response_dict.get("choices", [])
            if not choices:
                return None, "No choices in response."
                
            message = choices[0].get("message", {})
            
            # 检查 multimodal_contents
            multimodal_contents = message.get("multimodal_contents", [])
            
            image_data_base64 = None
            
            for content in multimodal_contents:
                if content.get("type") == "inline_data":
                    inline_data = content.get("inline_data", {})
                    if inline_data.get("mime_type", "").startswith("image/"):
                        image_data_base64 = inline_data.get("data")
                        break
            
            if not image_data_base64:
                 return None, "No image data found in response multimodal_contents."

            image_bytes = base64.b64decode(image_data_base64)
            logger.info(f"Image data successfully received from {self.model}.")
            return image_bytes, None

        except Exception as e:
            logger.error(f"Error calling {self.model} API: {e}")
            return None, str(e)

    async def close(self):
        await self.client.close()
