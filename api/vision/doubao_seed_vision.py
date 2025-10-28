"""
字节跳动 Seed-1.6 视觉大模型 API 适配器
"""
import os
import sys
import base64
import asyncio
from typing import List, Dict, Any, Tuple

from openai import AsyncOpenAI

# 将项目根目录添加到Python路径中
sys.path.append(str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from api.base import BaseAuditProvider
from utils.logger import logger

def encode_image_to_base64(image_path: str) -> str:
    """将图片文件编码为Base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as e:
        logger.error(f"无法读取图片文件 {image_path}: {e}")
        raise

class DoubaoSeedVisionProvider(BaseAuditProvider):
    """
    使用 openai SDK 调用豆包 Seed-1.6 视觉大模型。
    该模型与 OpenAI 的多模态接口兼容。
    """
    def __init__(self, model: str, api_key: str, base_url: str, **kwargs):
        if not api_key or api_key == "your_ark_api_key_here":
            raise ValueError(
                "ARK_API_KEY 无效或未在 .env 文件中配置。请提供一个有效的火山方舟 API 密钥。"
            )
        
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        logger.debug(f"DoubaoSeedVisionProvider 已初始化，使用模型: {self.model}")

    async def call_api(
        self, 
        prompt_text: str, 
        predefined_messages: List[Dict[str, Any]] = None, 
        **kwargs
    ) -> Tuple[str | None, str | None]:
        """
        使用多模态模型进行API调用。
        """
        image_path = kwargs.get("image_path")
        if not image_path:
            return None, "调用视觉模型必须提供 image_path 参数。"

        try:
            base64_image = await asyncio.to_thread(encode_image_to_base64, image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            logger.debug(f"正在调用视觉API，模型: {self.model}...")
            
            # 在将kwargs传递给SDK之前，移除我们内部使用的、但SDK无法识别的参数
            api_kwargs = kwargs.copy()
            api_kwargs.pop("image_path", None)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **api_kwargs
            )
            
            if not response.choices:
                error_message = "API调用成功，但返回的choices列表为空。"
                logger.error(error_message)
                return None, error_message

            content = response.choices[0].message.content
            logger.debug("视觉API调用成功。")
            return content, None

        except Exception as e:
            logger.error(f"调用视觉API时发生未知错误: {e}", exc_info=True)
            return None, str(e)

    async def close(self):
        """关闭 aiohttp 客户端会话。"""
        if self.client:
            await self.client.close()
            logger.info("DoubaoSeedVisionProvider client closed.")