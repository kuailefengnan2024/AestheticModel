"""
字节跳动即梦 图片生成API适配器

功能：封装字节跳动即梦系列模型的图片生成API调用，提供统一的图片生成接口
角色：作为字节跳动图片生成服务的适配器，将即梦特定的API格式转换为项目统一接口
架构：实现BaseImageProvider抽象基类，由ApiClientFactory进行创建。

支持模型：jimeng-pro、jimeng-basic等
"""
import time
from typing import Dict, Any
import httpx
import base64
from api.base import BaseImageProvider
from volcenginesdkarkruntime import Ark
from utils.logger import logger


class SeedreamProvider(BaseImageProvider):
    """
    使用火山方舟SDK生成图片的具体实现。
    """

    def __init__(self, model: str, api_key: str, base_url: str, **kwargs):
        """
        初始化客户端。
        """
        if not api_key or api_key == "your_ark_api_key_here":
            raise ValueError(
                "ARK_API_KEY无效或未在.env文件中配置。请提供一个有效的火山方舟API密钥。"
            )
        
        self.client = Ark(base_url=base_url, api_key=api_key, timeout=120)
        self.model = model
        self.api_params = kwargs  # 存储其他可能的参数
        logger.debug(f"SeedreamProvider已初始化，使用模型: {self.model}")

    async def call_api(self, prompt: str, **kwargs) -> tuple[bytes | None, str | None]:
        """
        调用火山方舟API生成图片，并返回图片的字节流。

        Args:
            prompt (str): 图片提示词。
            **kwargs: 包含API所需参数的字典，如 size, guidance_scale, seed, response_format, watermark

        Returns:
            一个元组 (image_bytes, error_message)。
            - 成功时, image_bytes 是图片的原始字节数据, error_message 为 None。
            - 失败时, image_bytes 为 None, error_message 包含错误描述。
        """
        # 合并构造函数参数和调用参数
        params = {**self.api_params, **kwargs}

        # 强制 response_format 为 url 或 b64_json，优先使用 url
        # 这是因为我们需要从适配器内部处理数据获取，而不是将URL或Base64字符串返回给调用者
        response_format = params.get("response_format", "url")
        if response_format not in ["url", "b64_json"]:
            logger.warning(f"不支持的响应格式 '{response_format}'，将强制使用 'url'。")
            response_format = "url"

        # 筛选出SDK支持的参数
        sdk_params = {
            "model": self.model,
            "prompt": prompt,
            "size": params.get("size", "1024x1024"),
            "response_format": response_format,
            "watermark": params.get("watermark", False)
        }
        
        # 只有在参数中明确提供了 guidance_scale 时才添加到 sdk_params
        # 因为某些模型（如 seedream 4.5）不支持此参数
        if "guidance_scale" in params:
            sdk_params["guidance_scale"] = params["guidance_scale"]

        # 只有在参数中明确提供了 seed 时才添加到 sdk_params
        # 因为某些模型（如 seedream 4.5）不支持此参数
        if "seed" in params:
             sdk_params["seed"] = params["seed"]

        try:
            # 使用 asyncio.to_thread 将同步的SDK调用转为异步
            import asyncio
            response = await asyncio.to_thread(self.client.images.generate, **sdk_params)

            if response_format == "url":
                image_url = response.data[0].url
                if not image_url:
                    return None, "API did not return a valid image URL."
                
                # 从URL下载图片字节
                async with httpx.AsyncClient() as client:
                    http_response = await client.get(image_url, timeout=120)
                    http_response.raise_for_status()
                    return http_response.content, None

            elif response_format == "b64_json":
                b64_data = response.data[0].b64_json
                if not b64_data:
                    return None, "API did not return valid base64 image data."
                
                # 解码Base64数据
                image_bytes = base64.b64decode(b64_data)
                return image_bytes, None

        except httpx.HTTPStatusError as e:
            error_details = e.response.text
            logger.error(f"下载豆包图片时出错 (HTTP {e.response.status_code}): {error_details}")
            return None, f"HTTP {e.response.status_code} - {error_details}"
        except Exception as e:
            logger.error(f"豆包图片生成API调用失败: {e}")
            return None, str(e)


    async def close(self):
        """关闭Ark客户端"""
        if self.client:
            try:
                # 使用 asyncio.to_thread 处理可能的阻塞关闭操作
                import asyncio
                await asyncio.to_thread(self.client.close)
            except Exception as e:
                logger.error(f"Error closing Bytedance client: {e}")
            finally:
                self.client = None