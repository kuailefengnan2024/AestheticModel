"""
API抽象基类定义模块

功能：定义统一的API接口规范，实现适配器模式的核心抽象层
角色：为不同厂商的API提供统一的接口标准，确保业务层代码与具体API实现解耦
架构：遵循依赖倒置原则，业务逻辑依赖抽象而非具体实现

包含三种核心API抽象：
- BaseLlmProvider: 大语言模型文本生成接口
- BaseImageProvider: AI图片生成接口
- BaseAuditProvider: 内容审核接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from config import settings


class BaseLlmProvider(ABC):
    """
    LLM API提供商的抽象基类。
    所有具体的LLM提供商都应从此类继承。
    """
    def __init__(self):
        """
        初始化基础属性，例如从全局配置中读取重试次数和延迟。
        """
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY

    @abstractmethod
    async def call_api(self, messages: list, **kwargs) -> Tuple[str | None, str | None]:
        """
        调用LLM API并返回文本响应。
        
        Args:
            messages (list): 一个遵循OpenAI格式的消息列表，例如:
                             [{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": "Hello!"}]
            **kwargs: 传递给API客户端的其他参数。
        
        Returns:
            一个元组 (response_text, error_message)。
        """
        pass

    async def close(self):
        """可选的关闭或清理资源的方法。"""
        pass


class BaseImageProvider(ABC):
    """
    图片生成提供商的抽象基类。
    """
    @abstractmethod
    async def call_api(self, prompt: str, **kwargs) -> Tuple[bytes | None, str | None]:
        """
        生成单张图片。

        Args:
            prompt (str): 用于生成图片的提示词。
            **kwargs: 其他可选参数（如size、quality等）

        Returns:
            一个元组 (image_bytes, error_message)。
            - 成功时, image_bytes 是图片的原始字节数据, error_message 为 None。
            - 失败时, image_bytes 为 None, error_message 包含错误描述。
        """
        raise NotImplementedError

    async def close(self):
        """可选的关闭或清理资源的方法。"""
        pass


class BaseAuditProvider(ABC):
    """
    内容审核与视觉理解提供商的抽象基类。
    """
    @abstractmethod
    async def call_api(self, prompt_text: str, **kwargs) -> Tuple[str | None, str | None]:
        """
        调用视觉API的核心方法。

        Args:
            prompt_text (str): 发送给模型的用户文本提示。
            **kwargs: 其他可选的、特定于实现的参数 (如 predefined_messages)。

        Returns:
            一个元组 (content, error_message)。
            - 成功时, content 是API返回的文本内容, error_message 为 None。
            - 失败时, content 为 None, error_message 包含错误描述。
        """
        raise NotImplementedError

    async def close(self):
        """可选的关闭或清理资源的方法。"""
        pass


class BaseImageEditorProvider(ABC):
    """
    图片编辑提供商的抽象基类。
    """
    @abstractmethod
    async def call_api(self, prompt: str, input_image_bytes: bytes, **kwargs) -> Tuple[bytes | None, str | None]:
        """
        根据文本指令编辑单张图片。

        Args:
            prompt (str): 用于编辑图片的文本指令。
            input_image_bytes (bytes): 需要被编辑的原始图片的字节数据。
            **kwargs: 其他可选参数

        Returns:
            一个元组 (output_image_bytes, error_message)。
            - 成功时, output_image_bytes 是编辑后图片的字节数据, error_message 为 None。
            - 失败时, output_image_bytes 为 None, error_message 包含错误描述。
        """
        raise NotImplementedError

    async def close(self):
        """可选的关闭或清理资源的方法。"""
        pass