"""
API客户端工厂

根据配置动态创建并返回相应的API客户端实例。
"""
from typing import Type

from .base import BaseLlmProvider, BaseImageProvider, BaseAuditProvider, BaseImageEditorProvider
from .llm.doubao15thinkpro import Doubao15ThinkproProvider
from .llm.openai import OpenAILLMProvider
from .llm.tuzi import TuziProvider
from .llm.gemini_2_5_pro import Gemini25ProProvider
from .vision.doubao_seed_1_6_vision import DoubaoSeedVisionProvider
from .vision.gemini_2_5_pro import Gemini25ProVisionProvider
from .image.seedream import SeedreamProvider
from .image.seedream_4_5 import Seedream45Provider
from .image.gpt_image_1 import GptImage1Provider
from .edit.gpt_image_1 import GptImage1EditorProvider

from config import settings
from utils.logger import logger

class ApiClientFactory:
    """
    一个根据配置创建API客户端的工厂类。
    """
    
    # 注册表：将提供商名称映射到其实现类
    _llm_providers: dict[str, Type[BaseLlmProvider]] = {
        "doubao15thinkpro": Doubao15ThinkproProvider,
        "openai": OpenAILLMProvider,
        "tuzi": TuziProvider,
        "gemini_2_5_pro": Gemini25ProProvider,
    }
    _image_providers: dict[str, Type[BaseImageProvider]] = {
        "seedream": SeedreamProvider,
        "seedream_4_5": Seedream45Provider,
        "gpt_image_1": GptImage1Provider,
    }
    _vision_providers: dict[str, Type[BaseAuditProvider]] = {
        "doubao_seed_1_6_vision": DoubaoSeedVisionProvider,
        "gemini_2_5_pro": Gemini25ProVisionProvider,
    }
    _image_editor_providers: dict[str, Type[BaseImageEditorProvider]] = {
        "gpt_image_1": GptImage1EditorProvider,
    }

    @staticmethod
    def create_llm_client(provider_name: str) -> BaseLlmProvider | None:
        """
        根据指定的提供商名称创建并返回一个LLM API客户端实例。

        :param provider_name: 要创建的提供商名称 (例如, 'tuzi', 'ark')。
        :return: 一个遵循 BaseLlmProvider 接口的客户端实例，如果配置无效则返回None。
        """
        logger.info(f"请求创建LLM客户端，提供商: {provider_name}")

        provider_class = ApiClientFactory._llm_providers.get(provider_name)
        if not provider_class:
            logger.error(f"未知的LLM提供商: {provider_name}")
            raise ValueError(f"未知的LLM提供商: {provider_name}")

        try:
            provider_config = settings.LLM_API_CONFIGS.get(provider_name, {})
            return provider_class(**provider_config)
        except (ValueError, TypeError) as e:
            logger.error(f"创建LLM客户端 '{provider_name}' 时出错: {e}")
            return None

    @staticmethod
    def create_image_client() -> BaseImageProvider | None:
        """
        根据全局配置创建并返回一个图片生成API客户端实例。

        :return: 一个遵循 BaseImageProvider 接口的客户端实例，如果配置无效则返回None。
        """
        provider_name = settings.IMAGE_API_PROVIDER
        logger.info(f"根据配置创建图片生成客户端，提供商: {provider_name}")

        provider_class = ApiClientFactory._image_providers.get(provider_name)
        if not provider_class:
            logger.error(f"未知的图片生成提供商: {provider_name}")
            raise ValueError(f"未知的图片生成提供商: {provider_name}")

        try:
            provider_config = settings.IMAGE_API_CONFIGS.get(provider_name, {})
            return provider_class(**provider_config)
        except (ValueError, TypeError) as e:
            logger.error(f"创建图片生成客户端 '{provider_name}' 时出错: {e}")
            return None

    @staticmethod
    def create_vision_client() -> BaseAuditProvider | None:
        """
        根据全局配置创建并返回一个视觉理解API客户端实例。

        :return: 一个遵循 BaseAuditProvider 接口的客户端实例，如果配置无效则返回None。
        """
        provider_name = settings.VISION_API_PROVIDER
        logger.info(f"根据配置创建视觉理解客户端，提供商: {provider_name}")

        provider_class = ApiClientFactory._vision_providers.get(provider_name)
        if not provider_class:
            logger.error(f"未知的视觉理解提供商: {provider_name}")
            raise ValueError(f"未知的视觉理解提供商: {provider_name}")

        try:
            provider_config = settings.VISION_API_CONFIGS.get(provider_name, {})
            return provider_class(**provider_config)
        except (ValueError, TypeError) as e:
            logger.error(f"创建视觉理解客户端 '{provider_name}' 时出错: {e}")
            return None

    @staticmethod
    def create_image_editor_client() -> BaseImageEditorProvider | None:
        """
        根据全局配置创建并返回一个图片编辑API客户端实例。
        """
        provider_name = settings.IMAGE_EDITOR_API_PROVIDER
        logger.info(f"根据配置创建图片编辑客户端，提供商: {provider_name}")

        provider_class = ApiClientFactory._image_editor_providers.get(provider_name)
        if not provider_class:
            logger.error(f"未知的图片编辑提供商: {provider_name}")
            raise ValueError(f"未知的图片编辑提供商: {provider_name}")

        try:
            provider_config = settings.IMAGE_EDITOR_API_CONFIGS.get(provider_name, {})
            return provider_class(**provider_config)
        except (ValueError, TypeError) as e:
            logger.error(f"创建图片编辑客户端 '{provider_name}' 时出错: {e}")
            return None


def get_llm_api(provider_name: str | None = None) -> BaseLlmProvider | None:
    """
    便捷函数：获取LLM API客户端实例。
    
    Args:
        provider_name (str, optional): 要使用的LLM提供商名称。如果为None，则使用默认设置。
    
    Returns:
        BaseLlmProvider | None: LLM API客户端实例，如果创建失败则返回None。
    """
    if provider_name is None:
        provider_name = settings.LLM_ROLES.get("default")
        
    return ApiClientFactory.create_llm_client(provider_name)


def get_image_api() -> BaseImageProvider | None:
    """
    便捷函数：获取图片生成API客户端实例。
    
    Returns:
        BaseImageProvider | None: 图片生成API客户端实例，如果创建失败则返回None。
    """
    return ApiClientFactory.create_image_client()

def get_vision_api() -> BaseAuditProvider | None:
    """
    便捷函数：获取视觉理解API客户端实例。
    
    Returns:
        BaseAuditProvider | None: 视觉理解API客户端实例，如果创建失败则返回None。
    """
    return ApiClientFactory.create_vision_client()


def get_image_editor_api() -> BaseImageEditorProvider | None:
    """
    便捷函数：获取图片编辑API客户端实例。
    """
    return ApiClientFactory.create_image_editor_client()