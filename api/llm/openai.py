"""
OpenAI LLM API适配器 (空实现)

功能：为OpenAI系列模型提供一个符合架构的占位符。
角色：作为OpenAI服务的适配器，将OpenAI特定的API格式转换为项目统一接口。
架构：实现BaseLlmProvider抽象基类，由ApiClientFactory进行创建。
"""

from api.base import BaseLlmProvider
import openai

class OpenAILLMProvider(BaseLlmProvider):
    """
    OpenAI API的占位符实现。
    注意：当前未实现具体的API调用逻辑。
    """
    
    def __init__(self, **kwargs):
        """
        初始化OpenAI提供商。
        """
        self.api_params = kwargs
        print(f"警告: OpenAILLMProvider 当前是空实现，使用参数: {self.api_params}。")
        # 在实际实现中，会在这里初始化OpenAI客户端：
        # self.client = openai.OpenAI(api_key=self.api_params.get("api_key"))

    async def call_api(self, prompt_text: str, **kwargs) -> tuple[str | None, str | None]:
        """
        生成文本（空实现）。
        """
        return None, "OpenAI LLM provider is not implemented yet."
    
    async def close(self):
        """关闭客户端（空实现）"""
        pass