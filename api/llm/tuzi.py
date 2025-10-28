"""
Tuzi LLM API适配器

功能：为 tuzi.com (Gemini) 模型提供统一接口。
角色：作为tuzi.com服务的适配器。
架构：实现BaseLlmProvider抽象基类，由ApiClientFactory进行创建。
"""

from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from api.base import BaseLlmProvider
from utils.logger import logger

class TuziProvider(BaseLlmProvider):
    """
    使用openai SDK调用 tuzi.com API。
    """

    def __init__(self, **kwargs):
        """
        初始化Tuzi提供商。

        Args:
            **kwargs: 包含api_key, base_url, model等参数的字典。
        """
        super().__init__()
        self.api_params = kwargs
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_params.get("api_key"),
                base_url=self.api_params.get("base_url")
            )
            logger.info(f"Tuzi.com LLM提供商初始化成功。")
        except Exception as e:
            logger.error(f"初始化Tuzi.com客户端时发生错误: {e}")
            raise ValueError(f"初始化Tuzi.com客户端失败: {e}")

    async def call_api(self, prompt_text: str, **kwargs) -> tuple[str | None, str | None]:
        """
        调用 tuzi.com 的聊天模型。

        Args:
            prompt_text (str): 发送给模型的用户提示词。
            **kwargs: 忽略此处的额外参数，使用初始化时传入的参数。

        Returns:
            一个元组 (content, error_message)。
            - 成功时, content 是API返回的文本内容, error_message 为 None。
            - 失败时, content 为 None, error_message 包含错误描述。
        """
        try:
            model = self.api_params.get("model")
            if not model:
                return None, "模型名称未在配置中指定"

            completion = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text
                    }
                ]
            )
            content = completion.choices[0].message.content
            logger.debug(f"Tuzi.com API调用成功，返回内容：{content[:50]}...")
            return content.strip(), None
        except APIConnectionError as e:
            logger.error(f"Tuzi.com API连接错误: {e.__cause__}")
            return None, f"API连接错误: {e.__cause__}"
        except RateLimitError as e:
            logger.error(f"Tuzi.com API达到速率限制: {e.status_code} {e.response}")
            return None, f"API达到速率限制"
        except Exception as e:
            logger.error(f"调用Tuzi.com API时发生未知错误: {e}")
            return None, f"调用API时发生未知错误: {e}"

    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.close() 