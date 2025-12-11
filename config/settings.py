"""
统一配置模块

集中管理项目中所有可配置的参数，包括：
1. 全局流水线控制参数 (如任务数量限制)
2. 核心工具配置 (如API厂商、模型参数等)
3. 文件输入输出路径
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
# 这行代码会自动查找项目根目录下的 .env 文件并加载它
load_dotenv()

# ==============================================================================
# 1. 全局流水线控制 (Global Pipeline Controls)
# ==============================================================================
# 这个部分放置最高优先级的、影响整个流水线的开关。

# 默认API任务处理数量限制。设为None则不限制，设为数字则限制为该数字。
# 此设置会影响所有调用API的步骤。
API_TASK_LIMIT = 30


# ==============================================================================
# 2. 基础路径配置 (Base Path Configurations)
# ==============================================================================
# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 输出目录
OUTPUTS_DIR = BASE_DIR / "outputs"
FINAL_OUTPUTS_DIR = OUTPUTS_DIR / "final"
GENERATED_IMAGES_DIR = FINAL_OUTPUTS_DIR / "generated_images"
PASSED_AUDIT_DIR = FINAL_OUTPUTS_DIR / "passed"
FAILED_AUDIT_DIR = FINAL_OUTPUTS_DIR / "failed"
REFERENCE_IMAGES_DIR = OUTPUTS_DIR / "reference_images"  # 参考图片的输出目录
PROJECT_DB_PATH = OUTPUTS_DIR / "project_data.db"

# 确保输出目录存在
def create_output_dirs():
    """创建所有必需的输出文件夹"""
    dirs_to_create = [
        OUTPUTS_DIR,
        FINAL_OUTPUTS_DIR,
        GENERATED_IMAGES_DIR,
        PASSED_AUDIT_DIR,
        FAILED_AUDIT_DIR,
        REFERENCE_IMAGES_DIR
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

create_output_dirs()


# ==============================================================================
# 3. 工具配置 (Tools Configurations)
# ==============================================================================
# 此处配置严格对应 tools/ 目录下的各个工具脚本。

# ------------------------------------------------------------------------------
# 配置: tools/text_generator.py
# ------------------------------------------------------------------------------
# 为不同任务场景指定LLM提供商
LLM_ROLES = {
    "brain": "gemini_2_5_pro",      # “大脑”(Planner)等核心决策任务。可选值: "tuzi", "doubao15thinkpro", "bytedance", "openai", "gemini_2_5_pro"
    "default": "doubao15thinkpro",     # 默认或工具类任务。可选值: "tuzi", "doubao15thinkpro", "bytedance", "openai", "gemini_2_5_pro"
}

# 所有可用的LLM API的详细参数
LLM_API_CONFIGS = {
    "tuzi": {
        "model": "gemini-2.5-pro-preview-06-05",
        "api_key": os.environ.get("TUZI_API_KEY"),
        "base_url": "https://api.tu-zi.com/v1",
        "fallback_prompt": "一个充满想象力的创意场景"
    },
    "doubao15thinkpro": {
        "model": "doubao-1-5-thinking-pro-250415",
        "api_key": os.environ.get("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "fallback_prompt": "一个充满想象力的创意场景"
    },
    "gemini_2_5_pro": {
        "model": "gemini-2.5-pro",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "base_url": "https://genai-va-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
    },
    "gemini_3_pro": {
        "model": "gemini-3-pro-preview-new",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "base_url": "https://genai-va-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi",
        "budget_tokens": 2000,
    }
}


# ------------------------------------------------------------------------------
# 配置: tools/image_generator.py
# ------------------------------------------------------------------------------
IMAGE_API_PROVIDER = "seedream"  # 可选值: "seedream", "gpt_image_1"

# 所有可用的图片API的详细参数
# 2048x2048是最大尺寸
IMAGE_API_CONFIGS = {
    "seedream": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seedream-3-0-t2i-250415",
        "size": "2048x1024",
        "guidance_scale": 3.0,
        "seed": -1,
        "response_format": "url",
        "watermark": False,
        "api_key": os.environ.get("ARK_API_KEY")
    },
    "seedream_4_5": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seedream-4-5-251128",
        "size": "2048x2048",
        "response_format": "url",
        "watermark": False,
        "api_key": os.environ.get("ARK_API_KEY")
    },
    "gpt_image_1": {
        "model": "gpt-image-1",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "base_url": "https://genai-va-og.tiktok-row.org/gpt/openapi/online/v2/crawl/openai/images/generations",
        "size": "1024x1024", # 支持 1024x1024, 1536x1024, 1024x1536, auto
        "quality": "standard", # 支持 high, medium, low, auto
    },
    "gemini_3_pro_image": {
        "model": "gemini-3-pro-image-preview",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "base_url": "https://genai-va-og.tiktok-row.org/gpt/openapi/online/multimodal/crawl/openai/deployments/gpt_openapi",
        # 路径说明：
        # 1. Cluster: 该模型属于 multimodal 集群，需使用 .../online/multimodal/crawl 路径 (区别于文本模型的 .../online/v2/crawl)。
        # 2. Compatibility: 需保留 /openai/deployments/gpt_openapi 后缀，以适配网关的 OpenAI 兼容接口。
    }
}

# ------------------------------------------------------------------------------
# 配置: tools/image_search.py
# ------------------------------------------------------------------------------
# 注意：此工具使用Selenium直接爬取，不使用Apify。
# Pinterest Cookie 文件的路径 (如果存在的话)
PINTEREST_COOKIES_PATH = BASE_DIR / "config" / "pinterest_cookies.json"


# ------------------------------------------------------------------------------
# 配置: tools/vision.py
# ------------------------------------------------------------------------------
VISION_API_PROVIDER = "doubao_seed_1_6_vision" # 可选值: "doubao_seed_1_6_vision", "gemini_2_5_pro"

# 所有可用的Vision API的详细参数
VISION_API_CONFIGS = {
    "doubao_seed_1_6_vision": {
        "model": "doubao-seed-1-6-vision-250815",
        "api_key": os.environ.get("ARK_API_KEY"),
        "base_url": "https://ark.cn-beijing.volces.com/api/v3"
    },
    "gemini_2_5_pro": {
        "model": "gemini-2.5-pro",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": "https://search-va.byteintl.net/gpt/openapi/online/v2/crawl",
        "api_version": "2024-03-01-preview"
    }
}

# ------------------------------------------------------------------------------
# 配置: tools/image_search.py (原 g_find_references.py)
# ------------------------------------------------------------------------------
# 使用 Apify 的 Pinterest Scraper Actor
# 你的 Apify API Token。建议通过环境变量 APIFY_TOKEN 进行设置。
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
# 要运行的 Apify Actor 的 ID
APIFY_PINTEREST_ACTOR_ID = "apify/pinterest-scraper"


# ------------------------------------------------------------------------------
# 配置: tools/image_editor.py (新增)
# ------------------------------------------------------------------------------
IMAGE_EDITOR_API_PROVIDER = "gpt_image_1"  # 当前唯一可选值: "gpt_image_1"

# 所有可用的图片编辑API的详细参数
IMAGE_EDITOR_API_CONFIGS = {
    "gpt_image_1": {
        "model": "gpt-image-1",
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "base_url": "https://search-va.byteintl.net/gpt/openapi/online/v2/crawl", # 假设与生成使用相同的基础端点
    }
}


# ==============================================================================
#  4. 通用API调用行为 (适用于所有API客户端)
# ==============================================================================

# 单个API请求失败后，允许自动重试的最大次数。
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# 首次重试前的基础等待秒数。后续每次重试的等待时间会以2的指数倍递增。
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# 单个API请求的最长等待时间（秒）。超过此时长仍未收到响应，则视为请求超时。
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "60"))


# ==========================================================================
#  5. 类方法 (可考虑移除或重构)
# ==========================================================================

@classmethod
def get_database_path(cls) -> str:
    """获取数据库路径的字符串形式"""
    return str(cls.PROJECT_DB_PATH)

# --- API 密钥与机密 ---
# 建议在生产环境中从环境变量加载这些值，此处为占位符
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BYTEDANCE_ARK_API_KEY = os.getenv("BYTEDANCE_ARK_API_KEY")
