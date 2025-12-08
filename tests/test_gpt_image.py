import asyncio
import os
import sys
import socket
from urllib.parse import urlparse
from pathlib import Path

# 确保项目根目录在 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from api.image.gpt_image_1 import GptImage1Provider
from config.settings import IMAGE_API_CONFIGS, OUTPUTS_DIR
from utils.logger import logger

# 从 .env 文件加载环境变量
load_dotenv()

def check_network_connectivity(url_str):
    """
    简单的网络连通性检查
    """
    logger.info(f"正在检查网络连通性: {url_str}")
    try:
        parsed = urlparse(url_str)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        
        logger.info(f"尝试解析主机名: {hostname}")
        host_ip = socket.gethostbyname(hostname)
        logger.info(f"DNS 解析成功: {hostname} -> {host_ip}")
        
        logger.info(f"尝试建立 TCP 连接 {host_ip}:{port} ...")
        sock = socket.create_connection((host_ip, port), timeout=5)
        sock.close()
        logger.success(f"TCP 连接成功: {hostname}:{port}")
        return True
    except socket.gaierror:
        logger.error(f"DNS 解析失败: 无法找到主机 {hostname}。请检查 VPN 或网络连接。")
        return False
    except socket.timeout:
        logger.error(f"连接超时: 无法连接到 {hostname}:{port}。")
        return False
    except Exception as e:
        logger.error(f"网络检查失败: {e}")
        return False

async def test_gpt_image_generation():
    """
    一个简单的函数，用于测试与 GPT Image API 的连接和图像生成。
    """
    logger.info("开始 GPT Image API 生成测试...")
    provider = None
    try:
        # 1. 从 settings.py 获取 GPT Image 的配置
        gpt_image_config = IMAGE_API_CONFIGS.get("gpt_image_1")
        if not gpt_image_config:
            logger.error("在 config/settings.py 中未找到 GPT Image 的配置 'gpt_image_1'")
            return

        # 检查 API 密钥是否已加载
        api_key = gpt_image_config.get("api_key")
        if not api_key:
            logger.error("未能加载 GPT_IMAGE_1_API_KEY。请检查您的 .env 文件和 config/settings.py。")
            return
        
        base_url = gpt_image_config.get("base_url")
        if not base_url:
            logger.error("配置中缺少 base_url")
            return

        logger.info(f"找到 GPT Image API Key: {api_key[:4]}...{api_key[-4:]}")

        # 0. 执行网络预检查
        if not check_network_connectivity(base_url):
            logger.warning("网络预检查失败，后续 API 调用可能会失败。")
        
        # 2. 初始化 provider
        logger.info("正在初始化 GptImage1Provider...")
        provider = GptImage1Provider(**gpt_image_config)
        logger.info("Provider 初始化成功。")

        # 3. 定义测试提示词
        # 从 tests/prompt 文件读取提示词
        prompt_path = Path(__file__).parent / "prompt"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            logger.info(f"已从文件读取提示词: {prompt[:50]}...")
        except Exception as e:
            logger.error(f"无法读取提示词文件 {prompt_path}: {e}")
            return
        
        # 4. 调用 API
        # 支持的尺寸: "1024x1024", "1536x1024", "1024x1536"
        target_size = "1024x1536" 
        logger.info(f"正在调用 GPT Image API 生成图片 (尺寸: {target_size})...")
        image_data, error = await provider.call_api(prompt, size=target_size)

        if error:
            logger.error(f"GPT Image API 调用失败，错误信息: {error}")
        elif image_data:
            logger.success("GPT Image API 调用成功！收到图片数据。")
            
            # 5. 保存图片
            output_path = OUTPUTS_DIR / "test_gpt_image_output.png"
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(image_data)
            
            logger.success(f"图片已保存至: {output_path}")
        else:
            logger.warning("API 调用成功但未返回图片数据且无错误信息。")

    except Exception as e:
        logger.error(f"测试过程中发生意外错误: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
    finally:
        if provider:
            await provider.close()
            logger.info("Provider client 已关闭。")

if __name__ == "__main__":
    # 使用 asyncio.run() 来运行异步的主函数
    asyncio.run(test_gpt_image_generation())
