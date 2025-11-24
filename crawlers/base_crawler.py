import requests
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseCrawler(ABC):
    """
    爬虫基类，封装通用的 HTTP 请求逻辑
    """
    def __init__(self, base_url: str, headers: Optional[Dict] = None):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.session.headers.update(self.headers)

    def fetch_page(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[requests.Response]:
        """
        抓取单个页面，包含重试逻辑
        """
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                logger.warning(f"Fetch failed ({url}): {e}. Retrying {attempt + 1}/{max_retries}...")
                time.sleep(2 * (attempt + 1))  # 指数退避
        
        logger.error(f"Failed to fetch {url} after {max_retries} attempts.")
        return None

    @abstractmethod
    def parse(self, response: requests.Response) -> Any:
        """
        解析响应内容的抽象方法，子类必须实现
        """
        pass

    @abstractmethod
    def run(self):
        """
        主运行逻辑
        """
        pass

