from .base_crawler import BaseCrawler
from bs4 import BeautifulSoup
import json
import os
import time

# ================= 配置区 =================
# TODO: 请在浏览器按 F12 -> Network -> 刷新页面 -> 找到第一个请求 -> Copy Cookie 粘贴到下面
COOKIE_STRING = """
ba_space_id=e75d595ffcb142d29596bd8b7b472c44; email=handongyang.729@bytedance.com; user_token=JTdCJTIybmFtZSUyMiUzQSUyMiVFOSU5RiVBOSVFNSU4NiVBQyVFOSU5OCVCMyUyMiUyQyUyMmZ1bGxfbmFtZSUyMiUzQSUyMiVFOSU5RiVBOSVFNSU4NiVBQyVFOSU5OCVCMyUyMDc3MTMwMTglMjIlMkMlMjJlbWFpbCUyMiUzQSUyMmhhbmRvbmd5YW5nLjcyOSU0MGJ5dGVkYW5jZS5jb20lMjIlMkMlMjJwaWN0dXJlJTIyJTNBJTIyaHR0cHMlM0ElMkYlMkZzMS1pbWZpbGUuZmVpc2h1Y2RuLmNvbSUyRnN0YXRpYy1yZXNvdXJjZSUyRnYxJTJGdjJfMzMyZTVmMDYtMDI5NS00M2Q5LWJhZDUtNzk5Y2QzZWQ2YTJnfiUzRmltYWdlX3NpemUlM0QyNDB4MjQwJTI2Y3V0X3R5cGUlM0QlMjZxdWFsaXR5JTNEJTI2Zm9ybWF0JTNEcG5nJTI2c3RpY2tlcl9mb3JtYXQlM0Qud2VicCUyMiUyQyUyMmVtcGxveWVlX2lkJTIyJTNBJTIyNzcxMzAxOCUyMiUyQyUyMmVtcGxveWVlX251bWJlciUyMiUzQSUyMjc3MTMwMTglMjIlMkMlMjJ0ZW5hbnRfYWxpYXMlMjIlM0ElMjJieXRlZGFuY2UlMjIlMkMlMjJ1c2VyX2lkJTIyJTNBJTIyaGQxaWgwZDZwdm44eHMxaWhpN3YlMjIlN0Q=; X-Risk-Browser-Id=d8e1cc40e16ed5912421ac49da0a271fa4e9fbf9da49273a6d203be3b69700e9; people-lang=zh; bdsso_lt_c0=djEuMC4w.2SrELuhHxV4HXyhUeplgZc04mZKvOJz5M5O02txtx7UrzxxLc5Lg4VB-a89XWCuiggYfcUOngd2mYyNhDnLdEpgoWh8tFaEsmV2IHCTmNmzpVbwuuf_lAneIUeDETO-g37rn9RHw2mqPD7--halRR0vFUrMTf0AnVDezkuCBZnRsEFevIBsSpZ7Qhr1z_x-KK_xnIMY5Pa1nphS36RRoEeB7tOuzPZByMi1KVoGz0hC4G3txQ1YIOo-5zRr2bDOgcaGx5uvzvMOCQm4DPuyptQ5HqAli0YDFOi2NMnrGbj0.dmoAx3riAQFdZlB1ulFfNiD8FAE; bdsso_lt_c0_ss=djEuMC4w.2SrELuhHxV4HXyhUeplgZc04mZKvOJz5M5O02txtx7UrzxxLc5Lg4VB-a89XWCuiggYfcUOngd2mYyNhDnLdEpgoWh8tFaEsmV2IHCTmNmzpVbwuuf_lAneIUeDETO-g37rn9RHw2mqPD7--halRR0vFUrMTf0AnVDezkuCBZnRsEFevIBsSpZ7Qhr1z_x-KK_xnIMY5Pa1nphS36RRoEeB7tOuzPZByMi1KVoGz0hC4G3txQ1YIOo-5zRr2bDOgcaGx5uvzvMOCQm4DPuyptQ5HqAli0YDFOi2NMnrGbj0.dmoAx3riAQFdZlB1ulFfNiD8FAE; bd_sso_3b6da9=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjQ1OTUzMjAsImlhdCI6MTc2Mzk5MDUyMCwiaXNzIjoic3NvLmJ5dGVkYW5jZS5jb20iLCJzdWIiOiJoZDFpaDBkNnB2bjh4czFpaGk3diIsInRlbmFudF9pZCI6ImhncTN0Y2NwM2kxc2pqbjU4emlrIn0.RBXX5vBcrLvUpYnzWzrnUPVb5VWIzbeuL8pWS6duU5GIEm3sZHcy_LttnD2m3ulF1wxv8pP87InD9y8JVZHWOk7gLT_Oyxk2kmdShYJjo32FIuv54ndPWMSiF1Fkqryjv5LzHzwnnUTrYfGrYaNEiz0g_5EajAXwOC5YEAkieHFoisSrFSnOCUK704Y4zKicrBtPNS7EuqhEgxCMvWVD7IfVnjUB6JUCsLt9fzIsiGDsm4YeZXVnLbZkEwR9CxBYnp2pGORsGEBgu5n3GPtPOB1vQZRAiyp27aur6qmc6wCRaj427qwIVWRb3TvtBaDjmYDmo9DFpTjmJe_MX7HOUA
"""
# =========================================

class ByteartistCrawler(BaseCrawler):
    def __init__(self, output_dir="data/raw_prompts"):
        super().__init__(base_url="https://byteartist-beta.bytedance.net")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 注入 Cookie 到 headers
        if "这里粘贴" not in COOKIE_STRING and COOKIE_STRING.strip():
            self.session.headers.update({
                "Cookie": COOKIE_STRING.strip().replace('\n', '')
            })
        else:
            print("Warning: Cookie not set! Internal sites will likely reject the request.")
        
    def parse(self, response):
        """
        这里是核心解析逻辑。
        因为不知道 Byteartist 是 SSR (HTML) 还是 CSR (API)，
        这里提供了两种模板。请你根据实际情况取消注释并修改。
        """
        prompts = []
        
        # --- 情况 A: 网站是 API 驱动的 (返回 JSON) ---
        try:
            # 尝试解析 JSON
            # 注意：通常 API 返回的结构比较复杂，需要你打印 print(data) 来看结构
            data = response.json()
            
            # 假设结构示例 (需要根据真实情况修改):
            # data['data']['feed_list'] -> [item1, item2...]
            # item['prompt'] -> "string"
            
            # 这是一个猜测的通用提取逻辑，你需要断点调试确认
            if isinstance(data, dict):
                # 尝试找常见的列表字段名
                items = data.get('data', {}).get('list', []) or data.get('feed', []) or []
                for item in items:
                    # 尝试找 prompt 字段
                    p = item.get('prompt') or item.get('text') or item.get('caption')
                    if p:
                        prompts.append(p)
                        
        except json.JSONDecodeError:
            # 不是 JSON，那可能是 HTML
            pass

        # --- 情况 B: 网站是静态 HTML ---
        if not prompts:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 示例：假设 Prompt 在 <div class="prompt-text"> 中
            # 你需要按 F12 检查 Byteartist 的真实类名
            # elements = soup.find_all('div', class_='prompt-text')
            # for el in elements:
            #     prompts.append(el.get_text(strip=True))
            
        return prompts

    def run(self, pages=5):
        print(f"Starting crawl for Byteartist (Internal)... Target pages: {pages}")
        
        if "这里粘贴" in COOKIE_STRING:
            print("ERROR: Please update the COOKIE_STRING in byteartist_crawler.py first!")
            return

        all_prompts = set()
        
        for page in range(1, pages + 1):
            # 构造 URL (需要你根据真实翻页规则修改)
            # 假设 API 路径是 /api/feed/list ??? 需要抓包确认
            # 暂时用首页 URL 代替
            target_url = f"{self.base_url}/api/community/feed/list" # 猜测的 API 路径
            
            # 猜测的分页参数
            params = {
                "page": page,
                "limit": 20,
                "offset": (page-1)*20
            }
            
            print(f"Fetching page {page}...")
            response = self.fetch_page(target_url, params=params)
            
            if response:
                # 调试用：如果是第一次跑，打印响应看看结构
                if page == 1:
                    print(f"DEBUG Response (First 500 chars): {response.text[:500]}")
                
                page_prompts = self.parse(response)
                if page_prompts:
                    print(f"  Found {len(page_prompts)} prompts.")
                    all_prompts.update(page_prompts)
                else:
                    print("  No prompts found. Check parsing logic or URL.")
            
            time.sleep(1)

        # 保存结果
        output_path = os.path.join(self.output_dir, "byteartist_prompts.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for p in all_prompts:
                f.write(p + "\n")
                
        print(f"Done! Saved {len(all_prompts)} unique prompts to {output_path}")

if __name__ == "__main__":
    crawler = ByteartistCrawler()
    crawler.run()
