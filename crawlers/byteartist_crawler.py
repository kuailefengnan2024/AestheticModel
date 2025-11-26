import sys
import os

# 允许直接运行此脚本
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_crawler import BaseCrawler
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
        核心解析逻辑 (根据用户提供的真实 JSON 结构适配)
        结构: root -> data -> artworks -> [list] -> item['prompt']
        """
        prompts = []
        
        try:
            data = response.json()
            
            # 1. 定位到列表
            artworks = data.get('data', {}).get('artworks', [])
            
            if not artworks:
                # 尝试兼容其他可能的字段名 (鲁棒性)
                artworks = data.get('data', {}).get('list', []) or data.get('feed', [])
                
            # 2. 遍历提取 Prompt
            for item in artworks:
                # 优先取 'prompt' 字段
                p = item.get('prompt')
                
                # 如果为空，尝试去 configs -> inputs 里面找 (有些复杂的结构藏在这里)
                if not p:
                    inputs = item.get('configs', {}).get('inputs', [])
                    if isinstance(inputs, list):
                        for inp in inputs:
                            if inp.get('name') == 'prompt':
                                p = inp.get('value')
                                break
                                
                if p and isinstance(p, str) and len(p.strip()) > 0:
                    prompts.append(p.strip())
                        
        except json.JSONDecodeError:
            print("Error: Response is not valid JSON.")
        except Exception as e:
            print(f"Error parsing JSON: {e}")

        return prompts

    def run(self, pages=5):
        print(f"Starting crawl for Byteartist (Internal/POST)... Target pages: {pages}")
        
        if "这里粘贴" in COOKIE_STRING:
            print("ERROR: Please update the COOKIE_STRING in byteartist_crawler.py first!")
            return

        all_prompts = set()
        
        # 真实的 API 地址
        target_url = "https://byteartist-api.bytedance.net/api/inference/artworks/queries"
        
        for page in range(1, pages + 1):
            print(f"Fetching page {page}...")
            
            # POST 请求体 (Payload) - 基于真实抓包数据
            # 注意：真实 API 的 page_index 是从 0 开始的
            payload = {
                "ba_version": 2,
                "page_index": page - 1,  # 循环是从1开始的，所以减1
                "page_size": 20,         # 我们可以尝试改大一点，比如20
                "inference_types": ["t2i", "i2i"],
                "status": "completed",
                "is_publish": True,      # 注意 Python 里是 True, JSON 里是 true
                "sence": "smart_image"
            }
            
            try:
                # 注意：这里改为 session.post
                response = self.session.post(target_url, json=payload, timeout=15)
                
                # 检查状态码，如果是 403/401 说明 Cookie 失效或被封
                if response.status_code != 200:
                    print(f"  Error: Status Code {response.status_code}")
                    # print(f"  Body: {response.text[:100]}") # 调试用
                    # 遇到 403 直接退出，不要死磕
                    if response.status_code in [401, 403]:
                        print("  Cookie expired or Access Denied. Stopping.")
                        break
                
                # 调试用：打印前200字符
                if page == 1:
                    print(f"DEBUG Response: {response.text[:200]}...")

                page_prompts = self.parse(response)
                if page_prompts:
                    print(f"  Found {len(page_prompts)} prompts.")
                    all_prompts.update(page_prompts)
                else:
                    print("  No prompts found on this page. (Maybe end of list?)")
                    # 如果连续 3 页都没数据，就可以提前退出了，省时间
                    # 这里简单处理：如果一页一条都没抓到，且状态码是200，可能就是真没数据了
                    if page > 10: # 前几页可能不稳定，10页以后如果空了就停
                         print("  Empty page detected. Stopping crawl.")
                         break
                    
            except Exception as e:
                print(f"  Request failed: {e}")
            
            time.sleep(1)

        # 保存结果 (使用 JSONL 格式，方便后续处理和分割)
        output_path = os.path.join(self.output_dir, "byteartist_prompts.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for p in all_prompts:
                # 构造一个简单的字典对象
                record = {"source": "byteartist", "prompt": p}
                # 写入一行 JSON
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print(f"Done! Saved {len(all_prompts)} unique prompts to {output_path}")

if __name__ == "__main__":
    crawler = ByteartistCrawler()
    crawler.run(pages=500)
