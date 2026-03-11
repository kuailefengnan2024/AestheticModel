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
COOKIE_STRING = """
email=handongyang.729@bytedance.com; X-Risk-Browser-Id=d8e1cc40e16ed5912421ac49da0a271fa4e9fbf9da49273a6d203be3b69700e9; people-lang=zh; user_token=JTdCJTIybmFtZSUyMiUzQSUyMiVFOSU5RiVBOSVFNSU4NiVBQyVFOSU5OCVCMyUyMiUyQyUyMmZ1bGxfbmFtZSUyMiUzQSUyMiVFOSU5RiVBOSVFNSU4NiVBQyVFOSU5OCVCMyUyMDc3MTMwMTglMjIlMkMlMjJlbWFpbCUyMiUzQSUyMmhhbmRvbmd5YW5nLjcyOSU0MGJ5dGVkYW5jZS5jb20lMjIlMkMlMjJwaWN0dXJlJTIyJTNBJTIyaHR0cHMlM0ElMkYlMkZzMy1pbWZpbGUuZmVpc2h1Y2RuLmNvbSUyRnN0YXRpYy1yZXNvdXJjZSUyRnYxJTJGdjNfMDBvMF9hOTQyYjkzOS03Mjg3LTRiMzktOWEwMy04NjAyMDcyZGRkOWd+JTNGaW1hZ2Vfc2l6ZSUzRDI0MHgyNDAlMjZjdXRfdHlwZSUzRCUyNnF1YWxpdHklM0QlMjZmb3JtYXQlM0RwbmclMjZzdGlja2VyX2Zvcm1hdCUzRC53ZWJwJTIyJTJDJTIyZW1wbG95ZWVfaWQlMjIlM0ElMjI3NzEzMDE4JTIyJTJDJTIyZW1wbG95ZWVfbnVtYmVyJTIyJTNBJTIyNzcxMzAxOCUyMiUyQyUyMnRlbmFudF9hbGlhcyUyMiUzQSUyMmJ5dGVkYW5jZSUyMiUyQyUyMnVzZXJfaWQlMjIlM0ElMjJoZDFpaDBkNnB2bjh4czFpaGk3diUyMiU3RA==; monitor_huoshan_web_id=7432205508379984219; _ga_R1FN4KJKJH=GS2.1.s1765267007$o1$g0$t1765267007$j60$l0$h0; _ga=GA1.2.2048288279.1765267007; ba_space_id=e75d595ffcb142d29596bd8b7b472c44; bdsso_lt_c0=djEuMC4w.iRnhIH_BO8aPFJcTHTKOjNM71VflcqxEEqr_2tK3ILfhTzIDDE1Bpmo7qUK709SMFd0O0AOpc0U2thJsREp3ku7M8qwLUpZObYGG_j-TuLp9z8f8vLnfRky5luMXJoj-s8_3EdnR_qMLmou3UyYlFJbIOsoE-vKYFftF3E3KbSZTmUnaE88gP4Fie1FJo-gS9vTcPEUgRajM7ZmglqYgVOjz3YFQxD7VRsOlxIwcjLbS0QjmHtlDCyBOHu0nqzJI_06jkGGddCp5Jf6YWocIMWVoPrUMQZmeXg2nENv8Ht0.t4S5CrKNsMyjIjghN071mqbTxbU; bdsso_lt_c0_ss=djEuMC4w.iRnhIH_BO8aPFJcTHTKOjNM71VflcqxEEqr_2tK3ILfhTzIDDE1Bpmo7qUK709SMFd0O0AOpc0U2thJsREp3ku7M8qwLUpZObYGG_j-TuLp9z8f8vLnfRky5luMXJoj-s8_3EdnR_qMLmou3UyYlFJbIOsoE-vKYFftF3E3KbSZTmUnaE88gP4Fie1FJo-gS9vTcPEUgRajM7ZmglqYgVOjz3YFQxD7VRsOlxIwcjLbS0QjmHtlDCyBOHu0nqzJI_06jkGGddCp5Jf6YWocIMWVoPrUMQZmeXg2nENv8Ht0.t4S5CrKNsMyjIjghN071mqbTxbU; monitor_session_id_flag=1; __tea_cache_tokens_3569={%22web_id%22:%227615899441161274915%22%2C%22user_unique_id%22:%227615899441161274915%22%2C%22timestamp%22:1773216305024%2C%22_type_%22:%22default%22}; monitor_session_id=0547238262251018848; bd_sso_3b6da9=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzM4MzQwNTMsImlhdCI6MTc3MzIyOTI1MywiaXNzIjoic3NvLmJ5dGVkYW5jZS5jb20iLCJzdWIiOiJoZDFpaDBkNnB2bjh4czFpaGk3diIsInRlbmFudF9pZCI6ImhncTN0Y2NwM2kxc2pqbjU4emlrIn0.NV2EaazPWxWk_kOYt1sg_pmiB7OWfvuai-YfyOdS0xsz05qfHMN9PV-aXNzIat8ZJJc6JNMcJTCBLQfBd_76-Lu17nI5DLvSquo9iaLNtBV-WgOZV9C-VkqhpbbgYGIn7PYPCNOSYIQpb-BKF0B1aEDhfGzZC1Rm_PJbf_E2cjT7KXwFYp33t0nYPRbvxhU35agNcTZovExhw3BkIPt2ag5JLM0xBtVElBN_dlab0PdxS3u5n1ILDmEZ3PBRL1uB3ed6rFo2Zs0QbjwZbTCyLQTbjkZxZgKRDCKNr63NfszmgAJHIvu14uLGiXOxskRxQjMlgcpZvyNsWwBbuWGkwg
"""

JWT_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6IiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJwYWFzLnBhc3Nwb3J0LmF1dGgiLCJleHAiOjE3NzMyMzI4NTMsImlhdCI6MTc3MzIyOTE5MywidXNlcm5hbWUiOiJoYW5kb25neWFuZy43MjkiLCJ0eXBlIjoicGVyc29uX2FjY291bnQiLCJyZWdpb24iOiJjbiIsInRydXN0ZWQiOnRydWUsInV1aWQiOiJkZTg4MzI0NC1mYjEzLTQ4MzktYjliNy1jNzE2MmU1MTMwMzciLCJzaXRlIjoib25saW5lIiwiYnl0ZWNsb3VkX3RlbmFudF9pZCI6ImJ5dGVkYW5jZSIsImJ5dGVjbG91ZF90ZW5hbnRfaWRfb3JnIjoiYnl0ZWRhbmNlIiwic2NvcGUiOiJieXRlZGFuY2UiLCJzZXF1ZW5jZSI6IkRlc2lnbiIsIm9yZ2FuaXphdGlvbiI6IuaKlumfs-ebtOaSrS3orr7orqEt6L-Q6JClIiwid29ya19jb3VudHJ5IjoiQ0hOIiwibG9jYXRpb24iOiJDTiIsImF2YXRhcl91cmwiOiJodHRwczovL3MxLWltZmlsZS5mZWlzaHVjZG4uY29tL3N0YXRpYy1yZXNvdXJjZS92MS92M18wMG8wX2E5NDJiOTM5LTcyODctNGIzOS05YTAzLTg2MDIwNzJkZGQ5Z34_aW1hZ2Vfc2l6ZT1ub29wXHUwMDI2Y3V0X3R5cGU9XHUwMDI2cXVhbGl0eT1cdTAwMjZmb3JtYXQ9cG5nXHUwMDI2c3RpY2tlcl9mb3JtYXQ9LndlYnAiLCJlbWFpbCI6ImhhbmRvbmd5YW5nLjcyOUBieXRlZGFuY2UuY29tIiwiZW1wbG95ZWVfaWQiOjc3MTMwMTgsIm5ld19lbXBsb3llZV9pZCI6NzcxMzAxOH0.nR8KEyCE8xQ3slNttStuMMNHTNCZ8vdC6N7EmuS97Fk9DQxFK119NRbL6LLIp-UkY7a_C6vpFYyOwUbIOnwNpCxC2CmWemyOAFULVMhYJxVLbTsI5wqGpAVEYkEH1pju_X5Sfa8d0MTXfnFJc_Ctm51jLRuIcRJBeRI_xq3lyeI"

# =========================================

class ByteartistCrawler(BaseCrawler):
    def __init__(self, output_dir="data/raw_prompts"):
        super().__init__(base_url="https://byteartist-beta.bytedance.net")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 注入 Cookie 和 JWT Token 到 headers
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
            "Referer": "https://byteartist-beta.bytedance.net/",
            "Origin": "https://byteartist-beta.bytedance.net",
            "x-jwt-token": JWT_TOKEN
        })

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
            
            # POST 请求体 (Payload) - 更新为 cURL 中的最新结构
            # media_type: 0 (图片?), 1 (视频?), 2 (GIF?), 4 (?)
            # 经过观察，media_type=[0] 通常只包含图片
            payload = {
                "page_index": page - 1,
                "page_size": 20,
                "is_publish": True,
                "Scenes": [0],
                "ba_version": 2,
                "status": "completed",
                "keywords": "",
                "media_type": [0]  # 仅保留 0，过滤掉视频和其他格式
            }
            
            try:
                response = self.session.post(target_url, json=payload, timeout=15)
                
                # 检查状态码
                if response.status_code != 200:
                    print(f"  Error: Status Code {response.status_code}")
                    if response.status_code in [401, 403]:
                        print("  Cookie expired or Access Denied. Stopping.")
                        break
                
                if page == 1:
                    print(f"DEBUG Response: {response.text[:200]}...")

                page_prompts = self.parse(response)
                if page_prompts:
                    print(f"  Found {len(page_prompts)} prompts.")
                    all_prompts.update(page_prompts)
                else:
                    print("  No prompts found on this page.")
                    if page > 10: 
                         print("  Empty page detected. Stopping crawl.")
                         break
                    
            except Exception as e:
                print(f"  Request failed: {e}")
            
            time.sleep(1)

        # 保存结果
        output_path = os.path.join(self.output_dir, "byteartist_prompts.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for p in all_prompts:
                record = {"source": "byteartist", "prompt": p}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print(f"Done! Saved {len(all_prompts)} unique prompts to {output_path}")

if __name__ == "__main__":
    crawler = ByteartistCrawler()
    crawler.run(pages=500)
