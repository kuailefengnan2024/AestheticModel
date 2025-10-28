import os
import sys

# 将项目根目录添加到Python的模块搜索路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试从环境变量中获取API Key
api_key = os.environ.get("GEMINI_25_API_KEY")

print(f"Python 脚本正在检查环境变量 'GEMINI_25_API_KEY'...")

if api_key:
    # 为了安全，我们不直接打印key，只打印它的一部分
    print(f"成功找到！Key的开头是: {api_key[:4]}****")
else:
    print("失败：未能找到环境变量。值为: None")

# 额外检查：确认是否能找到config模块和里面的配置
try:
    from config.settings import LLM_API_CONFIGS
    gemini_config = LLM_API_CONFIGS.get("gemini_2_5_pro")
    if gemini_config:
        print("成功导入 config/settings.py 并找到了 gemini_2_5_pro 的配置。")
        print(f"配置文件中的Key值是: {str(gemini_config.get('api_key'))[:4]}****")
    else:
        print("错误：虽然导入了settings.py，但没有找到gemini_2_5_pro的配置。")
except Exception as e:
    print(f"尝试导入 config/settings.py 时出错: {e}")
