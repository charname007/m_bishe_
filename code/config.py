"""
配置文件
请根据实际情况修改以下配置
"""

# 请求配置
REQUEST_CONFIG = {
    # 请求间隔（秒），建议设置2-5秒，避免请求过快
    "delay": 3,
    
    # 请求超时时间（秒）
    "timeout": 10,
    
    # 最大重试次数
    "max_retries": 3,
    
    # User-Agent，建议使用真实的浏览器User-Agent
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# Cookie配置
# 重要：需要从浏览器中获取真实的Cookie
# 方法：打开小红书网页 -> F12开发者工具 -> Network -> 找到任意请求 -> Headers -> Cookie
COOKIE = ""

# 代理配置（可选）
PROXY_CONFIG = {
    "http": "",  # 例如: "http://127.0.0.1:7890"
    "https": "",  # 例如: "https://127.0.0.1:7890"
}

# 数据存储配置
DATA_CONFIG = {
    # 数据保存路径
    "output_dir": "data",
    
    # 数据格式：json 或 csv
    "format": "json",
}

# 爬取配置
CRAWL_CONFIG = {
    # 要爬取的关键词
    "keywords": ["美食", "旅行"],
    
    # 每个关键词爬取的笔记数量
    "max_notes_per_keyword": 10,
}

