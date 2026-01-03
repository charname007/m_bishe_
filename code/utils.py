"""
工具函数
"""

import time
import random
import json
import csv
import os
from typing import Dict, List, Any
from fake_useragent import UserAgent


def get_random_user_agent() -> str:
    """获取随机User-Agent"""
    try:
        ua = UserAgent()
        return ua.random
    except:
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def random_delay(min_seconds: float = 2, max_seconds: float = 5):
    """随机延迟，模拟人类行为"""
    delay = random.uniform(min_seconds, max_seconds)
    time.sleep(delay)


def save_to_json(data: List[Dict], filepath: str):
    """保存数据到JSON文件"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    # 如果文件已存在，读取现有数据
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        data = existing_data + data
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_to_csv(data: List[Dict], filepath: str):
    """保存数据到CSV文件"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    if not data:
        return
    
    # 获取所有字段名
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    fieldnames = sorted(list(fieldnames))
    
    # 判断文件是否存在
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


def clean_text(text: str) -> str:
    """清理文本内容"""
    if not text:
        return ""
    return text.strip().replace('\n', ' ').replace('\r', '')


def format_timestamp(timestamp: int) -> str:
    """格式化时间戳"""
    import datetime
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

