"""
小红书爬虫主程序

⚠️ 警告：本程序仅供学习研究使用
请遵守相关法律法规和网站使用条款
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
from typing import Dict, List, Optional
import logging
from config import REQUEST_CONFIG, COOKIE, PROXY_CONFIG, DATA_CONFIG, CRAWL_CONFIG
from utils import get_random_user_agent, random_delay, save_to_json, save_to_csv, clean_text

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spider.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class XiaohongshuSpider:
    """小红书爬虫类"""
    
    def __init__(self):
        self.session = requests.Session()
        self.setup_session()
        
    def setup_session(self):
        """配置会话"""
        # 设置请求头
        headers = {
            'User-Agent': REQUEST_CONFIG.get('user_agent', get_random_user_agent()),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        
        # 如果有Cookie，添加到请求头
        if COOKIE:
            headers['Cookie'] = COOKIE
        
        self.session.headers.update(headers)
        
        # 设置代理
        if PROXY_CONFIG.get('http') or PROXY_CONFIG.get('https'):
            self.session.proxies = {
                'http': PROXY_CONFIG.get('http'),
                'https': PROXY_CONFIG.get('https'),
            }
    
    def search_notes(self, keyword: str, page: int = 1) -> Optional[List[Dict]]:
        """
        搜索笔记
        
        Args:
            keyword: 搜索关键词
            page: 页码
            
        Returns:
            笔记列表
        """
        try:
            # 小红书搜索API（注意：这是示例URL，实际URL可能需要从网页中获取）
            # 小红书可能使用动态API，需要分析网页请求
            search_url = f"https://www.xiaohongshu.com/search_result?keyword={keyword}&page={page}"
            
            logger.info(f"正在搜索关键词: {keyword}, 页码: {page}")
            
            response = self.session.get(
                search_url,
                timeout=REQUEST_CONFIG.get('timeout', 10)
            )
            
            if response.status_code != 200:
                logger.warning(f"请求失败，状态码: {response.status_code}")
                return None
            
            # 解析HTML
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 注意：小红书的页面结构可能经常变化
            # 需要根据实际情况调整选择器
            notes = []
            
            # 示例：查找笔记卡片（需要根据实际HTML结构调整）
            note_cards = soup.find_all('div', class_='note-item')  # 这只是示例，实际class可能不同
            
            for card in note_cards:
                try:
                    note_data = self.parse_note_card(card)
                    if note_data:
                        notes.append(note_data)
                except Exception as e:
                    logger.error(f"解析笔记卡片失败: {e}")
                    continue
            
            # 如果页面是动态加载的（使用JavaScript），可能需要使用Selenium
            # 或者分析网络请求，找到真实的API接口
            
            logger.info(f"成功获取 {len(notes)} 条笔记")
            return notes
            
        except Exception as e:
            logger.error(f"搜索笔记失败: {e}")
            return None
    
    def parse_note_card(self, card) -> Optional[Dict]:
        """
        解析笔记卡片
        
        Args:
            card: BeautifulSoup元素
            
        Returns:
            笔记数据字典
        """
        try:
            # 这里需要根据实际的HTML结构来解析
            # 以下是示例代码，需要根据实际情况调整
            
            note_data = {
                'title': '',
                'author': '',
                'content': '',
                'likes': 0,
                'comments': 0,
                'url': '',
                'images': [],
            }
            
            # 示例解析（需要根据实际HTML结构调整）
            title_elem = card.find('div', class_='title')
            if title_elem:
                note_data['title'] = clean_text(title_elem.get_text())
            
            author_elem = card.find('div', class_='author')
            if author_elem:
                note_data['author'] = clean_text(author_elem.get_text())
            
            # ... 其他字段的解析
            
            return note_data
            
        except Exception as e:
            logger.error(f"解析笔记卡片失败: {e}")
            return None
    
    def crawl(self):
        """执行爬取任务"""
        all_notes = []
        
        for keyword in CRAWL_CONFIG.get('keywords', []):
            logger.info(f"开始爬取关键词: {keyword}")
            
            max_notes = CRAWL_CONFIG.get('max_notes_per_keyword', 10)
            page = 1
            notes_count = 0
            
            while notes_count < max_notes:
                notes = self.search_notes(keyword, page)
                
                if not notes:
                    logger.warning(f"未获取到笔记，可能已到达最后一页")
                    break
                
                all_notes.extend(notes)
                notes_count += len(notes)
                
                # 随机延迟，避免请求过快
                random_delay(
                    REQUEST_CONFIG.get('delay', 3) - 1,
                    REQUEST_CONFIG.get('delay', 3) + 2
                )
                
                page += 1
                
                if notes_count >= max_notes:
                    break
        
        # 保存数据
        if all_notes:
            output_dir = DATA_CONFIG.get('output_dir', 'data')
            os.makedirs(output_dir, exist_ok=True)
            
            filename = f"{output_dir}/xiaohongshu_notes_{int(time.time())}"
            format_type = DATA_CONFIG.get('format', 'json')
            
            if format_type == 'json':
                save_to_json(all_notes, f"{filename}.json")
            elif format_type == 'csv':
                save_to_csv(all_notes, f"{filename}.csv")
            
            logger.info(f"成功保存 {len(all_notes)} 条笔记到 {filename}.{format_type}")
        else:
            logger.warning("未获取到任何笔记数据")


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("小红书爬虫启动")
    logger.info("=" * 50)
    
    # 检查配置
    if not COOKIE:
        logger.warning("⚠️  未配置Cookie，可能会被限制访问")
        logger.warning("请在config.py中配置Cookie")
    
    spider = XiaohongshuSpider()
    spider.crawl()
    
    logger.info("=" * 50)
    logger.info("爬虫任务完成")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

