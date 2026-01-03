"""
小红书爬虫 - Selenium版本
用于处理JavaScript渲染的动态页面

⚠️ 警告：本程序仅供学习研究使用
请遵守相关法律法规和网站使用条款
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import json
import logging
from typing import Dict, List, Optional
import os
from config import REQUEST_CONFIG, DATA_CONFIG, CRAWL_CONFIG
from utils import random_delay, save_to_json, save_to_csv, clean_text

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spider_selenium.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class XiaohongshuSpiderSelenium:
    """小红书爬虫类 - Selenium版本"""
    
    def __init__(self, headless: bool = False):
        """
        初始化爬虫
        
        Args:
            headless: 是否使用无头模式（不显示浏览器窗口）
        """
        self.driver = None
        self.headless = headless
        self.setup_driver()
    
    def setup_driver(self):
        """配置Chrome浏览器驱动"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # 反检测设置
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # 设置User-Agent
        user_agent = REQUEST_CONFIG.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        chrome_options.add_argument(f'user-agent={user_agent}')
        
        # 其他设置
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            # 执行反检测脚本
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                '''
            })
            logger.info("Chrome浏览器驱动初始化成功")
        except Exception as e:
            logger.error(f"Chrome浏览器驱动初始化失败: {e}")
            logger.error("请确保已安装Chrome浏览器和ChromeDriver")
            raise
    
    def search_notes(self, keyword: str, max_notes: int = 10) -> List[Dict]:
        """
        搜索笔记
        
        Args:
            keyword: 搜索关键词
            max_notes: 最大获取笔记数量
            
        Returns:
            笔记列表
        """
        notes = []
        
        try:
            # 构建搜索URL
            search_url = f"https://www.xiaohongshu.com/search_result?keyword={keyword}"
            logger.info(f"正在访问: {search_url}")
            
            self.driver.get(search_url)
            
            # 等待页面加载
            time.sleep(3)
            
            # 滚动页面加载更多内容
            scroll_pause_time = 2
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while len(notes) < max_notes:
                # 滚动到底部
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
                
                # 计算新的滚动高度
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                
                # 解析当前页面的笔记
                current_notes = self.parse_notes_from_page()
                for note in current_notes:
                    if note not in notes and len(notes) < max_notes:
                        notes.append(note)
                
                # 如果页面没有继续加载，退出循环
                if new_height == last_height:
                    logger.info("已到达页面底部")
                    break
                
                last_height = new_height
                
                if len(notes) >= max_notes:
                    break
            
            logger.info(f"成功获取 {len(notes)} 条笔记")
            return notes[:max_notes]
            
        except Exception as e:
            logger.error(f"搜索笔记失败: {e}")
            return notes
    
    def parse_notes_from_page(self) -> List[Dict]:
        """
        从当前页面解析笔记
        
        Returns:
            笔记列表
        """
        notes = []
        
        try:
            # 等待笔记元素加载
            # 注意：这些选择器需要根据小红书实际页面结构调整
            wait = WebDriverWait(self.driver, 10)
            
            # 查找笔记卡片（需要根据实际页面结构调整选择器）
            # 这里使用通用的选择器，实际使用时需要调整
            note_elements = self.driver.find_elements(By.CSS_SELECTOR, ".note-item, .feed-item, [class*='note']")
            
            for element in note_elements:
                try:
                    note_data = self.parse_note_element(element)
                    if note_data:
                        notes.append(note_data)
                except Exception as e:
                    logger.debug(f"解析单个笔记失败: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"解析页面笔记失败: {e}")
        
        return notes
    
    def parse_note_element(self, element) -> Optional[Dict]:
        """
        解析单个笔记元素
        
        Args:
            element: Selenium WebElement
            
        Returns:
            笔记数据字典
        """
        try:
            note_data = {
                'title': '',
                'author': '',
                'content': '',
                'likes': 0,
                'comments': 0,
                'url': '',
                'images': [],
            }
            
            # 尝试获取标题
            try:
                title_elem = element.find_element(By.CSS_SELECTOR, ".title, [class*='title']")
                note_data['title'] = clean_text(title_elem.text)
            except:
                pass
            
            # 尝试获取作者
            try:
                author_elem = element.find_element(By.CSS_SELECTOR, ".author, [class*='author'], [class*='user']")
                note_data['author'] = clean_text(author_elem.text)
            except:
                pass
            
            # 尝试获取内容
            try:
                content_elem = element.find_element(By.CSS_SELECTOR, ".content, [class*='content'], .desc")
                note_data['content'] = clean_text(content_elem.text)
            except:
                pass
            
            # 尝试获取链接
            try:
                link_elem = element.find_element(By.CSS_SELECTOR, "a")
                note_data['url'] = link_elem.get_attribute('href')
            except:
                pass
            
            # 尝试获取图片
            try:
                img_elements = element.find_elements(By.CSS_SELECTOR, "img")
                note_data['images'] = [img.get_attribute('src') for img in img_elements if img.get_attribute('src')]
            except:
                pass
            
            return note_data if note_data.get('title') or note_data.get('content') else None
            
        except Exception as e:
            logger.debug(f"解析笔记元素失败: {e}")
            return None
    
    def crawl(self):
        """执行爬取任务"""
        all_notes = []
        
        try:
            for keyword in CRAWL_CONFIG.get('keywords', []):
                logger.info(f"开始爬取关键词: {keyword}")
                
                max_notes = CRAWL_CONFIG.get('max_notes_per_keyword', 10)
                notes = self.search_notes(keyword, max_notes)
                all_notes.extend(notes)
                
                # 随机延迟
                random_delay(REQUEST_CONFIG.get('delay', 3), REQUEST_CONFIG.get('delay', 3) + 2)
            
            # 保存数据
            if all_notes:
                output_dir = DATA_CONFIG.get('output_dir', 'data')
                os.makedirs(output_dir, exist_ok=True)
                
                filename = f"{output_dir}/xiaohongshu_notes_selenium_{int(time.time())}"
                format_type = DATA_CONFIG.get('format', 'json')
                
                if format_type == 'json':
                    save_to_json(all_notes, f"{filename}.json")
                elif format_type == 'csv':
                    save_to_csv(all_notes, f"{filename}.csv")
                
                logger.info(f"成功保存 {len(all_notes)} 条笔记到 {filename}.{format_type}")
            else:
                logger.warning("未获取到任何笔记数据")
                
        finally:
            self.close()
    
    def close(self):
        """关闭浏览器"""
        if self.driver:
            self.driver.quit()
            logger.info("浏览器已关闭")


def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("小红书爬虫启动 (Selenium版本)")
    logger.info("=" * 50)
    
    # 使用无头模式（不显示浏览器窗口）
    # 如果需要看到浏览器操作过程，可以设置为False
    spider = XiaohongshuSpiderSelenium(headless=False)
    
    try:
        spider.crawl()
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        spider.close()
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        spider.close()
    
    logger.info("=" * 50)
    logger.info("爬虫任务完成")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

