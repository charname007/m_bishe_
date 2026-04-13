"""
社交媒体文本预处理模块
用于知识图谱实体提取
"""

import re
import unicodedata
from typing import List, Tuple, Optional
try:
    import opencc
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False
    print("警告: opencc未安装，繁简转换将跳过。安装: pip install opencc")


class SocialMediaTextPreprocessor:
    """小红书文本预处理器"""

    def __init__(self, convert_to_simplified: bool = True):
        """
        Args:
            convert_to_simplified: 是否进行繁简转换
        """
        self._do_convert_simplified = convert_to_simplified and HAS_OPENCC
        if self._do_convert_simplified:
            self.cc = opencc.OpenCC('t2s')  # 繁体转简体

        # 小红书话题正则: #xxx[话题]# 或 #xxx#
        self.topic_pattern = re.compile(r'#([^#\[\]]+)(?:\[话题\])?#')

        # 小红书表情正则: [xxxR] 如 [大笑R] [哭惹R]
        self.emoji_pattern = re.compile(r'\[[\u4e00-\u9fa5a-zA-Z]+R\]')

        # 金额正则: 匹配各种金额表达
        self.amount_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(万|w|千|k|亿|块|元|块钱|美元|美金|¥|$|￥)',
            re.IGNORECASE
        )

    def normalize_width(self, text: str) -> str:
        """
        全角转半角
        处理: 数字、字母、标点、空格、括号等
        """
        if not text:
            return text

        result = []
        for char in text:
            # 全角转半角的核心逻辑
            code = ord(char)

            # 全角数字 ０-９ (FF10-FF19) → 半角 0-9 (30-39)
            if 0xFF10 <= code <= 0xFF19:
                result.append(chr(code - 0xFF10 + 0x30))

            # 全角大写字母 Ａ-Ｚ (FF21-FF3A) → 半角 A-Z (41-5A)
            elif 0xFF21 <= code <= 0xFF3A:
                result.append(chr(code - 0xFF21 + 0x41))

            # 全角小写字母 ａ-ｚ (FF41-FF5A) → 半角 a-z (61-7A)
            elif 0xFF41 <= code <= 0xFF5A:
                result.append(chr(code - 0xFF41 + 0x61))

            # 全角空格 (3000) → 半角空格 (20)
            elif code == 0x3000:
                result.append(' ')

            # 其他常见全角符号映射
            elif char in self._FULL_TO_HALF_MAP:
                result.append(self._FULL_TO_HALF_MAP[char])

            else:
                result.append(char)

        return ''.join(result)

    # 全角标点符号映射表
    _FULL_TO_HALF_MAP = {
        '，': ',',    '。': '.',    '！': '!',    '？': '?',
        '；': ';',    '：': ':',    '「': '"',    '」': '"',
        '『': "'",    '』': "'",    '（': '(' ,   '）': ')',
        '【': '[',    '】': ']',    '〈': '<',    '〉': '>',
        '《': '<',    '》': '>',    '﹑': ',',    '﹐': ',',
        '﹒': '.',    '﹔': ';',    '﹕': ':',    '﹖': '?',
        '﹗': '!',    '－': '-',    '——': '--',
    }

    def convert_to_simplified(self, text: str) -> str:
        """繁体转简体"""
        if not text or not self._do_convert_simplified:
            return text
        return self.cc.convert(text)

    def extract_topics(self, text: str) -> Tuple[str, List[str]]:
        """
        提取并清洗话题标签
        返回: (清洗后的文本, 话题列表)
        """
        if not text:
            return text, []

        topics = []
        # 替换话题标记为纯文本标签（便于后续处理）
        def replace_topic(match):
            topic = match.group(1).strip()
            topics.append(topic)
            return f' {topic} '  # 保留标签内容，移除#号

        cleaned_text = self.topic_pattern.sub(replace_topic, text)
        return cleaned_text.strip(), topics

    def remove_xhs_emoji(self, text: str) -> str:
        """
        移除小红书表情标记 [xxxR]
        可选: 替换为空或替换为描述文本
        """
        if not text:
            return text
        # 直接移除表情标记
        return self.emoji_pattern.sub(' ', text).strip()

    def extract_amounts(self, text: str) -> Tuple[str, List[dict]]:
        """
        提取金额信息
        返回: (原文本, 金额列表)
        金额格式: {"value": float, "unit": str, "normalized": float}
        """
        if not text:
            return text, []

        amounts = []
        # 单位转换系数
        unit_map = {
            '万': 10000, 'w': 10000, 'W': 10000,
            '千': 1000, 'k': 1000, 'K': 1000,
            '亿': 100000000,
            '块': 1, '元': 1, '块钱': 1,
            '美元': 1, '美金': 1,
            '¥': 1, '￥': 1, '$': 1,
        }

        def convert_amount(match):
            num = float(match.group(1))
            unit = match.group(2)
            coefficient = unit_map.get(unit.lower() or unit, 1)
            normalized = num * coefficient
            amounts.append({
                'original': match.group(0),
                'value': num,
                'unit': unit,
                'normalized': normalized
            })
            return match.group(0)  # 保留原文

        self.amount_pattern.sub(convert_amount, text)
        return text, amounts

    def clean_html_entities(self, text: str) -> str:
        """清理HTML实体和特殊字符"""
        if not text:
            return text

        # 常见HTML实体替换
        html_entities = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>',
            '&quot;': '"', '&apos;': "'", '&nbsp;': ' ',
            '&#39;': "'", '&#34;': '"',
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)

        # 清理Unicode控制字符
        text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c in '\n\t')

        return text

    def remove_urls(self, text: str) -> str:
        """移除URL链接"""
        if not text:
            return text

        # 匹配常见URL模式
        url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+|'
            r'www\.[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE
        )
        return url_pattern.sub('', text)

    def normalize_whitespace(self, text: str) -> str:
        """统一空白字符"""
        if not text:
            return text

        # 多个空格变单个
        text = re.sub(r'[^\S\n]+', ' ', text)
        # 多个换行变最多2个
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行首行尾空格
        text = '\n'.join(line.strip() for line in text.split('\n'))

        return text.strip()

    def preprocess(self, text: str) -> dict:
        """
        完整预处理流程
        返回结构化结果，便于知识图谱构建
        """
        if not text:
            return {
                'cleaned_text': '',
                'topics': [],
                'amounts': [],
                'original': ''
            }

        original = text

        # 1. 全半角统一
        text = self.normalize_width(text)

        # 2. 繁简转换
        text = self.convert_to_simplified(text)

        # 3. 清理HTML实体
        text = self.clean_html_entities(text)

        # 4. 提取话题标签
        text, topics = self.extract_topics(text)

        # 5. 移除小红书表情标记
        text = self.remove_xhs_emoji(text)

        # 6. 提取金额信息
        text, amounts = self.extract_amounts(text)

        # 7. 移除URL
        text = self.remove_urls(text)

        # 8. 统一空白字符
        text = self.normalize_whitespace(text)

        return {
            'cleaned_text': text,
            'topics': topics,
            'amounts': amounts,
            'original': original
        }


def preprocess_tags(tag_string: str) -> List[str]:
    """
    预处理标签字符串
    输入: "副业,自媒体,职场生活,AI,打工"
    输出: ['副业', '自媒体', '职场生活', 'AI', '打工']
    """
    if not tag_string:
        return []

    # 去除话题标记后缀
    tags = re.sub(r'\[话题\]', '', tag_string)

    # 分割并清理
    tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]

    return tag_list


def preprocess_amount_field(amount_str: str) -> Optional[float]:
    """
    预处理点赞数/收藏数等字段
    输入: "1.4万", "8127", "2.6万"
    输出: 14000, 8127, 26000
    """
    if not amount_str or amount_str.strip() == '':
        return None

    amount_str = amount_str.strip()

    # 纯数字
    try:
        return float(amount_str)
    except ValueError:
        pass

    # 中文数字单位
    unit_map = {'万': 10000, 'w': 10000, '亿': 100000000}
    for unit, coeff in unit_map.items():
        if unit in amount_str:
            try:
                num = float(amount_str.replace(unit, '').strip())
                return num * coeff
            except ValueError:
                continue

    return None


class WeiboTextPreprocessor(SocialMediaTextPreprocessor):
    """微博文本预处理器（继承通用预处理器）"""

    # 微博无意义文本黑名单
    MEANINGLESS_BLACKLIST = {
        '转发微博', 'repost', '转',
        '分享图片', '分享视频', '签到', '打卡',
        '马', '马克', 'mark', 'm', '码住',
        '绝了', '好美',
    }

    def __init__(self, convert_to_simplified: bool = True):
        super().__init__(convert_to_simplified)

        # 微博转发链正则: // @用户: 或 //@用户:
        self.forward_pattern = re.compile(r'//\s*@([^:：]+)[:：]\s*')

        # 微博表情正则: [doge] [允悲] [吃瓜] 等
        self.weibo_emoji_pattern = re.compile(r'\[[a-zA-Z\u4e00-\u9fa5]+\]')

        # 微博短链正则: t.cn、weibo.com、weibo.cn
        self.weibo_url_pattern = re.compile(
            r'(?:https?://)?(?:t\.cn|weibo\.com|weibo\.cn)[/\w]*',
            re.IGNORECASE
        )

        # 微博链接占位符: O网页链接、O绿洲
        self.link_placeholder_pattern = re.compile(r'O[^\s]+链接')

        # 微博噪音正则: 分享占位符、长文本标记
        self.noise_pattern = re.compile(
            r'(分享图片|分享视频|微博视频号|绿洲|'
            r'展开全文c|…全文|\.+全文)'
        )

        # 微博地理位置正则: 2北京·三里屯 (保留，用于提取)
        self.location_pattern = re.compile(r'2([\u4e00-\u9fa5]+)·([\u4e00-\u9fa5]+)')

        # @用户正则
        self.mention_pattern = re.compile(r'@([a-zA-Z0-9_\u4e00-\u9fa5\-]+)')

        # 重复情绪词正则: 哈哈哈+
        self.repeat_emoji_pattern = re.compile(r'(哈){2,}|(好){2,}')

    def split_forwards(self, text: str) -> dict:
        """
        拆分微博转发链条
        返回: {
            'author_text': '当前用户发言',
            'forward_chain': [{'user': 'xxx', 'text': 'xxx'}, ...]
        }
        """
        if not text:
            return {'author_text': '', 'forward_chain': []}

        # 按转发标记切分
        parts = self.forward_pattern.split(text)

        # parts[0] = 当前用户发言
        # 后续交替为 [用户名, 发言内容]
        author_text = parts[0].strip() if parts else ''
        forward_chain = []

        for i in range(1, len(parts) - 1, 2):
            user = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ''
            if user and content:
                forward_chain.append({'user': user, 'text': content})

        return {
            'author_text': author_text,
            'forward_chain': forward_chain
        }

    def extract_mentions(self, text: str) -> Tuple[str, List[str]]:
        """
        提取@用户
        返回: (清洗后的文本, 用户列表)
        清洗策略: 保留用户名，去掉@符号
        """
        if not text:
            return text, []

        mentions = []
        def replace_mention(match):
            user = match.group(1)
            mentions.append(user)
            return user  # 保留名字，去掉@

        cleaned = self.mention_pattern.sub(replace_mention, text)
        return cleaned, mentions

    def extract_locations(self, text: str) -> Tuple[str, List[dict]]:
        """
        提取地理位置打卡
        返回: (原文本, 位置列表)
        位置格式: {"city": "北京", "place": "三里屯"}
        """
        if not text:
            return text, []

        locations = []
        def extract_location(match):
            city = match.group(1)
            place = match.group(2)
            locations.append({'city': city, 'place': place})
            return ''  # 从文本中移除定位标记

        cleaned = self.location_pattern.sub(extract_location, text)
        return cleaned.strip(), locations

    def remove_weibo_emoji(self, text: str) -> str:
        """
        移除微博表情标记 [doge] [允悲]
        可选: 维护情感字典做反转标记
        """
        if not text:
            return text
        # 当前简单实现: 直接移除
        return self.weibo_emoji_pattern.sub('', text).strip()

    def remove_weibo_urls(self, text: str) -> str:
        """
        移除微博短链和链接占位符
        """
        if not text:
            return text

        # 移除短链
        text = self.weibo_url_pattern.sub('', text)
        # 移除链接占位符
        text = self.link_placeholder_pattern.sub('', text)
        return text.strip()

    def clean_weibo_noise(self, text: str) -> str:
        """
        清洗微博系统噪音
        """
        if not text:
            return text
        return self.noise_pattern.sub('', text).strip()

    def normalize_repeat_emoji(self, text: str) -> str:
        """
        处理重复情绪词: 哈哈哈 -> 哈, 好好好 -> 好
        用于黑名单匹配判断
        """
        if not text:
            return text
        return self.repeat_emoji_pattern.sub(lambda m: m.group(1) or m.group(2), text)

    def has_explicit_entities(self, result: dict) -> bool:
        """
        检查是否有预处理阶段提取的显式实体
        """
        return bool(
            result.get('mentions') or
            result.get('topics') or
            result.get('amounts') or
            result.get('locations')
        )

    def is_meaningless_text(self, text: str) -> bool:
        """
        判断是否为无意义文本（黑名单匹配）
        """
        if not text:
            return True

        # 处理重复情绪词后判断
        normalized = self.normalize_repeat_emoji(text)
        normalized_lower = normalized.lower().strip()

        # 黑名单完全匹配
        if normalized_lower in self.MEANINGLESS_BLACKLIST:
            return True
        if text.strip() in self.MEANINGLESS_BLACKLIST:
            return True

        # 单字情绪词: 哈、好 等
        if len(normalized_lower) <= 2 and normalized_lower in {'哈', '好', '嗯', '哦', '行'}:
            return True

        return False

    def should_filter_short_text(self, result: dict) -> bool:
        """
        三层短文本过滤判断
        返回 True 表示应该过滤
        """
        cleaned = result.get('cleaned_text', '').strip()

        # 第一层: 清洗后长度 < 5
        if len(cleaned) < 5:
            # 第三层: 有显式实体则保留
            if self.has_explicit_entities(result):
                return False
            return True

        # 第二层: 黑名单匹配
        if self.is_meaningless_text(cleaned):
            return True

        return False

    def preprocess(self, text: str) -> dict:
        """
        微博完整预处理流程
        """
        if not text:
            return {
                'cleaned_text': '',
                'author_text': '',
                'forward_chain': [],
                'topics': [],
                'mentions': [],
                'amounts': [],
                'locations': [],
                'original': '',
                'should_filter': True,
            }

        original = text

        # 1. 先拆分转发链（在清洗前，保留原始结构）
        forward_result = self.split_forwards(text)
        author_text = forward_result['author_text']

        # 只对当前用户发言做后续清洗
        text = author_text

        # 2. 全半角统一
        text = self.normalize_width(text)

        # 3. 繁简转换
        text = self.convert_to_simplified(text)

        # 4. 清理HTML实体
        text = self.clean_html_entities(text)

        # 5. 清洗微博系统噪音
        text = self.clean_weibo_noise(text)

        # 6. 提取并清洗话题标签
        text, topics = self.extract_topics(text)

        # 7. 提取@用户
        text, mentions = self.extract_mentions(text)

        # 8. 提取地理位置
        text, locations = self.extract_locations(text)

        # 9. 移除微博表情
        text = self.remove_weibo_emoji(text)

        # 10. 提取金额信息
        text, amounts = self.extract_amounts(text)

        # 11. 移除微博短链
        text = self.remove_weibo_urls(text)

        # 12. 统一空白字符
        text = self.normalize_whitespace(text)

        result = {
            'cleaned_text': text,
            'author_text': text,  # 清洗后的当前用户发言
            'forward_chain': forward_result['forward_chain'],
            'topics': topics,
            'mentions': mentions,
            'amounts': amounts,
            'locations': locations,
            'original': original,
        }

        # 13. 判断是否过滤
        result['should_filter'] = self.should_filter_short_text(result)

        return result


# 使用示例
if __name__ == '__main__':
    # 测试通用预处理器
    preprocessor = SocialMediaTextPreprocessor()

    test_cases = [
        "#副业[话题]# [大笑R] 今天赚了１.５万块钱！",
        "台北程式員接單經歷 #編程[话题]# 收入３０００w",
        "普通人在家可做的６个工作💰 #赚钱[话题]#",
        "终于！靠写代码赚上钱了！，收到４０块",
    ]

    print("=" * 60)
    print("通用预处理测试结果:")
    print("=" * 60)

    for text in test_cases:
        result = preprocessor.preprocess(text)
        print(f"\n原文: {result['original']}")
        print(f"清洗: {result['cleaned_text']}")
        print(f"话题: {result['topics']}")
        print(f"金额: {result['amounts']}")

    # 测试微博预处理器
    weibo_processor = WeiboTextPreprocessor()

    weibo_test_cases = [
        "这手机太好用了！ //@数码大V: 确实，性能拉满 //@吃瓜群众: 看看评测视频 O网页链接",
        "#iPhone15# [doge] 今天买了iPhone15，真香",
        "转发微博 http://t.cn/abc123",
        "@中国移动 服务太差了，投诉！",
        "哈哈哈哈",
        "2北京·三里屯 今天在这吃了顿好的",
        "马",
        "分享图片",
    ]

    print("\n" + "=" * 60)
    print("微博预处理测试结果:")
    print("=" * 60)

    for text in weibo_test_cases:
        result = weibo_processor.preprocess(text)
        print(f"\n原文: {result['original']}")
        print(f"清洗: {result['cleaned_text']}")
        print(f"转发链: {result['forward_chain']}")
        print(f"话题: {result['topics']}")
        print(f"@用户: {result['mentions']}")
        print(f"位置: {result['locations']}")
        print(f"金额: {result['amounts']}")
        print(f"是否过滤: {result['should_filter']}")