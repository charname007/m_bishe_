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
        self.convert_to_simplified = convert_to_simplified and HAS_OPENCC
        if self.convert_to_simplified:
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
        if not text or not self.convert_to_simplified:
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


# 使用示例
if __name__ == '__main__':
    preprocessor = SocialMediaTextPreprocessor()

    # 测试样例
    test_cases = [
        "#副业[话题]# [大笑R] 今天赚了１.５万块钱！",
        "台北程式員接單經歷 #編程[话题]# 收入３０００w",
        "普通人在家可做的６个工作💰 #赚钱[话题]#",
        "终于！靠写代码赚上钱了！，收到４０块",
    ]

    print("=" * 60)
    print("预处理测试结果:")
    print("=" * 60)

    for text in test_cases:
        result = preprocessor.preprocess(text)
        print(f"\n原文: {result['original']}")
        print(f"清洗: {result['cleaned_text']}")
        print(f"话题: {result['topics']}")
        print(f"金额: {result['amounts']}")

    # 测试字段处理
    print("\n" + "=" * 60)
    print("字段预处理测试:")
    print("=" * 60)

    test_amounts = ["1.4万", "8127", "2.6万", "1560", None, ""]
    for amt in test_amounts:
        print(f"  '{amt}' → {preprocess_amount_field(amt)}")

    test_tags = "副业,自媒体,职场生活,AI,打工"
    print(f"\n  标签: '{test_tags}' → {preprocess_tags(test_tags)}")