# 微博文本预处理规则文档

## 概述

微博数据预处理管道针对知识图谱三元组抽取场景设计，核心目标是：
- 保留有价值的实体信息
- 去除干扰噪音
- 防止大模型提取时产生幻觉

---

## 预处理流程（13步）

### 1. 转发链拆分 `split_forwards()`
**优先级：最高**

微博转发文本格式：`当前发言 // @用户A: 历史发言 // @用户B: 更早发言`

处理逻辑：
- 使用正则 `//\s*@([^:：]+)[:：]\s*` 切分文本
- 分离当前用户发言与历史转发者言论
- 三元组提取时仅使用 `author_text`，避免张冠李戴

输出结构：
```python
{
    'author_text': '当前用户发言',
    'forward_chain': [{'user': '数码大V', 'text': '确实，性能拉满'}, ...]
}
```

---

### 2. 全半角统一 `normalize_width()`
将全角字符转为半角：
- 全角数字：`０-９` → `0-9`
- 全角字母：`Ａ-Ｚ`、`ａ-ｚ` → `A-Z`、`a-z`
- 全角标点：`，。！？` → `,.!?`
- 全角空格 → 半角空格

示例：`１.５万块钱` → `1.5万块钱`

---

### 3. 繁简转换 `convert_to_simplified()`
使用 OpenCC 进行繁体转简体。

示例：`台北程式員` → `台北程序员`

---

### 4. 清理HTML实体 `clean_html_entities()`
- HTML实体替换：`&amp;` → `&`、`&lt;` → `<`
- 移除Unicode控制字符（保留 `\n` `\t`）

---

### 5. 微博系统噪音清洗 `clean_weibo_noise()`
移除微博自动生成的无意义文本：
- 分享占位符：`分享图片`、`分享视频`、`微博视频号`
- 链接占位符：`O网页链接`、`O绿洲`
- 长文本标记：`...全文`、`展开全文c`

---

### 6. 话题标签提取 `extract_topics()`
匹配格式：`#xxx#` 或 `#xxx[话题]#`

处理策略：
- 提取话题内容存入 `topics` 列表
- **保留文字、删除井号**（句子结构不断裂）

示例：`今天买了#iPhone15#，真香` → `今天买了iPhone15，真香`

---

### 7. @用户提取 `extract_mentions()`
匹配格式：`@用户名`（支持中文、英文、数字、下划线、减号）

处理策略：
- 提取用户名存入 `mentions` 列表
- **保留用户名、删除@符号**

示例：`感谢 @小明 的帮助` → `感谢 小明 的帮助`

---

### 8. 地理位置提取 `extract_locations()`
匹配格式：`2城市·地点`（微博定位打卡格式）

处理策略：
- 提取位置信息存入 `locations` 列表
- 格式：`{"city": "北京", "place": "三里屯"}`
- 从文本中移除定位标记

---

### 9. 微博表情移除 `remove_weibo_emoji()`
匹配格式：`[doge]`、`[允悲]`、`[吃瓜]` 等

处理策略：直接移除

**高级建议**：表情具有情感反转效果（如 `[doge]` 表示反讽），可维护情感字典做映射。

---

### 10. 金额提取 `extract_amounts()`
匹配格式：`\d+(?:\.\d+)?\s*(万|w|千|k|亿|块|元|块钱|美元|¥|$|￥)`

单位转换系数：
| 单位 | 系数 |
|------|------|
| 万/w | 10000 |
| 千/k | 1000 |
| 亿 | 100000000 |
| 块/元/¥/$ | 1 |

输出格式：
```python
{'original': '1.5万', 'value': 1.5, 'unit': '万', 'normalized': 15000.0}
```

---

### 11. 微博短链移除 `remove_weibo_urls()`
匹配微博短链格式：
- `http://t.cn/xxxx`
- `https://weibo.com/xxxx`
- `https://weibo.cn/sinaurl?u=...`

同时移除链接占位符 `O网页链接`

---

### 12. 空白字符统一 `normalize_whitespace()`
- 多空格 → 单空格
- 多换行 → 最多2换行
- 清理行首行尾空格

---

### 13. 短文本过滤判断 `should_filter_short_text()`
**三层过滤策略**：

#### 第一层：清洗后长度阈值
- 清洗后文本长度 < 5 字符
- 计算时机：在所有清洗步骤完成后

#### 第二层：无意义黑名单
高频无意义文本黑名单：
| 类型 | 黑名单词 |
|------|----------|
| 默认转发词 | `转发微博`、`repost`、`转` |
| 打卡分享词 | `分享图片`、`分享视频`、`签到`、`打卡` |
| 纯情绪词 | `马`、`马克`、`mark`、`m`、`码住` |
| 重复情绪 | `哈哈+`、`绝了`、`好美` |

#### 第三层：有实体则保留
即使短文本，如果有以下显式实体则保留：
- `mentions`：@用户列表
- `topics`：话题列表
- `amounts`：金额列表
- `locations`：位置列表

---

## 输出结构

```python
{
    'cleaned_text': str,       # 清洗后的完整文本
    'author_text': str,        # 当前用户发言（去除转发链）
    'forward_chain': list,     # 转发链条 [{'user': str, 'text': str}]
    'topics': list,            # 话题列表
    'mentions': list,          # @用户列表
    'amounts': list,           # 金额列表 [{'original', 'value', 'unit', 'normalized'}]
    'locations': list,         # 位置列表 [{'city', 'place'}]
    'original': str,           # 原始文本
    'should_filter': bool,     # 是否应该过滤
}
```

---

## 数据库表结构

新增字段（相比原通用表）：
| 字段 | 类型 | 说明 |
|------|------|------|
| `mentions` | JSONB | @用户列表 |
| `locations` | JSONB | 位置打卡列表 |
| `forward_chain` | JSONB | 转发链条 |

索引：
- `idx_mentions`：GIN索引，支持JSONB数组查询

---

## 实体判断逻辑

### 显式实体（预处理阶段可提取）
有格式标记的实体：
- @用户 → `mentions`
- #话题# → `topics`
- 金额 → `amounts`
- 定位打卡 → `locations`

### 隐式实体（需大模型识别）
无格式标记的实体：
- 无@的人名：`张三说...`
- 无#的产品名：`买了iPhone`
- 无定位的地名：`去了故宫`

**当前策略**：短文本过滤仅依赖显式实体判断，隐式实体需大模型处理。

---

## 使用示例

```python
from utils.text_preprocessor import WeiboTextPreprocessor

preprocessor = WeiboTextPreprocessor(convert_to_simplified=True)
result = preprocessor.preprocess("这手机太好用了！ //@数码大V: 确实，性能拉满")

print(result['author_text'])      # "这手机太好用了！"
print(result['forward_chain'])    # [{'user': '数码大V', 'text': '确实，性能拉满'}]
print(result['should_filter'])    # False
```

---

## 注意事项

1. **转发链拆分优先级最高**：直接影响三元组主体归属准确性
2. **话题处理保留文字**：删除井号但保留内容，确保句子结构完整
3. **地理位置保留**：可作为位置实体用于三元组提取
4. **短文本过滤时机**：在所有清洗完成后判断，避免误过滤