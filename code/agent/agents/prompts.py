"""
四步骤智能体工作流提示词模板 - 使用LangChain ChatPromptTemplate
P2改进：简化评估提示词，单次评估包含评分和修正
P5改进：添加 Filter 筛选提示词
P10改进：添加批量LLM调用提示词
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate


# ===== Step 0: Filter 筛选提示词模板（P5新增） =====

FILTER_SYSTEM = """你是一位"地理文本筛选专家"，负责快速判断文本是否值得处理。
你的任务是高效筛选，识别包含武汉地理信息的文本，跳过无价值文本和非武汉地区文本以节省处理成本。"""

FILTER_USER = """## 快速筛选标准

**有效文本（is_valid=true）**：
- 提及武汉地理实体：武汉的道路、POI、建筑物、街区、地名等（如珞喻路、武汉大学、街道口）
- 涉及空间关系：位于、旁边、附近、在...内、相邻等
- 地理相关活动：逛街、打卡、拍照、游玩等（暗示地点）
- 即使主语省略，但有地理暗示（如"这里的樱花很好看"）
- 无法确定是否武汉地区时，默认放行（保守策略）

**无效文本（is_valid=false）**：
- 过短文本：少于5个有效字符
- 无地理信息：纯情感表达、无关话题、纯表情/乱码
- 纯抽象内容：时间、数字、无地点的活动描述
- **非武汉地区**：明确提及武汉以外的城市/地区，且无任何武汉相关实体
  - 明确非武汉城市：北京、上海、广州、深圳、成都、杭州、南京、重庆、西安等
  - 明确非武汉景点：故宫、长城、西湖、外滩、兵马俑等
  - 注意：如果文本同时提及武汉和非武汉地区，应放行（is_valid=true）

## 武汉地区判断规则

**判定为武汉相关（is_non_wuhan_region=false）**：
- 明确提及"武汉"、"汉口"、"武昌"、"汉阳"、"光谷"等武汉核心区域名
- 提及武汉知名地标：武汉大学、黄鹤楼、东湖、江汉路、珞喻路等
- 提及武汉特有元素：樱花（武大樱花）、长江大桥（武汉段）

**判定为非武汉地区（is_non_wuhan_region=true）**：
- 明确提及其他城市名且无武汉相关内容
- 如"北京故宫很壮观"、"上海外滩夜景漂亮"

**无法确定时（保守策略）**：
- 如果地名无法确定城市归属（如"大学门口"、"商业街"），设 is_non_wuhan_region=false，放行
- region_hint 设为"未知"

## 边界模糊处理
- 如果判断困难，返回 confidence="low"，让后续流程处理
- 有地理暗示但无明确实体时，建议保留（is_valid=true, confidence="low")
- 地区归属存疑时，默认放行（保守策略）

## 任务示例

示例1:
输入: "武汉大学在珞喻路上，樱花开了很漂亮"
输出: {{
  "is_valid": true,
  "confidence": "high",
  "has_geo_entity": true,
  "has_spatial_relation": true,
  "geo_entity_hint": "武汉大学、珞喻路",
  "is_non_wuhan_region": false,
  "region_hint": "武汉"
}}

示例2:
输入: "今天心情不好"
输出: {{
  "is_valid": false,
  "skip_reason": "无地理信息，纯情感表达",
  "confidence": "high",
  "has_geo_entity": false,
  "has_spatial_relation": false,
  "is_non_wuhan_region": false,
  "region_hint": null
}}

示例3:
输入: "😂😂😂太好笑了"
输出: {{
  "is_valid": false,
  "skip_reason": "过短，纯表情，无语义内容",
  "confidence": "high",
  "has_geo_entity": false,
  "has_spatial_relation": false,
  "is_non_wuhan_region": false,
  "region_hint": null
}}

示例4:
输入: "北京故宫真的很壮观，推荐大家去"
输出: {{
  "is_valid": false,
  "skip_reason": "非武汉地区，明确提及北京故宫",
  "confidence": "high",
  "has_geo_entity": true,
  "has_spatial_relation": false,
  "geo_entity_hint": "故宫",
  "is_non_wuhan_region": true,
  "region_hint": "北京"
}}

示例5:
输入: "西湖边的风景不错"
输出: {{
  "is_valid": false,
  "skip_reason": "非武汉地区，西湖位于杭州",
  "confidence": "high",
  "has_geo_entity": true,
  "has_spatial_relation": true,
  "geo_entity_hint": "西湖",
  "is_non_wuhan_region": true,
  "region_hint": "杭州"
}}

示例6:
输入: "大学门口那条路堵车了"
输出: {{
  "is_valid": true,
  "confidence": "low",
  "has_geo_entity": true,
  "has_spatial_relation": true,
  "geo_entity_hint": "大学门口、那条路",
  "is_non_wuhan_region": false,
  "region_hint": "未知"
}}

示例7:
输入: "从武汉去北京出差，顺便逛了故宫"
输出: {{
  "is_valid": true,
  "confidence": "high",
  "has_geo_entity": true,
  "has_spatial_relation": false,
  "geo_entity_hint": "武汉、北京、故宫",
  "is_non_wuhan_region": false,
  "region_hint": "武汉（同时提及非武汉）"
}}

## 待筛选文本
{raw_text}

请快速判断并输出筛选结果（JSON格式）。"""

FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", FILTER_SYSTEM),
        ("human", FILTER_USER),
    ]
)


# ===== Step 0.5: Normalize 归一化提示词模板（P6新增） =====

NORMALIZE_SYSTEM = """你是一位"文本语义归一化专家"，负责将社交媒体文本改写为标准句式。
你的任务是：消解省略主语、展开模糊指代、归一化别名简称，同时严格保留原文语义。"""

NORMALIZE_USER = """## 归一化规则

**必须遵守**：
1. 不添加原文不存在的信息
2. 保留原文的核心语义和情感
3. 仅改写/展开，不筛除内容

**归一化类型**：

**1. 别名归一化（alias）**
- 简称 → 全称：如"武大" → "武汉大学"，"华师" → "华中师范大学"
- 网络用语 → 标准：如"yyds" → "非常棒"

**2. 指代消解（reference）**
- 省略主语补充：如"拍照很好看" → "(地点名)拍照很好看"
- 模糊词展开：如"这里" → 具体地点名（如能从上下文推断）
- 指代词替换：如"那边" → 具体地点名

**3. 活动归一化（activity）**
- 口语 → 标准：如"打卡" → "游览参观"，"逛街" → "购物游览"

## 边界处理

**无法推断时**：
- 模糊指代无法确定具体实体 → 保留原词，标记 confidence="low"
- 没有明显省略主语 → 保持原文，has_changes=false

## 任务示例

示例1:
输入: "武大的樱花开了，很多人在行政楼前拍照"
输出: {{
  "normalized_text": "武汉大学的樱花开放了，很多游客在武汉大学行政楼前合影留念",
  "normalizations": [
    {{\"raw\": \"武大\", \"normalized\": \"武汉大学\", \"type\": \"alias\"}},
    {{\"raw\": \"很多人\", \"normalized\": \"很多游客\", \"type\": \"reference\"}},
    {{\"raw\": \"拍照\", \"normalized\": \"合影留念\", \"type\": \"activity\"}}
  ],
  "confidence": "high",
  "preserved_semantics": true,
  "has_changes": true
}}

示例2:
输入: "群光广场就在珞喻路上，离华中师范大学很近"
输出: {{
  "normalized_text": "群光广场位于珞喻路，距离华中师范大学很近",
  "normalizations": [
    {{\"raw\": \"就在\", \"normalized\": \"位于\", \"type\": \"reference\"}},
    {{\"raw\": \"离...很近\", \"normalized\": \"距离...很近\", \"type\": \"reference\"}}
  ],
  "confidence": "high",
  "preserved_semantics": true,
  "has_changes": true
}}

示例3:
输入: "这里挺好玩"
输出: {{
  "normalized_text": "这里挺好玩",
  "normalizations": [],
  "confidence": "low",
  "preserved_semantics": true,
  "has_changes": false
}}
说明: "这里"无法推断具体地点，保留原词

示例4:
输入: "武汉大学在珞喻路上"
输出: {{
  "normalized_text": "武汉大学位于珞喻路",
  "normalizations": [
    {{\"raw\": \"在...上\", \"normalized\": \"位于\", \"type\": \"reference\"}}
  ],
  "confidence": "high",
  "preserved_semantics": true,
  "has_changes": true
}}

## 待归一化文本
{raw_text}

请输出归一化结果（JSON格式）。"""

NORMALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", NORMALIZE_SYSTEM),
        ("human", NORMALIZE_USER),
    ]
)


# ===== Step 0.7: QA Scaffold 提示词模板（P8新增） =====

QA_SCAFFOLD_SYSTEM = """你是一位"地理语义分析师"，擅长通过结构化问答来深入理解文本的地理语义。
你的任务是用5W1H框架生成问答对，构建语义脚手架帮助后续提取更准确。"""

QA_SCAFFOLD_USER = """## 5W1H 引导框架

请针对以下文本生成结构化问答对，帮助后续的实体识别和关系抽取更准确：

| 维度 | 问题方向 | 重点关注 |
|------|----------|----------|
| **Who** | 涉及的地点/实体是谁？ | 捕获地理实体名称、简称、别名 |
| **What** | 这些地点有什么特征？ | 捕获属性、功能、特色、评价 |
| **When** | 时间背景是什么？ | 季节、时段、事件时机 |
| **Where** | 位于哪里？相互位置？ | 空间关系、邻近、方位、所属 |
| **Why** | 作者为什么提到？ | 情感、推荐、评价动机 |
| **How** | 如何到达/体验？ | 交通方式、活动方式、可达性 |

## 边界处理

**简单文本（跳过详细抽取）**：
- 无地理信息：如"今天心情不好"
- 纯表情/乱码：如"😂😂😂"
- 过短文本：少于5个有效字符

**复杂文本（生成完整QA）**：
- 包含地理实体
- 有空间关系描述
- 有活动/情感表达

## 任务示例

示例1:
输入: "武大的樱花开了，很多人在行政楼前拍照打卡"

输出: {{
  "qa_pairs": [
    {{
      "question": "文中提到的主要地点是谁？",
      "answer": "武汉大学（简称武大）和武汉大学行政楼",
      "dimension": "who",
      "entities_involved": ["武汉大学", "行政楼"],
      "confidence": "high"
    }},
    {{
      "question": "这些地点有什么特征？",
      "answer": "武汉大学有樱花景观，行政楼是拍照打卡点",
      "dimension": "what",
      "entities_involved": ["武汉大学", "行政楼"],
      "confidence": "high"
    }},
    {{
      "question": "时间背景是什么？",
      "answer": "樱花开放季节，可能是春季",
      "dimension": "when",
      "entities_involved": [],
      "confidence": "medium"
    }},
    {{
      "question": "这些地点的位置关系？",
      "answer": "行政楼在武汉大学内部，'行政楼前'表明具体位置",
      "dimension": "where",
      "entities_involved": ["武汉大学", "行政楼"],
      "confidence": "high"
    }},
    {{
      "question": "作者为什么提到这些地点？",
      "answer": "推荐拍照打卡，表达对樱花景观的正面情感",
      "dimension": "why",
      "entities_involved": ["武汉大学"],
      "confidence": "medium"
    }},
    {{
      "question": "人们如何体验这些地点？",
      "answer": "在行政楼前拍照打卡",
      "dimension": "how",
      "entities_involved": ["行政楼"],
      "confidence": "high"
    }}
  ],
  "semantic_summary": "武汉大学在樱花季吸引游客，行政楼前是热门拍照打卡点",
  "entity_hints": ["武汉大学", "行政楼", "樱花"],
  "relation_hints": ["位于", "具有功能", "相对方位"],
  "context_dependencies": ["武大是武汉大学简称", "行政楼隶属于武汉大学"],
  "overall_confidence": "high",
  "should_skip_detailed_extraction": false
}}

示例2:
输入: "今天心情不好"

输出: {{
  "qa_pairs": [],
  "semantic_summary": "纯情感表达，无地理信息",
  "entity_hints": [],
  "relation_hints": [],
  "context_dependencies": [],
  "overall_confidence": "high",
  "should_skip_detailed_extraction": true
}}

示例3:
输入: "群光广场就在珞喻路上，离华中师范大学很近"

输出: {{
  "qa_pairs": [
    {{
      "question": "文中提到的地点是谁？",
      "answer": "群光广场、珞喻路、华中师范大学",
      "dimension": "who",
      "entities_involved": ["群光广场", "珞喻路", "华中师范大学"],
      "confidence": "high"
    }},
    {{
      "question": "这些地点的位置关系？",
      "answer": "群光广场位于珞喻路，距离华中师范大学很近（邻近关系）",
      "dimension": "where",
      "entities_involved": ["群光广场", "珞喻路", "华中师范大学"],
      "confidence": "high"
    }}
  ],
  "semantic_summary": "群光广场位于珞喻路，与华中师范大学邻近",
  "entity_hints": ["群光广场", "珞喻路", "华中师范大学"],
  "relation_hints": ["位于", "相对方位", "包含"],
  "context_dependencies": ["华中师范大学可简称华师"],
  "overall_confidence": "high",
  "should_skip_detailed_extraction": false
}}

## 待处理文本（已归一化）
{normalized_text}

请输出QA脚手架结果（JSON格式）。"""

QA_SCAFFOLD_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SCAFFOLD_SYSTEM),
        ("human", QA_SCAFFOLD_USER),
    ]
)


# ===== v3.4新增：实体区分规则常量 =====
# 此常量用于避免功能/事件实体被误分类为POI
ENTITY_DISTINCTION_RULES = """⚠️ **重要区分规则**：
- "购物"、"餐饮"、"头皮护理"、"打车"、"交通"等 → **功能**（不是POI）
- "樱花节"、"开业"、"EHD双店长来武汉"等 → **事件**（不是POI）
- 只有具体地点名称（如"武汉大学"、"某某咖啡厅")才是 **POI**

**判断标准**：
- 如果是**用途/活动类型** → 功能
- 如果是**发生的事情** → 事件
- 如果是**具体地点名称** → POI"""


# ===== Step 1: NER 提示词模板 =====

NER_SYSTEM = """你是一位"地理语义专家"，精通城市地理实体识别与社交媒体语料分析。
你的任务是从小红书文本中提取地理知识实体。"""

# NER实体定义部分（不含区分规则，用于拼接）
_NER_ENTITY_TYPES = """## 候选目标（v3.4扩展版：6种实体）
请识别以下类别的实体：

### 空间实体（GIS标准）—— 4种
- 道路(Road): 街道、大道、小巷等（如：关山大道）
- POI(Point of Interest): 具体店名、地标、机构（如：武汉大学、某某咖啡厅）
- 建筑物(Building): 具体的楼宇、商场主体（如：泛悦汇）
- 街区(Block): 具有边界感的生活区域（如：街道口、华农校区）

### 语义实体（v3.4新增）—— 2种
- 功能(Function): 场所可进行的用途类型（如：餐饮、购物、休闲、交通）
- 事件(Event): 发生的具体事件（如：樱花节、封路、开业）"""

# NER提示词模板（拼接实体定义 + 区分规则）
NER_USER = "\n\n".join(
    [
        _NER_ENTITY_TYPES,
        ENTITY_DISTINCTION_RULES,
        """## 思维链(CoT)
1. 首先，识别句中指代具体位置的专有名词
2. 其次，根据上下文判断其实体粒度
3. 最后，将其归入上述候选目标之一

## QA脚手架提示（如有）
前置QA分析可能发现以下实体提示，可作为参考：
{entity_hints}

上下文依赖提醒：
{context_dependencies}

## 任务示例
输入: "在洪山区的街道口，泛悦汇三楼的这家书店氛围感拉满。"
输出: {{\"道路\": [], \"POI\": [\"书店\"], \"建筑物\": [\"泛悦汇\"], \"街区\": [\"街道口\"], \"功能\": [], \"事件\": []}}

## 待处理文本
{raw_text}

请输出实体识别结果（JSON格式）。""",
    ]
)

NER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", NER_SYSTEM),
        ("human", NER_USER),
    ]
)


# ===== Step 2: RE 提示词模板（v3.2精简版：8个关系体系） =====

RE_SYSTEM = """你是一位"地理语义专家"，擅长梳理非结构化文本中的语义逻辑。
你精通社交媒体地理文本分析，能够准确提取实体间的关系和属性。"""

RE_USER = """## 候选目标
请识别实体间的以下三元组关系：<头实体, 关系, 尾实体, 属性>

### 空间基础关系（3个）—— 图谱骨架（v3.2精简版）

| 关系 | 语义定义 | Head类型 | Tail类型 | 关系属性 |
|------|----------|----------|----------|----------|
| **位于** | A坐落于B处（空间定位/归属） | 地理实体（道路/POI/建筑物/街区） | 道路/街区 | 无 |
| **包含** | A空间包含B（位于的反向） | 街区 | POI/建筑物/道路 | 无 |
| **相对方位** | A和B空间邻近+相对方位关系 | 地理实体 | 地理实体 | **距离值**+**方向值**（可选，v3.4删除联动推荐） |

**注**：原"相邻"、"距离"、"方向"已合并为"相对方位"关系，通过属性区分。
**地理实体** = 道路/POI/建筑物/街区（4种空间实体），功能实体/事件实体不能参与空间关系。

### 社交语义关系（1个）—— 图谱血肉（v3.2精简版）

| 关系 | 语义定义 | Head类型 | Tail类型 | 关系属性 |
|------|----------|----------|----------|----------|
| **具有功能** | 场所可进行的功能用途 | 地理实体（场所） | 功能节点（v3.4：10大类）或功能实体 | **时段**+**适合人群**+**具有限制**+**情感倾向**+**功能描述**（可选） |

**注**：原"承载活动"改为"具有功能"；"推荐指数"和"引发情感"已改为实体属性而非关系。

### 对比评价关系（3个）—— 特色

| 关系 | 语义定义 | Head类型 | Tail类型 | 关系属性 |
|------|----------|----------|----------|----------|
| **优于** | A在某方面好于B | 地理实体 | 地理实体 | **维度**（列表） |
| **相似** | A和B在某方面相似 | 地理实体 | 地理实体 | **维度**（列表） |
| **劣于** | A在某方面不如B | 地理实体 | 地理实体 | **维度**（列表） |

### 事件关系（1个）

| 关系 | 语义定义 | Head类型 | Tail类型 | 关系属性 |
|------|----------|----------|----------|----------|
| **发生事件** | 场所发生的特定事件 | 地理实体（场所） | 事件节点（LLM归纳命名）或事件实体 | 无（属性全部在事件节点上） |

---

## 重要说明：实体属性而非关系

以下语义应作为**实体属性**而非三元组关系抽取：

| 属性 | 说明 | 类型/示例 |
|------|------|--------|
| **推荐指数** | 整体推荐程度 | 超推/推荐/一般/不推荐 |
| **情感倾向** | 实体整体情感印象 | 正面/中性/负面 |
| **特征标签** | 实体特征描述（开放文本） | **开放文本**：保留原文表达，如氛围超好、随手拍好看、遛娃神器、松弛感、治愈感等（非枚举约束） |

**注**：交通方式、交通便利度、消费档次等已删除，由外部数据补充。

---

## 属性详细说明

### 相对方位关系属性（合并原相邻+距离+方向，v3.4删除联动推荐）

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **距离值** | 枚举（可选） | 距离远近程度 | 近/中等/远 |
| **方向值** | 枚举（可选） | 方位方向 | 东/南/西/北/东北/西南/东侧/西侧/对面/旁边 |

### 具有功能关系属性（v3.4开放文本版）

| 属性 | 类型 | 说明 | 示例 |
|------|------|------|--------|
| **时段** | 文本（可选） | 功能适用时段 | 周末/晚上/樱花季/春季/夏季等 |
| **适合人群** | 文本（开放） | 功能适合人群（保留原文表达） | 带孩子来玩、闺蜜聚会、大学生打卡等 |
| **具有限制** | 列表（开放） | 功能限制条件（保留原文表达） | 排队两小时、停车超级难、需提前预约等 |
| **情感倾向** | 枚举（可选） | 功能体验情感 | 正面/中性/负面 |

### 功能节点枚举（具有功能的tail，v3.4扩展版：10大类，新增交通）

| 类型 | 说明 | 社交媒体频率 |
|------|------|-------------|
| **餐饮** | 吃饭、探店、下午茶等餐饮活动 | 高频 |
| **购物** | 逛街、买东西等消费活动 | 高频 |
| **休闲** | 游玩、散步、放松等休闲活动 | 高频 |
| **社交** | 聚会、打卡、约会等社交活动 | 高频 |
| **观景** | 赏花、观展、拍照等观赏活动 | 高频 |
| **交通** | 打车、公交、地铁等出行功能 | 高频（v3.4新增） |
| **住宿** | 住酒店、民宿体验等住宿活动 | 中频 |
| **文化** | 学习、体验、参观等文化活动 | 中频 |
| **工作** | 办公、产业等工作相关 | 低频 |
| **其他** | 无法归类的功能 | 兜底 |

### 对比关系属性

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **维度** | 列表 | 对比的方面 | 价格/环境/服务/人流量/品质/交通/口味/其他（8个） |
| **维度描述** | 文本（可选） | 当维度=其他时，具体描述对比内容 | 自由文本，必须有原文依据 |

**注**：v3.3新增"其他"维度作为兜底，用于无法归纳到7个核心维度的情况。当使用"其他"时，必须在"维度描述"中说明具体对比内容。

### 事件节点属性（全部在事件节点上，非关系属性）

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **事件类别** | 枚举 | 事件大类 | 自然事件/人文事件/商业活动/社会事件/业态变更/停业/关闭/其他（7个） |
| **状态** | 枚举 | 事件状态 | 正在进行/已结束/计划中/周期性 |
| **时间** | 文本 | 事件时间 | 每年3月、樱花季、2024年等 |
| **详细描述** | 文本 | 事件详情 | 自由文本 |
| **情感倾向** | 枚举 | 事件情感 | 正面/中性/负面 |

---

## 思维链(CoT)
1. 观察已识别的实体对
2. 分析句中的动词/形容词/名词，判断其反映的关系类型
3. 判断是否需要提取属性（时段、维度、人群、限制等）
4. 若句子存在主语省略，请结合段落背景推断
5. **重要原则**：所有属性和关系必须有原文依据（明确出现/暗示表达/语义推断），禁止凭空创造（幻觉）
6. **注意**：推荐指数、情感倾向、特征标签等应作为实体属性而非关系

---

## 任务示例

### 示例1：空间关系+相对方位关系
输入文本: "街道口商圈里面有群光广场、银泰城，逛完可以去吃饭"
已知实体: {{\"道路\": [], \"POI\": [\"群光广场\", \"银泰城\"], \"建筑物\": [], \"街区\": [\"街道口商圈\"]}}
输出:
{{\"triples\": [
    {{\"head\": \"街道口商圈\", \"relation\": \"包含\", \"tail\": \"群光广场\", \"evidence\": \"里面有群光广场\"}},
    {{\"head\": \"街道口商圈\", \"relation\": \"包含\", \"tail\": \"银泰城\", \"evidence\": \"里面有银泰城\"}},
    {{\"head\": \"群光广场\", \"relation\": \"相对方位\", \"tail\": \"银泰城\", \"evidence\": \"一起在商圈里"}}
  ]
}}

### 示例2：具有功能关系+属性
输入文本: "周末很适合带孩子来玩，但排队很久"
已知实体: {{\"道路\": [], \"POI\": [\"公园\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"公园\", \"relation\": \"具有功能\", \"tail\": \"休闲\", \"evidence\": \"很适合带孩子来玩\", \"attributes\": {{\"时段\": \"周末\", \"适合人群\": \"亲子\", \"具有限制\": [\"排队久\"], \"情感倾向\": \"正面\"}}}}
  ]
}}

### 示例3：对比关系
输入文本: "群光比银泰贵，但环境好很多"
已知实体: {{\"道路\": [], \"POI\": [\"群光广场\", \"银泰\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"群光广场\", \"relation\": \"劣于\", \"tail\": \"银泰\", \"evidence\": \"群光比银泰贵\", \"attributes\": {{\"维度\": [\"价格\"]}}}},
    {{\"head\": \"群光广场\", \"relation\": \"优于\", \"tail\": \"银泰\", \"evidence\": \"环境好很多\", \"attributes\": {{\"维度\": [\"环境\"]}}}}
  ]
}}

### 示例4：事件关系（属性在事件节点上）
输入文本: "樱花节正在举办，超级治愈，强烈推荐"
已知实体: {{\"道路\": [], \"POI\": [\"武汉大学\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"武汉大学\", \"relation\": \"发生事件\", \"tail\": \"樱花节\", \"evidence\": \"樱花节正在举办"}}
  ]
}}
**注**："超级治愈"、"强烈推荐"应作为实体属性（情感倾向=正面、推荐指数=超推），而非关系。

### 示例5：相对方位关系（合并距离+方向）
输入文本: "咖啡厅就在地铁站附近，对面是书店"
已知实体: {{\"道路\": [], \"POI\": [\"咖啡厅\", \"地铁站\", \"书店\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"咖啡厅\", \"relation\": \"相对方位\", \"tail\": \"地铁站\", \"evidence\": \"就在地铁站附近\", \"attributes\": {{\"距离值\": \"近\"}}}},
    {{\"head\": \"书店\", \"relation\": \"相对方位\", \"tail\": \"咖啡厅\", \"evidence\": \"对面是书店\", \"attributes\": {{\"方向值\": \"对面\"}}}}
  ]
}}

### 示例6：业态变更事件
输入文本: "这家书店改成咖啡厅了"
已知实体: {{\"道路\": [], \"POI\": [\"书店\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"书店\", \"relation\": \"发生事件\", \"tail\": \"业态变更\", \"evidence\": \"改成咖啡厅了"}}
  ]
}}
**注**：事件属性在事件节点上：事件类别=业态变更、状态=已结束、详细描述="改成咖啡厅了"。

---

## QA脚手架提示（如有）
前置QA分析可能发现以下关系提示，可作为参考：
{relation_hints}

上下文依赖提醒：
{context_dependencies}

## 已识别实体
{entities}

## 待处理文本
{raw_text}

请输出关系抽取结果（JSON格式）。"""

RE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RE_SYSTEM),
        ("human", RE_USER),
    ]
)


# ===== Step 3: 三元组评估提示词模板 =====

EVAL_1_SYSTEM = """你是一位"地理语义评审专家"。你的任务是对生成的地理关系三元组进行真实性与逻辑性评估。"""

EVAL_1_USER = """## 评估维度
- 语义准确性(SEM): 三元组是否准确代表了原文意思？（1-5分）
- 事实真实性(FAC): 是否符合地理常识？（1-5分）
- 一致性(CON): 关系方向是否正确？（1-5分）

评分标准：
- 5分: 完全正确
- 4分: 基本正确，有小瑕疵
- 3分: 可接受
- 2分: 有问题
- 1分: 错误

## QA脚手架提示（如有）
前置QA分析的语义理解可作为参考：
{semantic_summary}

上下文依赖提醒：
{context_dependencies}

## 待评估三元组
{triples}

## 原始文本
{raw_text}

请输出评分结果（JSON格式）。"""

EVAL_PROMPT_1 = ChatPromptTemplate.from_messages(
    [
        ("system", EVAL_1_SYSTEM),
        ("human", EVAL_1_USER),
    ]
)


EVAL_2_SYSTEM = """你是同一位"地理语义评审专家"，现在进行二次验证。"""

EVAL_2_USER = """## 任务
请结合原始语料，重新验证你刚才的评分：
1. 检查是否存在"虚假幻觉"（三元组在原文中不存在依据）
2. 检查是否存在"逻辑偏差"（关系方向或类型错误）

## 任务强调
如果发现错误，请给出修正后的三元组。

## 你刚才的评分结果
{previous_scores}

## 原始语料
{raw_text}

请输出二次验证结果（JSON格式）。"""

EVAL_PROMPT_2 = ChatPromptTemplate.from_messages(
    [
        ("system", EVAL_2_SYSTEM),
        ("human", EVAL_2_USER),
    ]
)


# P2改进：简化的单次评估提示词（合并评分和修正）
EVAL_SIMPLIFIED_SYSTEM = (
    """你是一位"地理语义评审专家"。你的任务是评估三元组并在发现错误时直接修正。"""
)

EVAL_SIMPLIFIED_USER = """## 评估维度
- 语义准确性(SEM): 三元组是否准确代表了原文意思？（1-5分）
- 事实真实性(FAC): 是否符合地理常识？（1-5分）
- 一致性(CON): 关系方向是否正确？（1-5分）

评分标准：
- 5分: 完全正确
- 4分: 基本正确，有小瑕疵
- 3分: 可接受
- 2分: 有问题，需要修正
- 1分: 错误，必须修正

## 修正规则
如果评分低于3分，请在corrections中给出修正后的三元组：
- 修正关系类型（如：将"旁边"标准化为"相对方位"）
- 修正关系方向（如：将<A, 位于, B>改为<B, 位于, A>)
- 删除无效三元组（如：幻觉、无依据）

## QA脚手架提示（如有）
前置QA分析的语义理解可作为参考：
{semantic_summary}

上下文依赖提醒：
{context_dependencies}

## 待评估三元组
{triples}

## 原始文本
{raw_text}

请输出评估结果（JSON格式），包含评分和可选修正。"""

EVAL_PROMPT_SIMPLIFIED = ChatPromptTemplate.from_messages(
    [
        ("system", EVAL_SIMPLIFIED_SYSTEM),
        ("human", EVAL_SIMPLIFIED_USER),
    ]
)


# ===== Step 4: 属性标注提示词模板（v3.0精简版） =====

LABEL_SYSTEM = """你是一位"地理知识管理专家"，精通GIS标准、城市规划术语和社交媒体语义分析。
你的任务是将初步知识片段转化为具备专业语义背景的结构化知识。"""

LABEL_USER = """## 任务描述
请为已识别的实体和关系打上专业属性标签（v3.3：特征标签开放文本，细分简化）。

---

## 实体属性标注（v3.3：5个属性）

### 基础分类属性（必须有原文依据）

| 属性 | 说明 |
|------|------|
| **类别** | 4大类枚举（道路/POI/建筑物/街区），用于NER边界识别 |
| **细分** | **开放文本**：仅记录文本中明确提及的分类词（如"餐厅"、"商场"）。权威分类由数据源在对齐阶段补充 |

**重要说明**：
- 类别：4大类是GIS标准分类，帮助判断实体边界
- 细分：不做强制枚举约束，仅记录文本中明确表达的分类词
- 实体入库时通过entity_alignment关联高德POI，继承权威分类

### 文本属性（从语料提取，必须有原文依据）

| 属性 | 类型/说明 |
|------|----------|
| **特征标签** | **开放文本**：保留原文真实表达，如"氛围超好"、"随手拍好看"、"遛娃神器"等（非枚举约束） |
| 推荐指数 | 超推、推荐、一般、不推荐（枚举） |
| 情感倾向 | 正面、中性、负面（枚举） |

### 特征标签标注原则（v3.3开放文本设计）

**设计原因**：社交媒体特征表达多样且新词涌现，预定义枚举无法完全覆盖。

**标注原则**：
1. **语义准确**：提取原文中准确表达实体特征的自然语言
2. **保留原貌**：不做标准化归一化，保留用户真实表达风格
3. **原文依据**：所有特征必须在原文有明确或暗示依据
4. **多特征支持**：一个实体可有多个特征标签

**常见特征类型参考**（仅供参考，不强制约束）：
- 氛围情绪：氛围好、松弛感、治愈感、安静、私密、解压
- 拍照体验：出片、拍照好看、随手拍好看、适合拍照
- 风格审美：文艺、复古、网红、ins风、日系、韩风
- 知名度：热门、宝藏、打卡圣地、老字号
- 服务体验：服务好、环境好
- 价格评价：不贵、平价、性价比高
- 人群适配：遛娃神器、亲子友好、情侣约会首选
- 便利性：交通方便、好停车、地铁直达

**注意**：所有属性必须有原文依据（明确出现、暗示表达、语义推断）。禁止凭空创造（幻觉）。

---

## 关系属性标注（v3.4精简版）

| 属性 | 适用关系 | 枚举值/类型 |
|------|----------|-------------|
| 距离值 | 相对方位 | 近/中等/远 |
| 方向值 | 相对方位 | 东/南/西/北/东北/西南/东侧/西侧/对面/旁边 |
| 时段 | 具有功能 | 周末/晚上/樱花季等（文本） |
| 适合人群 | 具有功能 | **开放文本**（保留原文表达：带孩子来玩、闺蜜聚会等） |
| 具有限制 | 具有功能 | **开放文本列表**（保留原文表达：排队两小时、停车超级难等） |
| 情感倾向 | 具有功能 | 正面/中性/负面 |
| 维度 | 优于、相似、劣于 | 价格/环境/服务/人流量/品质/交通/口味/其他（多选，8个） |
| 维度描述 | 优于、相似、劣于 | 自由文本（当维度=其他时使用） |

**注意**：仅标注语料中明确出现的属性，不可凭空创造。所有属性可选。

---

## 任务示例

输入实体: ["武汉大学", "群光广场", "街道口"]
输入关系: [
  "<武汉大学, 位于, 珞喻路>",
  "<街道口, 包含, 群光广场>"
]
原始文本: "武汉大学樱花季超治愈，强烈推荐学生党打卡，旁边群光广场可以坐地铁到，人均消费约100元"

输出:
{{\"entities\": {{
    \"武汉大学\": {{
      \"类别\": \"POI\",
      \"细分\": \"教育\",
      \"特征标签\": [\"网红\", \"热门\", \"超治愈\"],
      \"推荐指数\": \"超推\",
      \"情感倾向\": \"正面\"}},
    \"群光广场\": {{
      \"类别\": \"建筑物\",
      \"细分\": \"商业综合体\",
      \"特征标签\": [\"热门\"],
      \"推荐指数\": null,
      \"情感倾向\": \"正面\"}},
    \"街道口\": {{
      \"类别\": \"街区\",
      \"细分\": \"商圈\",
      \"特征标签\": null,
      \"推荐指数\": null,
      \"情感倾向\": null}}
  }},
  \"relations\": {{
    \"<武汉大学, 位于, 珞喻路>\": {{}},
    \"<街道口, 包含, 群光广场>\": {{}}
  }}
}}

---

## QA脚手架提示（如有）
前置QA分析的语义理解可作为参考：
{semantic_summary}

实体提示：
{entity_hints}

关系提示：
{relation_hints}

## 待标注实体
{entities}

## 待标注关系
{relations}

## 原始文本
{raw_text}

请输出属性标注结果（JSON格式）。"""

LABEL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", LABEL_SYSTEM),
        ("human", LABEL_USER),
    ]
)


# ===== 辅助函数 =====


def format_entities(entities: dict) -> str:
    """格式化实体字典用于提示词"""
    result = []
    for entity_type, names in entities.items():
        if names:
            result.append(f"- {entity_type}: {', '.join(names)}")
        else:
            result.append(f"- {entity_type}: (无)")
    return "\n".join(result)


# ===== QA Scaffold 上下文格式化函数（P8新增） =====


def format_entity_hints(entity_hints: list) -> str:
    """格式化实体提示用于 NER 提示词"""
    if not entity_hints:
        return "(无实体提示)"
    return f"可能涉及的实体: {', '.join(entity_hints)}"


def format_relation_hints(relation_hints: list) -> str:
    """格式化关系提示用于 RE 提示词"""
    if not relation_hints:
        return "(无关系提示)"
    return f"可能涉及的关系类型: {', '.join(relation_hints)}"


def format_context_dependencies(context_dependencies: list) -> str:
    """格式化上下文依赖用于提示词"""
    if not context_dependencies:
        return "(无上下文依赖)"
    return "\n".join([f"- {dep}" for dep in context_dependencies])


def format_triples(triples: list) -> str:
    """格式化三元组列表用于提示词（v2.2改进：支持attributes）"""
    if not triples:
        return "(无三元组)"
    lines = []
    for t in triples:
        # 基础三元组字符串
        base_str = f"<{t['head']}, {t['relation']}, {t['tail']}>"

        # 如果有属性，添加属性描述
        attrs = t.get("attributes", {})
        if attrs:
            attr_strs = []
            for key, value in attrs.items():
                if isinstance(value, list):
                    attr_strs.append(f"{key}=[{','.join(value)}]")
                else:
                    attr_strs.append(f"{key}={value}")
            lines.append(f"- {base_str} [{', '.join(attr_strs)}]")
        else:
            lines.append(f"- {base_str}")
    return "\n".join(lines)


def format_triples_with_evidence(triples: list) -> str:
    """格式化三元组列表（含证据）用于提示词"""
    if not triples:
        return "(无三元组)"
    lines = []
    for t in triples:
        base_str = f"<{t['head']}, {t['relation']}, {t['tail']}>"
        evidence = t.get("evidence", "")
        attrs = t.get("attributes", {})

        parts = [base_str]
        if attrs:
            attr_strs = []
            for key, value in attrs.items():
                if isinstance(value, list):
                    attr_strs.append(f"{key}=[{','.join(value)}]")
                else:
                    attr_strs.append(f"{key}={value}")
            parts.append(f"[{', '.join(attr_strs)}]")
        if evidence:
            parts.append(f'证据:"{evidence}"')

        lines.append(f"- {' '.join(parts)}")
    return "\n".join(lines)


# ===== Self-Check: 实体校验提示词模板 =====

SELF_CHECK_NER_SYSTEM = """你是一位"实体校验专家"，负责独立审视NER抽取结果。
你的任务是客观评估，不带偏见地检查遗漏、识别别名、过滤无关实体。
你需要判断整体质量并决定是否需要重新抽取。"""

SELF_CHECK_NER_USER = """## 校验任务
请对NER抽取结果进行独立校验：

1. **遗漏检查**：原文是否提及地理实体但未抽取？
   - 检查是否有明确的地名、道路、建筑被遗漏
   - 检查是否有简称/别名被忽略

2. **别名识别**：抽取的实体是否有简称需归一化？
   - 如"武大"应归一化为"武汉大学"
   - 如"华农"应归一化为"华中农业大学"

3. **无关过滤**：抽取的实体是否为非地理实体？
   - 过滤人名、时间、数字等非地理实体
   - 过滤过于泛化的词（如"这里"、"那里"）

4. **置信度判断**：
   - high: 遗漏≤1个，无严重别名问题
   - medium: 遗漏2-3个，或有别名问题但可归一化
   - low: 遗漏>3个，或有多处重要实体遗漏

## QA脚手架提示（如有）
前置QA分析的语义理解可作为参考：
{semantic_summary}

上下文依赖提醒：
{context_dependencies}

## 已抽取实体
{entities}

## 原始文本
{raw_text}

## 重试提示（如有）
{retry_hint}

请输出校验结果（JSON格式）。"""

SELF_CHECK_NER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_NER_SYSTEM),
        ("human", SELF_CHECK_NER_USER),
    ]
)


# ===== Self-Check: 三元组校验提示词模板 =====

SELF_CHECK_RE_SYSTEM = """你是一位"三元组校验专家"，负责独立审视RE抽取结果。
你的任务是客观评估，检测幻觉、验证关系、匹配证据。
你需要判断整体质量并决定是否需要重新抽取。"""

SELF_CHECK_RE_USER = """## 校验任务
请对RE抽取结果进行独立校验：

1. **幻觉检测**：三元组是否在原文中有依据？
   - 检查三元组是否凭空生成（无原文支持）
   - 检查关系是否在原文中确实存在
   - 幻觉三元组应标记为rejected

2. **关系验证**：关系类型和方向是否正确？
   - 检查关系类型是否匹配原文语义
   - 检查头尾实体顺序是否正确（A-位于-B vs B-位于-A）

3. **证据匹配**：标注的证据是否真实存在于原文？
   - 验证evidence字段是否来自原文
   - 无效证据应标记并给出修正建议

4. **置信度判断**：
   - high: 幻觉≤1个，无严重错误
   - medium: 幻觉2-3个，或有小错误可修正
   - low: 幻觉>3个，或有严重关系方向错误

5. **重抽建议**：
   - 如果confidence=low且rejected_triples>3，建议重抽
   - 指明重抽目标（ner/re）和原因

## QA脚手架提示（如有）
前置QA分析的语义理解可作为参考：
{semantic_summary}

上下文依赖提醒：
{context_dependencies}

## 已抽取三元组
{triples}

## 原始文本
{raw_text}

## 已校验实体（来自Self-Check-NER）
{verified_entities}

## 重试提示（如有）
{retry_hint}

请输出校验结果（JSON格式）。"""

SELF_CHECK_RE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_RE_SYSTEM),
        ("human", SELF_CHECK_RE_USER),
    ]
)


# ===== Self-Check 辅助函数 =====


def format_verified_entities(entities: list) -> str:
    """格式化校验后的实体列表"""
    if not entities:
        return "(无)"
    lines = []
    for e in entities:
        aliases = e.get("aliases", [])
        alias_str = f" (别名: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {e['name']} [{e['type']}] 置信度:{e['confidence']}{alias_str}")
    return "\n".join(lines)


def format_retry_hint(problem_entities: list, problem_triples: list) -> str:
    """格式化重试提示"""
    hints = []
    if problem_entities:
        hints.append(f"上次遗漏的实体建议: {', '.join(problem_entities[:5])}")
    if problem_triples:
        hints.append(f"上次问题三元组: {problem_triples[:3]}")
    return "\n".join(hints) if hints else "(无重试提示)"


# ===== P9新增：联合抽取提示词（v3.0精简版：13种关系） =====

JOINT_NER_RE_SYSTEM = """你是一位"地理语义联合抽取专家"，擅长在一次推理中同时识别实体和关系。
你的优势在于：能够全局理解文本，避免实体边界识别错误对关系判定的干扰。"""

JOINT_NER_RE_USER = (
    """## 任务描述
请从文本中**同时**抽取：
1. 地理实体和语义实体（6种类型，v3.4扩展版）
2. 实体间的语义关系（8种关系类型）
3. 每个抽取的证据依据

## 实体类型定义（v3.4扩展版：6种）

### 空间实体（GIS标准）—— 4种
| 类型 | 定义 | 示例 |
|------|------|------|
| 道路 | 交通通道 | 珞喻路、关山大道 |
| POI | 具体地点/机构 | 武汉大学、群光广场 |
| 建筑物 | 建筑设施 | 泛悦汇、融科天城 |
| 街区 | 地理区域 | 街道口、光谷商圈 |

### 语义实体（v3.4新增）—— 2种
| 类型 | 定义 | 示例 |
|------|------|------|
| 功能 | 场所可进行的用途类型 | 餐饮、购物、休闲、交通 |
| 事件 | 发生的具体事件 | 樱花节、封路、开业、停业 |

"""
    + ENTITY_DISTINCTION_RULES
    + """

## 关系类型（v3.4精简版：8种）
### 空间基础关系（3个）—— 图谱骨架
| 关系 | 语义定义 | Head类型 | Tail类型 | 关系属性 |
|------|----------|----------|----------|----------|
| 位于 | A坐落于B处（空间定位/归属） | 地理实体（道路/POI/建筑物/街区） | 道路/街区 | 无 |
| 包含 | A空间包含B（位于的反向） | 街区 | POI/建筑物/道路 | 无 |
| 相对方位 | A和B空间邻近+相对方位关系 | 地理实体 | 地理实体 | **距离值**+**方向值**（可选，v3.4删除联动推荐） |

**注**：原"相邻"、"距离"、"方向"已合并为"相对方位"关系。**地理实体** = 道路/POI/建筑物/街区。

### 社交语义关系（1个）—— 图谱血肉
| 关系 | 语义定义 | Head类型 | Tail类型 | 关系属性 |
|------|----------|----------|----------|----------|
| 具有功能 | 场所可进行的功能用途 | 地理实体（场所） | 功能节点（9大类）或功能实体 | **时段**+**适合人群**(开放文本)+**具有限制**(开放文本列表)+**情感倾向**（可选） |

**注**：原"承载活动"改为"具有功能"；"推荐指数"和"引发情感"已改为实体属性。

### 对比评价关系（3个）—— 特色
| 关系 | 语义定义 | Head类型 | Tail类型 | 关系属性 |
|------|----------|----------|----------|----------|
| 优于 | A在某方面好于B | 地理实体 | 地理实体 | **维度**（列表） |
| 相似 | A和B在某方面相似 | 地理实体 | 地理实体 | **维度**（列表） |
| 劣于 | A在某方面不如B | 地理实体 | 地理实体 | **维度**（列表） |

### 事件关系（1个）
| 关系 | 语义定义 | Head类型 | Tail类型 | 关系属性 |
|------|----------|----------|----------|----------|
| 发生事件 | 场所发生的特定事件 | 地理实体（场所） | 事件节点或事件实体 | 无（属性全部在事件节点上） |

---

## 重要说明：实体属性而非关系

以下语义应作为**实体属性**而非三元组关系抽取：

| 属性 | 类型/示例 |
|------|--------|
| 推荐指数 | 超推/推荐/一般/不推荐 |
| 情感倾向 | 正面/中性/负面 |
| 特征标签 | **开放文本**：保留原文表达，如氛围超好、随手拍好看、遛娃神器、松弛感、治愈感等（非枚举约束） |

**注**：交通方式、交通便利度、消费档次等已删除，由外部数据补充。

---

## 属性详细说明

### 相对方位关系属性（v3.4删除联动推荐）
| 属性 | 枚举值 |
|------|--------|
| 距离值 | 近/中等/远 |
| 方向值 | 东/南/西/北/东北/西南/东侧/西侧/对面/旁边 |

### 具有功能关系属性（v3.4开放文本版）
| 属性 | 类型/枚举值 |
|------|--------|
| 时段 | 周末/晚上/樱花季/春季/夏季等（文本） |
| 适合人群 | **开放文本**：带孩子来玩、闺蜜聚会、大学生打卡等（保留原文表达） |
| 具有限制 | **开放文本列表**：排队两小时、停车超级难、需提前预约等（保留原文表达） |
| 情感倾向 | 正面/中性/负面 |

### 功能节点枚举（具有功能的tail，v3.2精简版：9大类）
| 类型 | 说明 | 社交媒体频率 |
|------|------|-------------|
| 餐饮 | 吃饭、探店、下午茶等餐饮活动 | 高频 |
| 购物 | 逛街、买东西等消费活动 | 高频 |
| 休闲 | 游玩、散步、放松等休闲活动 | 高频 |
| 社交 | 聚会、打卡、约会等社交活动 | 高频 |
| 观景 | 赏花、观展、拍照等观赏活动 | 高频 |
| 住宿 | 住酒店、民宿体验等住宿活动 | 中频 |
| 文化 | 学习、体验、参观等文化活动 | 中频 |
| 工作 | 办公、产业等工作相关 | 低频 |
| 其他 | 无法归类的功能 | 兜底 |

### 对比关系属性
| 属性 | 枚举值 |
|------|--------|
| 维度 | 价格/环境/服务/人流量/品质/交通/口味/其他（8个） |

### 事件节点属性（全部在事件节点上）
| 属性 | 枚举值 |
|------|--------|
| 事件类别 | 自然事件/人文事件/商业活动/社会事件/业态变更/停业/关闭/其他（7个） |
| 状态 | 正在进行/已结束/计划中/周期性 |
| 时间 | 每年3月、樱花季、2024年等 |
| 详细描述 | 自由文本 |
| 情感倾向 | 正面/中性/负面 |

---

## QA脚手架提示（如有）
{entity_hints}
{relation_hints}
{context_dependencies}

## 导师指导（如有）
{mentor_guidance}

## 联合抽取策略（CoT）
1. **第一步**：扫描文本，识别所有可能的地名、道路、建筑等
2. **第二步**：对识别的实体，判断其类型和类别
3. **第三步**：分析实体之间的语义关系，抽取三元组
4. **第四步**：为每个抽取提供原文依据（evidence）
5. **第五步**：评估整体置信度
6. **重要原则**：所有属性和关系必须有原文依据（明确出现/暗示表达/语义推断），禁止凭空创造（幻觉）
7. **重要**：推荐指数、情感倾向、特征标签应作为实体属性而非关系

## 任务示例

### 示例1：基础联合抽取+包含关系
输入: "街道口商圈里面有群光广场和银泰城，逛完可以去吃饭"

输出:
 {{
  "entities": [
    {{\"name\": "街道口商圈", "type": "街区", "category": "商圈", "aliases": [], "evidence": "街道口"}},
    {{\"name\": "群光广场", "type": "建筑物", "category": "商业综合体", "aliases": [], "evidence": "群光广场"}},
    {{\"name\": "银泰城", "type": "建筑物", "category": "商业综合体", "aliases": [], "evidence": "银泰城"}}
  ],
  "triples": [
    {{\"head": "街道口商圈", "relation": "包含", "tail": "群光广场", "evidence": "里面有群光广场", "confidence": "high"}},
    {{\"head": "街道口商圈", "relation": "包含", "tail": "银泰城", "evidence": "里面有银泰城", "confidence": "high"}},
    {{\"head": "群光广场", "relation": "相对方位", "tail": "银泰城", "evidence": "一起在商圈里", "confidence": "high"}}
  ],
  "entity_relation_mapping": {{
    "街道口商圈": ["<街道口商圈, 包含, 群光广场>", "<街道口商圈, 包含, 银泰城>"]
  }},
  "overall_confidence": "high"
 }}

### 示例2：复杂语义关系+实体属性
输入: "群光广场就在珞喻路上，比街道口更热闹，周末适合带娃逛街，可以坐地铁到"

输出:
 {{
  "entities": [
    {{\"name\": "群光广场", "type": "建筑物", "category": "商业综合体", "aliases": [], "evidence": "群光广场", "attributes": {{\"特征标签\": ["网红", "热门"]}}}},
    {{\"name\": "珞喻路", "type": "道路", "category": "主干道", "aliases": [], "evidence": "珞喻路"}},
    {{\"name\": "街道口", "type": "街区", "category": "商圈", "aliases": [], "evidence": "街道口"}}
  ],
  "triples": [
    {{\"head\": "群光广场", "relation": "位于", "tail": "珞喻路", "evidence": "就在珞喻路上", "confidence": "high"}},
    {{\"head": "群光广场", "relation": "优于", "tail": "街道口", "evidence": "比街道口更热闹", "confidence": "medium", "attributes": {{\"维度\": ["人流量"]}}}},
    {{\"head": "群光广场", "relation": "具有功能", "tail": "购物", "evidence": "周末适合带娃逛街", "confidence": "high", "attributes": {{\"时段": "周末", "适合人群": "亲子"}}}}
  ],
  "entity_relation_mapping": {{
    "群光广场": ["<群光广场, 位于, 珞喻路>", "<群光广场, 优于, 街道口>", "<群光广场, 具有功能, 购物>"]
  }},
  "overall_confidence": "high"
 }}

### 示例3：业态变更事件（替代原"变化为"关系）
输入: "这家书店改成咖啡厅了"

输出:
 {{
  "entities": [
    {{\"name\": "书店", "type": "POI", "category": "文化", "aliases": [], "evidence": "书店"}}
  ],
  "triples": [
    {{\"head": "书店", "relation": "发生事件", "tail": "业态变更", "evidence": "改成咖啡厅了", "confidence": "high"}}
  ],
  "event_nodes": [
    {{\"name": "业态变更", "事件类别": "业态变更", "状态": "已结束", "详细描述": "改成咖啡厅了", "情感倾向": "中性"}}
  ],
  "entity_relation_mapping": {{
    "书店": ["<书店, 发生事件, 业态变更>"]
  }},
  "overall_confidence": "medium"
 }}

## 待处理文本
{raw_text}

请输出联合抽取结果（JSON格式）。"""
)

JOINT_NER_RE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JOINT_NER_RE_SYSTEM),
        ("human", JOINT_NER_RE_USER),
    ]
)


# ===== P9新增：Self-Check-Joint提示词（含Reflexion） =====

SELF_CHECK_JOINT_SYSTEM = """你是一位"联合抽取校验专家"，负责独立审视Joint Extraction结果。
你的任务是：客观评估、检测幻觉、验证关系、并生成**自然语言反思建议**供重试轮参考。"""

SELF_CHECK_JOINT_USER = """## 校验任务

### 1. 实体校验
- **遗漏检查**：原文是否提及地理实体但未抽取？
- **类型验证**：实体类型是否正确？
- **无关过滤**：是否抽取了非地理实体？

### 2. 关系校验
- **幻觉检测**：三元组是否在原文中有依据？
- **关系验证**：关系类型和方向是否正确？
- **证据匹配**：evidence字段是否来自原文？

### 3. Reflexion反思（核心）
请生成自然语言形式的反思建议，指导下一轮抽取改进：
- 总结本次抽取的主要问题
- 分析问题产生的原因
- 提出具体的改进策略

### 4. 置信度判断
- high: 遗漏≤1，幻觉≤1，无严重错误
- medium: 遗漏2-3，幻觉2-3，可修正
- low: 遗漏>3，幻觉>3，需重抽

## 待校验结果
实体: {entities}
三元组: {triples}

## 原始文本
{raw_text}

## QA脚手架提示
{semantic_summary}
{context_dependencies}

## 重试历史（如有）
上一轮反思: {previous_reflection}
改进尝试: {improvement_attempts}

请输出校验结果，重点输出reflection_text和improvement_strategy。"""

SELF_CHECK_JOINT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_JOINT_SYSTEM),
        ("human", SELF_CHECK_JOINT_USER),
    ]
)


# ===== P9新增：Self-Check-QA提示词 =====

SELF_CHECK_QA_SYSTEM = """你是一位"QA脚手架校验专家"，负责独立审视QA Scaffold结果。
你的任务是：评估问答质量、检查实体覆盖度、验证关系覆盖度，并生成反思建议。"""

SELF_CHECK_QA_USER = """## 校验任务

### 1. QA质量评估
- **问答一致性**：问答内容是否与原文一致？
- **维度完整性**：5W1H维度是否覆盖关键信息？
- **实体覆盖度**：QA是否识别了所有关键地理实体？

### 2. 实体提示验证
- entity_hints是否遗漏重要实体？
- 是否包含非地理实体？

### 3. 关系提示验证
- relation_hints是否覆盖主要关系类型？
- 是否有误导性提示？

### 4. Reflexion反思
请生成反思建议：
- 总结QA生成的不足之处
- 建议改进方向

### 5. 置信度判断
- high: 实体遗漏≤1，维度完整
- medium: 实体遗漏2-3，部分维度缺失
- low: 实体遗漏>3，QA质量差

## 待校验QA结果
问答对: {qa_pairs}
实体提示: {entity_hints}
关系提示: {relation_hints}
语义摘要: {semantic_summary}

## 原始文本
{raw_text}

请输出校验结果。"""

SELF_CHECK_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_QA_SYSTEM),
        ("human", SELF_CHECK_QA_USER),
    ]
)


# ===== P9新增：Self-Check-Eval提示词 =====

SELF_CHECK_EVAL_SYSTEM = """你是一位"评估结果校验专家"，负责审视三元组评分结果。
你的任务是：验证评分合理性、检查修正效果，并生成反思建议。"""

SELF_CHECK_EVAL_USER = """## 校验任务

### 1. 评分一致性检查
- 评分是否与三元组质量匹配？
- 是否存在评分过高（幻觉三元组得高分）或过低（正确三元组得低分）？

### 2. 修正效果验证
- corrected_triples是否正确应用了修正？
- 是否遗漏了需要修正的三元组？

### 3. Reflexion反思
请生成反思建议：
- 评估过程中的问题分析
- 改进建议

### 4. 置信度判断
- high: 评分准确，修正完整
- medium: 有评分偏差但可控
- low: 评分不合理，需重评

## 待校验评估结果
三元组评分: {eval_scores}
修正后三元组: {corrected_triples}
评估通过: {eval_passed}

## 原始文本
{raw_text}

请输出校验结果。"""

SELF_CHECK_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_EVAL_SYSTEM),
        ("human", SELF_CHECK_EVAL_USER),
    ]
)


# ===== P9新增：Self-Check-Label提示词 =====

SELF_CHECK_LABEL_SYSTEM = """你是一位"标注结果校验专家"，负责审视属性标注结果。
你的任务是：验证属性合理性、检查完整性，并生成反思建议。"""

SELF_CHECK_LABEL_USER = """## 校验任务

### 1. 实体属性验证（Schema v3.3）
- 类别是否正确？（道路/POI/建筑物/街区，用于NER边界识别）
- 细分是否为开放文本？（仅记录文本中明确提及的分类词，不做强制枚举）
- 特征标签是否与原文匹配？（开放文本，保留原文表达）
- 推荐指数是否合理？（超推/推荐/一般/不推荐）
- 情感倾向是否准确？（正面/中性/负面）

### 2. 关系属性验证（Schema v3.2）
- 相对方位关系属性：距离值、方向值是否正确？（v3.4已删除联动推荐）
- 功能关系属性：时段、适合人群、具有限制、情感倾向是否合理？
- 对比关系属性：维度列表是否准确？（价格/环境/服务等8个维度，含"其他"）
- 是否遗漏关键属性？

### 3. Reflexion反思
请生成反思建议：
- 标注过程中的问题分析
- 改进建议

### 4. 置信度判断
- high: 属性完整准确
- medium: 部分属性缺失或有偏差
- low: 属性标注质量差

## 待校验标注结果
实体属性: {entity_attrs}
关系属性: {relation_attrs}

## 原始文本
{raw_text}

请输出校验结果。"""

SELF_CHECK_LABEL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_LABEL_SYSTEM),
        ("human", SELF_CHECK_LABEL_USER),
    ]
)


# ===== P9新增：Self-Check-Filter提示词（可选） =====

SELF_CHECK_FILTER_SYSTEM = """你是一位"文本筛选校验专家"，负责独立审视Filter筛选结果。
你的任务是：验证筛选判定的合理性，检测误筛（有效文本被判定为无效）和误判（无效文本被判定为有效），并生成反思建议。"""

SELF_CHECK_FILTER_USER = """## 校验任务

### 1. 筛选判定验证
- **is_valid判定是否合理？**
- 如果判定为无效，是否确实没有地理信息？
- 如果判定为有效，是否真的包含地理实体或空间关系？

### 2. 误筛检测
- 是否有地理实体被遗漏？
- 是否有模糊指代（如"这里"、"那边"）暗示地理信息？

### 3. 误判检测
- 是否将纯情感/无关文本判定为有效？
- 是否误读了文本内容？

### 4. Reflexion反思
请生成反思建议：
- 筛选判定的主要问题分析
- 改进策略

### 5. 置信度判断
- high: 判定准确，无误筛/误判
- medium: 有轻微偏差但可控
- low: 判定不合理，需要重筛

## 待校验筛选结果
is_valid: {is_valid}
confidence: {confidence}
skip_reason: {skip_reason}
has_geo_entity: {has_geo_entity}
has_spatial_relation: {has_spatial_relation}
geo_entity_hint: {geo_entity_hint}

## 原始文本
{raw_text}

请输出校验结果。"""

SELF_CHECK_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_FILTER_SYSTEM),
        ("human", SELF_CHECK_FILTER_USER),
    ]
)


# ===== P9新增：Self-Check-Normalize提示词（可选） =====

SELF_CHECK_NORMALIZE_SYSTEM = """你是一位"文本归一化校验专家"，负责独立审视Normalize归一化结果。
你的任务是：验证归一化质量，检查语义保留，检测信息添加/丢失问题，并生成反思建议。"""

SELF_CHECK_NORMALIZE_USER = """## 校验任务

### 1. 归一化质量验证
- **别名归一化是否正确？**（如"武大"→"武汉大学"）
- **指代消解是否合理？**（如"这里"→具体地点）
- **活动归一化是否恰当？**（如"打卡"→"游览参观"）

### 2. 语义保留检查
- 是否保留了原文的核心语义？
- 是否添加了原文不存在的信息？（不应添加）
- 是否丢失了原文关键信息？（不应丢失）

### 3. 归一化记录校验
- 每条归一化记录是否合理？
- 是否有过度归一化或不当归一化？

### 4. Reflexion反思
请生成反思建议：
- 归一化过程中的问题分析
- 改进策略

### 5. 置信度判断
- high: 归一化准确，语义完全保留
- medium: 有轻微偏差但语义保留
- low: 归一化有问题，需要重做

## 待校验归一化结果
normalized_text: {normalized_text}
confidence: {confidence}
has_changes: {has_changes}
preserved_semantics: {preserved_semantics}

归一化记录:
{normalizations}

## 原始文本
{raw_text}

请输出校验结果。"""

SELF_CHECK_NORMALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_NORMALIZE_SYSTEM),
        ("human", SELF_CHECK_NORMALIZE_USER),
    ]
)


# ===== P9新增：格式化辅助函数 =====


def format_joint_entities(entities: list) -> str:
    """格式化联合抽取的实体列表"""
    if not entities:
        return "(无实体)"
    lines = []
    for e in entities:
        aliases = e.get("aliases", [])
        alias_str = f" (别名: {', '.join(aliases)})" if aliases else ""
        evidence = e.get("evidence", "")
        lines.append(
            f'- {e.get("name", "")} [{e.get("type", "")}] 类别:{e.get("category", "")}{alias_str} 证据:"{evidence}"'
        )
    return "\n".join(lines)


def format_joint_triples(triples: list) -> str:
    """格式化联合抽取的三元组列表"""
    if not triples:
        return "(无三元组)"
    lines = []
    for t in triples:
        base = f"<{t.get('head', '')}, {t.get('relation', '')}, {t.get('tail', '')}>"
        confidence = t.get("confidence", "")
        evidence = t.get("evidence", "")
        attrs = t.get("attributes", {})
        attr_str = ""
        if attrs:
            attr_str = f" [{', '.join(f'{k}={v}' for k, v in attrs.items())}]"
        lines.append(f'- {base} 置信度:{confidence}{attr_str} 证据:"{evidence}"')
    return "\n".join(lines)


def format_qa_pairs_for_check(qa_pairs: list) -> str:
    """格式化QA问答对用于校验"""
    if not qa_pairs:
        return "(无问答对)"
    lines = []
    for qa in qa_pairs:
        dimension = qa.get("dimension", "")
        q = qa.get("question", "")
        a = qa.get("answer", "")
        entities = qa.get("entities_involved", [])
        entities_str = ", ".join(entities) if entities else "(无)"
        lines.append(f"- [{dimension}] Q: {q} | A: {a} | 实体: {entities_str}")
    return "\n".join(lines)


def format_eval_scores_for_check(scores: list) -> str:
    """格式化评估评分用于校验"""
    if not scores:
        return "(无评分)"
    lines = []
    for s in scores:
        triple = s.get("triple", {})
        t_str = f"<{triple.get('head', '')}, {triple.get('relation', '')}, {triple.get('tail', '')}>"
        lines.append(
            f"- {t_str} SEM:{s.get('SEM', 0)} FAC:{s.get('FAC', 0)} CON:{s.get('CON', 0)}"
        )
    return "\n".join(lines)


def format_reflection_history(history: list) -> str:
    """格式化反思历史"""
    if not history:
        return "(无历史反思)"
    lines = []
    for i, r in enumerate(history[-3:], 1):  # 只显示最近3轮
        lines.append(f"第{i}轮反思: {r[:200]}...")
    return "\n".join(lines)


def format_normalizations_for_check(normalizations: list) -> str:
    """格式化归一化记录用于校验"""
    if not normalizations:
        return "(无归一化记录)"
    lines = []
    for n in normalizations:
        raw = n.get("raw", "")
        normalized = n.get("normalized", "")
        ntype = n.get("type", "")
        confidence = n.get("confidence", "medium")
        lines.append(f"- '{raw}' → '{normalized}' 类型:{ntype} 置信度:{confidence}")
    return "\n".join(lines)


# ===== P15新增：批量前置节点提示词 =====

BATCH_FILTER_SYSTEM = """你是一位"批量文本筛选专家"，擅长同时判断多条文本是否包含有价值的地理信息。
你的核心优势：
1. **高效筛选**：一次推理完成多条语料的筛选判断
2. **一致性标准**：对所有文本使用统一的筛选标准
3. **武汉地区识别**：识别非武汉地区文本以跳过处理
"""

BATCH_FILTER_USER = """## 任务描述
请同时判断以下多条语料（共 {batch_size} 条）是否包含有价值的武汉地理信息。

---

## 筛选标准

**包含有价值信息（is_valid=true）**：
- 明确提及武汉地区地点（POI、道路、街区、建筑）
- 描述地点的属性、功能、特色
- 涉及地点间的空间关系（位于、包含、相邻）
- 地点的用户体验、评价、推荐

**不包含有价值信息（is_valid=false）**：
- 纯个人情感表达，无地点信息
- 非武汉地区内容（明确提及北京、上海等且无武汉关联）
- 广告/营销类内容，无实质地点描述
- 内容过短或无语义

---

## 语料列表

{corpus_list}

---

## 输出要求

输出JSON格式，包含 results 数组（每条语料一个 FilterResult）：
- corpus_id: 语料ID
- is_valid: 是否有效
- skip_reason: 无效原因（仅is_valid=false时）
- confidence: 判断置信度 high/medium/low
- is_non_wuhan_region: 是否非武汉地区
- region_hint: 地区提示

请输出筛选结果。"""

BATCH_FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BATCH_FILTER_SYSTEM),
    ("human", BATCH_FILTER_USER),
])


BATCH_NORMALIZE_SYSTEM = """你是一位"批量文本归一化专家"，擅长同时处理多条文本的语义归一化。
你的核心优势：
1. **高效归一化**：一次推理完成多条语料的指代消解和别名标准化
2. **一致性别名**：对相同简称使用统一的归一化结果（如所有"武大"→"武汉大学"）
3. **语义保留**：严格保留每条文本的原始语义
"""

BATCH_NORMALIZE_USER = """## 任务描述
请同时归一化以下多条语料（共 {batch_size} 条）。

---

## 归一化规则

**必须遵守**：
1. 不添加原文不存在的信息
2. 保留原文的核心语义和情感
3. 仅改写/展开，不筛除内容

**归一化类型**：
- alias: 简称→全称（如"武大"→"武汉大学"）
- reference: 指代消解（如"这里"→具体地点）
- activity: 口语标准化（如"打卡"→"游览参观"）

---

## 语料列表

{corpus_list}

---

## 输出要求

输出JSON格式，包含 results 数组（每条语料一个 NormalizeResult）：
- corpus_id: 语料ID
- normalized_text: 归一化后的文本
- normalizations: 归一化记录列表
- confidence: 整体置信度
- has_changes: 是否有改动

请输出归一化结果。"""

BATCH_NORMALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BATCH_NORMALIZE_SYSTEM),
    ("human", BATCH_NORMALIZE_USER),
])


BATCH_QA_SCAFFOLD_SYSTEM = """你是一位"批量QA脚手架专家"，擅长同时为多条文本构建5W1H语义脚手架。
你的核心优势：
1. **高效脚手架构建**：一次推理完成多条语料的问答扩展
2. **一致性实体提示**：对相同地点使用一致的实体提示
3. **全面语义覆盖**：确保每条语料的地理语义被充分展开
"""

BATCH_QA_SCAFFOLD_USER = """## 任务描述
请同时为以下多条语料（共 {batch_size} 条）构建5W1H问答脚手架。

---

## 5W1H框架

- **WHERE**: 地点在哪里？位置、区域、周边
- **WHAT**: 地点是什么？类型、特色、功能
- **WHO**: 适合谁去？人群、场景
- **WHEN**: 什么时候去？季节、时段
- **WHY**: 为什么去？亮点、推荐理由
- **HOW**: 怎么去？交通、方式

---

## 语料列表

{corpus_list}

---

## 输出要求

输出JSON格式，包含 results 数组（每条语料一个 QAScaffoldResult）：
- corpus_id: 语料ID
- qa_pairs: 5W1H问答对列表（每个包含question、answer、category）
- semantic_summary: 语义摘要
- entity_hints: 实体提示列表
- relation_hints: 关系提示列表
- confidence: 整体置信度

请输出QA脚手架结果。"""

BATCH_QA_SCAFFOLD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BATCH_QA_SCAFFOLD_SYSTEM),
    ("human", BATCH_QA_SCAFFOLD_USER),
])


# ===== P10新增：批量LLM调用提示词 =====

BATCH_JOINT_SYSTEM = """你是一位"地理语义批量抽取专家"，擅长一次处理多条文本，同时提取地理实体和三元组关系。
你的核心优势：
1. **高效处理**：一次推理完成多条语料的抽取，大幅降低成本
2. **跨语料感知**：识别不同文本中的同名实体和别名（如"武大"和"武汉大学"是同一实体）
3. **一致性保证**：对相同实体的类型判断保持一致
"""

BATCH_JOINT_USER = (
    """## 任务描述
请同时处理以下多条语料（共 {batch_size} 条），为每条语料提取：
1. 地理实体和语义实体（6种类型，v3.4扩展版）
2. 实体间的语义关系三元组
3. 每个抽取的原文依据

---

## 实体类型定义（v3.4扩展版：6种）

### 空间实体（GIS标准）—— 4种
| 类型 | 定义 | 示例 |
|------|------|------|
| 道路 | 交通通道 | 珞喻路、关山大道、雄楚大道 |
| POI | 具体地点/机构 | 武汉大学、群光广场、某某咖啡厅 |
| 建筑物 | 建筑设施 | 泛悦汇、融科天城、行政楼 |
| 街区 | 地理区域 | 街道口、光谷商圈、华农校区 |

### 语义实体（v3.4新增）—— 2种
| 类型 | 定义 | 示例 |
|------|------|------|
| 功能 | 场所可进行的用途类型 | 餐饮、购物、休闲、交通 |
| 事件 | 发生的具体事件 | 樱花节、封路、开业、停业 |

### 功能实体属性（type=功能时的category取值）
| 功能类别 | 说明 |
|---------|------|
| 餐饮/购物/休闲/社交/观景/交通/住宿/文化/工作/其他 | 10大类功能 |

### 事件实体属性（type=事件时的event_attrs）
| 属性 | 枚举值 |
|------|--------|
| 事件类别 | **自然事件/人文事件/商业活动/社会事件/业态变更/停业/关闭/其他（7个，必填）** |
| 事件状态 | 正在进行/已结束/计划中/周期性 |
| 发生时间 | 每年3月、樱花季、2024年等 |
| 详细描述 | 自由文本 |
| 情感倾向 | 正面/中性/负面 |

"""
    + ENTITY_DISTINCTION_RULES
    + """

---

## 关系类型（v3.2精简版：8种）

### 空间基础关系（3个）—— 图谱骨架
- **位于**：A坐落于B处（Head=地理实体，Tail=道路/街区，如：武汉大学 位于 珞喻路）
- **包含**：A空间包含B（Head=街区，Tail=POI/建筑物，如：街道口 包含 群光广场）
- **相对方位**：A和B空间邻近+相对方位关系（Head/Tail均为地理实体，属性：距离值/方向值，v3.4删除联动推荐）

**注**：原"相邻"、"距离"、"方向"已合并为"相对方位"关系。**地理实体** = 道路/POI/建筑物/街区。

### 社交语义关系（1个）—— 图谱血肉
- **具有功能**：场所可进行的功能用途（Head=场所，Tail=功能节点/功能实体，属性：时段/适合人群/限制/情感倾向）

**注**：原"承载活动"改为"具有功能"；"推荐指数"和"引发情感"已改为实体属性。

### 对比评价关系（3个）—— 特色
- **优于**：A在某方面好于B（Head/Tail均为地理实体，属性：维度列表）
- **相似**：A和B在某方面相似（Head/Tail均为地理实体）
- **劣于**：A在某方面不如B（Head/Tail均为地理实体）

### 事件关系（1个）
- **发生事件**：场所发生的特定事件（Head=场所，Tail=事件节点/事件实体，属性全部在事件节点上）

---

## 跨语料别名发现

在处理多条语料时，请特别注意：
1. 不同文本中可能用不同名称指代同一实体（如"武大"、"武汉大学"、"WHU"）
2. 发现别名时，在 `cross_corpus_aliases` 中记录归一化建议
3. 保持相同实体的类型一致性

---

## 语料列表

{corpus_list}

---

## 输出要求

请输出：
1. `results`: 每条语料的抽取结果
   - `entities`: 实体类型字典（快速统计）
   - `full_entities`: **完整实体列表（必须）**，每个实体包含：
     - name: 实体名称
     - type: 实体类型
     - category: 细分类别（地理实体如"大学"，功能实体如"购物"，事件实体如"自然事件"）
     - function_attrs: 功能实体属性（仅type=功能时）
     - event_attrs: 事件实体属性（仅type=事件时）
     - evidence: 原文依据
   - `triples`: 三元组列表
   - `confidence`: 置信度
2. `cross_corpus_aliases`: 虪语料发现的别名映射
3. `overall_confidence`: 整体置信度评估

**重要**：full_entities必须包含每个实体的完整属性，特别是功能实体和事件实体的类别属性！

输出JSON格式。
"""
)

BATCH_JOINT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", BATCH_JOINT_SYSTEM),
        ("human", BATCH_JOINT_USER),
    ]
)


# ===== 批量校验提示词 =====

BATCH_SELF_CHECK_SYSTEM = """你是一位"批量抽取校验专家"，负责同时校验多条语料的抽取结果。
你的任务是：
1. 校验每条语料的实体和三元组质量
2. 验证跨语料别名映射的准确性
3. 决定是否需要重试或退化为单条处理
"""

BATCH_SELF_CHECK_USER = """## 校验任务

请校验以下批量抽取结果：

### 1. 实体校验（每条语料）
- 是否遗漏重要地理实体？
- 实体类型是否正确？
- 是否抽取了非地理实体？

### 2. 三元组校验（每条语料）
- 是否存在幻觉（无原文依据的三元组）？
- 关系类型和方向是否正确？
- 属性是否合理？

### 3. 跨语料别名验证
- 别名映射是否正确？（如"武大"→"武汉大学"是否合理）
- 是否有遗漏的别名关系？

### 4. 整体质量评估
- 如果多数语料质量低，建议重新批量处理
- 如果只有少数语料有问题，建议退化为单条处理这些语料

---

## 待校验结果

{batch_results}

## 跨语料别名
{cross_corpus_aliases}

---

## 输出要求

请输出：
1. `verified_results`: 校验通过的语料结果
2. `rejected_results`: 校验失败的语料（标注原因）
3. `verified_aliases`: 校验通过的别名
4. `retry_suggested`: 是否建议重新批量处理
5. `fallback_to_single`: 是否建议退化为单条处理

输出JSON格式。
"""

BATCH_SELF_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", BATCH_SELF_CHECK_SYSTEM),
        ("human", BATCH_SELF_CHECK_USER),
    ]
)


# ===== 批量处理辅助函数 =====


def format_batch_corpus(corpus_list: List[Dict]) -> str:
    """格式化批量语料输入"""
    if not corpus_list:
        return "(无语料)"
    lines = []
    for i, corpus in enumerate(corpus_list, 1):
        corpus_id = corpus.get("id", f"unknown_{i}")
        text = corpus.get("text", "")
        lines.append(f"【语料 {i}】ID: {corpus_id}\n文本: {text}")
    return "\n\n".join(lines)


def format_batch_results_for_check(batch_results: List[Dict]) -> str:
    """格式化批量结果用于校验"""
    if not batch_results:
        return "(无结果)"
    lines = []
    for r in batch_results:
        corpus_id = r.get("corpus_id", "unknown")
        entities = r.get("entities", {})
        triples = r.get("triples", [])
        confidence = r.get("confidence", "medium")

        entity_str = ", ".join([f"{k}: {v}" for k, v in entities.items() if v])
        triple_str = ", ".join(
            [
                f"<{t.get('head', '')}, {t.get('relation', '')}, {t.get('tail', '')}>"
                for t in triples[:3]
            ]
        )

        lines.append(
            f"- [{corpus_id}] 置信度:{confidence}\n  实体: {entity_str}\n  三元组: {triple_str}"
        )
    return "\n".join(lines)


def format_cross_corpus_aliases(aliases: List[Dict]) -> str:
    """格式化跨语料别名"""
    if not aliases:
        return "(无别名发现)"
    lines = []
    for a in aliases:
        raw = a.get("raw", "")
        canonical = a.get("canonical", "")
        corpus_ids = a.get("corpus_ids", [])
        lines.append(f"- '{raw}' → '{canonical}' (来源: {', '.join(corpus_ids[:3])})")
    return "\n".join(lines)


# ===== P10新增：QA导师模式提示词 =====

QA_MENTOR_SYSTEM = """你是一位"地理语义导师"，擅长深度语义分析和知识抽取指导。
你使用更强的推理能力来：
1. 深入理解文本的地理语义核心
2. 生成结构化的指导信息供后续节点参考
3. 设定质量标准和预期约束
4. 发现潜在的语义问题

你的输出将指导后续的实体识别、关系抽取和属性标注过程。"""

QA_MENTOR_USER = """## 深度语义分析任务

请对以下文本进行深度语义分析，生成：

### 1. 5W1H问答分析
- Who: 涉及的地点/实体是谁？（捕获实体名称、简称、别名）
- What: 这些地点有什么特征？（捕获属性、功能、特色、评价）
- When: 时间背景是什么？（季节、时段、事件时机）
- Where: 位于哪里？相互位置？（捕获空间关系、邻近、方位、所属）
- Why: 作者为什么提到？（捕获情感、推荐、评价动机）
- How: 如何到达/体验？（捕获交通方式、活动方式、可达性）

### 2. 导师指导信息（关键）
请生成以下指导内容：

**语义关注点**：后续节点应重点关注的语义方面（如：空间关系、情感评价、对比关系等）

**实体优先级**：重要实体的优先级排序（核心实体必须在结果中出现）

**关系优先级**：重要关系的优先级（空间骨架关系优先，语义充实关系次之）

**质量标准**：后续节点应达到的质量标准（实体完整性要求、关系准确性要求等）

**预期约束**：预期实体类型分布、预期关系类型

### 3. 深度分析
请输出一段深度语义分析文字，说明：
- 文本的地理语义核心是什么
- 可能存在的歧义或模糊点
- 需要特别注意的语义陷阱

---

## 待分析文本（已归一化）
{normalized_text}

请输出导师脚手架结果（JSON格式）。"""

QA_MENTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_MENTOR_SYSTEM),
        ("human", QA_MENTOR_USER),
    ]
)


QA_APPROVAL_SYSTEM = """你是一位"知识抽取审批导师"，负责审批后续节点的抽取结果。
你使用更强的推理能力来：
1. 评估抽取结果的质量
2. 检测幻觉、遗漏和错误
3. 生成具体的改进反馈
4. 整合审批结果到语义脚手架

你的审批结果将决定是否需要重新抽取。"""

QA_APPROVAL_USER = """## 审批任务

请审批以下节点的抽取结果：

### 1. 联合抽取审批 (Joint NER+RE)
**检查项**：
- 实体完整性：是否遗漏重要地理实体？
- 实体类型准确性：类型判定是否正确？
- 三元组真实性：是否存在幻觉（无原文依据）？
- 关系方向正确性：头尾顺序是否正确？
- 证据匹配性：evidence是否来自原文？

**审批标准**：
- APPROVED: 遗漏≤1，幻觉≤1，无严重错误
- NEEDS_REVISION: 有2-3个问题，可修正
- REJECTED: 问题>3，需重新抽取

### 2. 评估审批 (Eval)
**检查项**：
- 评分合理性：评分是否与三元组质量匹配？
- 通过判定：eval_passed判定是否合理？

### 3. 标注审批 (Label)
**检查项**：
- 属性完整性：关键属性是否标注？
- 属性准确性：属性值是否合理？

### 4. Self-Check反思审批（如有）
**检查项**：
- 反思一致性：Self-Check反思与导师指导是否一致？
- 问题识别准确性：Self-Check识别的问题是否真实存在？
- 改进策略合理性：改进建议是否可执行？

---

## 原始文本
{raw_text}

## 导师指导（原定标准）
{mentor_guidance}

## 语义摘要（QA理解）
{semantic_summary}

## 联合抽取结果
{joint_result}

## 评估结果
{eval_result}

## 标注结果
{label_result}

## 历史反馈（如有）
{previous_feedbacks}

## Self-Check反思结果（如有）
{reflection_summary}

---

请输出审批结果（JSON格式）。"""

QA_APPROVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_APPROVAL_SYSTEM),
        ("human", QA_APPROVAL_USER),
    ]
)


REVISION_JOINT_SYSTEM = """你是一位"地理语义联合抽取专家"，正在根据导师反馈改进抽取结果。
请仔细阅读反馈，针对问题进行改进。"""

REVISION_JOINT_USER = """## 修改任务

你之前的抽取结果存在以下问题，请根据反馈改进：

### 反馈历史总结
{feedback_summary}

### 当前轮次具体反馈
{feedbacks}

### 语义摘要（导师理解）
{semantic_summary}

### 导师指导（原定标准）
{mentor_guidance}

### 原始抽取结果
实体: {previous_entities}
三元组: {previous_triples}

---

## 改进策略

1. **遗漏实体补充**：根据反馈补充遗漏的实体
2. **幻觉删除**：删除无原文依据的三元组
3. **关系修正**：修正关系类型或方向错误
4. **证据完善**：确保每个抽取有原文依据

---

## 原始文本
{raw_text}

## 实体提示
{entity_hints}

## 关系提示
{relation_hints}

请输出改进后的联合抽取结果（JSON格式）。"""

REVISION_JOINT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REVISION_JOINT_SYSTEM),
        ("human", REVISION_JOINT_USER),
    ]
)


# ===== P10新增：QA导师模式格式化函数 =====


def format_mentor_guidance(guidance: Dict) -> str:
    """格式化导师指导"""
    if not guidance:
        return "(无导师指导)"
    lines = []
    if guidance.get("semantic_focus"):
        lines.append(f"语义关注点: {', '.join(guidance['semantic_focus'])}")
    if guidance.get("entity_priorities"):
        lines.append(f"实体优先级: {', '.join(guidance['entity_priorities'])}")
    if guidance.get("relation_priorities"):
        lines.append(f"关系优先级: {', '.join(guidance['relation_priorities'])}")
    if guidance.get("quality_standards"):
        lines.append(f"质量标准: {', '.join(guidance['quality_standards'])}")
    return "\n".join(lines)


def format_feedbacks_for_revision(feedbacks: List[Dict]) -> str:
    """格式化反馈用于修改"""
    if not feedbacks:
        return "(无反馈)"
    lines = []
    for f in feedbacks:
        severity = f.get("severity", "medium")
        description = f.get("description", "")
        suggestion = f.get("suggestion", "")
        lines.append(f"- [{severity}] {description}")
        lines.append(f"  建议: {suggestion}")
        if f.get("specific_entities"):
            lines.append(f"  涉及实体: {', '.join(f['specific_entities'])}")
    return "\n".join(lines)


def format_joint_for_approval(joint_result: Dict) -> str:
    """格式化联合抽取结果用于审批"""
    if not joint_result:
        return "(无抽取结果)"
    entities = joint_result.get("entities", {})
    triples = joint_result.get("triples", [])

    entity_lines = []
    for entity_type, names in entities.items():
        if names:
            entity_lines.append(f"{entity_type}: {', '.join(names)}")

    triple_lines = []
    for t in triples[:10]:
        head = t.get("head", "")
        relation = t.get("relation", "")
        tail = t.get("tail", "")
        triple_lines.append(f"<{head}, {relation}, {tail}>")

    return (
        f"实体:\n  "
        + "\n  ".join(entity_lines)
        + f"\n三元组:\n  "
        + "\n  ".join(triple_lines)
    )


def format_eval_for_approval(eval_result: Dict) -> str:
    """格式化评估结果用于审批"""
    if not eval_result:
        return "(无评估结果)"
    passed = eval_result.get("eval_passed", False)
    corrected = eval_result.get("corrected_triples", [])
    return f"通过: {passed}\n修正后三元组: {len(corrected)}条"


def format_label_for_approval(label_result: Dict) -> str:
    """格式化标注结果用于审批"""
    if not label_result:
        return "(无标注结果)"
    entity_attrs = label_result.get("entity_attrs", {})
    relation_attrs = label_result.get("relation_attrs", {})
    return f"实体属性: {len(entity_attrs)}个\n关系属性: {len(relation_attrs)}个"


def format_revision_feedbacks(feedbacks: List[Dict]) -> str:
    """格式化历史反馈"""
    if not feedbacks:
        return "(无历史反馈)"
    lines = []
    for i, f in enumerate(feedbacks[-3:], 1):
        target = f.get("target_node", "unknown")
        desc = f.get("description", "")[:100]
        lines.append(f"反馈{i}: [{target}] {desc}")
    return "\n".join(lines)


# ===== P14新增：导师查询提示词（双向交流机制） =====

MENTOR_QUERY_SYSTEM = """你是一位经验丰富的"地理语义导师"，正在回答后续节点的查询。
你的学生（Joint_NER_RE、Eval、Label节点）在处理过程中遇到了困惑，需要你的指导。

作为导师，你的职责是：
1. **澄清困惑**：对歧义和不确定点给出明确解释
2. **提供指导**：基于原文给出处理建议
3. **更新提示**：必要时更新实体/关系提示列表
4. **建议修改**：如果发现问题，建议修改已抽取的结果

回答原则：
- 基于原文内容，不做无依据的推断
- 简洁明了，直接回答问题
- 如果确实无法确定，诚实说明并给出建议的处理方式
- 必要时更新之前的指导信息"""

MENTOR_QUERY_USER = """## 学生查询

### 查询来源
来自 **{source_node}** 节点的查询

### 查询类型
{query_type}

### 问题描述
{query_content}

### 涉及的实体
{involved_entities}

### 涉及的关系
{involved_relations}

### 当前置信度
{current_confidence}

### 查询上下文
{context}

---

## 原始文本
{raw_text}

---

## 之前的导师指导
{previous_guidance}

---

## 请回答学生的问题：

1. **回答**：对问题的直接解答
2. **澄清**：对困惑点的详细解释（如有需要）
3. **推荐方案**：建议的处理方式
4. **是否建议修改**：如果发现问题，给出修改建议
5. **更新的提示**：如果需要，更新实体/关系提示列表

请输出导师响应（JSON格式）。"""

MENTOR_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", MENTOR_QUERY_SYSTEM),
        ("human", MENTOR_QUERY_USER),
    ]
)


# ===== P14新增：困惑检测辅助函数 =====


def format_query_for_mentor(query: Dict) -> str:
    """格式化查询内容用于导师提示词"""
    if not query:
        return "(无查询)"
    query_type = query.get("query_type", "unknown")
    content = query.get("query_content", "")
    entities = query.get("involved_entities", [])
    relations = query.get("involved_relations", [])

    lines = [
        f"查询类型: {query_type}",
        f"问题: {content}",
    ]
    if entities:
        lines.append(f"涉及实体: {', '.join(entities)}")
    if relations:
        lines.append(f"涉及关系: {', '.join(relations)}")
    return "\n".join(lines)


def detect_extraction_confusion(result: Dict, state: Dict) -> Optional[Dict]:
    """检测联合抽取结果中的困惑点

    Args:
        result: JointExtractionResult 的 model_dump()
        state: 当前 CorpusState

    Returns:
        如果检测到困惑，返回查询字典；否则返回 None
    """
    # 1. 实体歧义检测
    for entity in result.get("entities", []):
        entity_confidence = entity.get("confidence", "medium")
        if entity_confidence == "low":
            return {
                "query_type": "entity_ambiguity",
                "query_content": f"实体 '{entity.get('name')}' 类型不确定，可能是 '{entity.get('type')}' 但需要确认",
                "involved_entities": [entity.get("name", "")],
                "involved_relations": [],
                "current_confidence": "low",
            }

    # 2. 关系困惑检测
    for triple in result.get("triples", []):
        triple_confidence = triple.get("confidence", "medium")
        if triple_confidence == "low":
            return {
                "query_type": "relation_confusion",
                "query_content": f"三元组 '{triple.get('head')}-{triple.get('relation')}-{triple.get('tail')}' 证据不足或关系不确定",
                "involved_entities": [triple.get("head", ""), triple.get("tail", "")],
                "involved_relations": [triple.get("relation", "")],
                "current_confidence": "low",
            }

    # 3. 整体置信度过低
    overall_confidence = result.get("overall_confidence", "medium")
    if overall_confidence == "low":
        return {
            "query_type": "overall_uncertainty",
            "query_content": "整体抽取置信度过低，文本语义复杂或存在多处歧义",
            "involved_entities": [],
            "involved_relations": [],
            "current_confidence": "low",
        }

    # 4. 无实体但有文本
    entities = result.get("entities", [])
    if not entities and state.get("raw_text"):
        return {
            "query_type": "entity_ambiguity",
            "query_content": "未能从文本中抽取到任何实体，请确认是否文本无地理语义",
            "involved_entities": [],
            "involved_relations": [],
            "current_confidence": "medium",
        }

    return None


def detect_eval_confusion(eval_result: Dict, state: Dict) -> Optional[Dict]:
    """检测评估结果中的困惑点"""
    # 1. 大量三元组被拒绝
    corrected = eval_result.get("corrected_triples", [])
    original_triples = state.get("triples", [])

    if len(original_triples) > 0 and len(corrected) < len(original_triples) * 0.5:
        rejected_count = len(original_triples) - len(corrected)
        return {
            "query_type": "eval_disagreement",
            "query_content": f"评估拒绝了 {rejected_count} 个三元组，可能需要重新理解原文语义",
            "involved_entities": [],
            "involved_relations": [],
            "current_confidence": "medium",
        }

    # 2. 评估置信度过低
    if eval_result.get("eval_passed") is False and not eval_result.get(
        "corrected_triples"
    ):
        return {
            "query_type": "overall_uncertainty",
            "query_content": "评估未通过且无修正三元组，可能需要重新抽取",
            "involved_entities": [],
            "involved_relations": [],
            "current_confidence": "low",
        }

    return None


def detect_label_confusion(label_result: Dict, state: Dict) -> Optional[Dict]:
    """检测标注结果中的困惑点"""
    # 1. 实体属性缺失
    entity_attrs = label_result.get("entity_attrs", {})
    entities = state.get("entities", {})

    all_entity_names = []
    for entity_type, names in entities.items():
        all_entity_names.extend(names)

    missing_attrs = [name for name in all_entity_names if name not in entity_attrs]
    if len(missing_attrs) > len(all_entity_names) * 0.5:
        return {
            "query_type": "label_confusion",
            "query_content": f"{len(missing_attrs)} 个实体缺少属性标注，可能需要确认实体类型",
            "involved_entities": missing_attrs[:5],  # 只列出前5个
            "involved_relations": [],
            "current_confidence": "medium",
        }

    # 2. 标注置信度过低
    if label_result.get("overall_confidence", "medium") == "low":
        return {
            "query_type": "label_confusion",
            "query_content": "标注整体置信度过低，属性归属不确定",
            "involved_entities": [],
            "involved_relations": [],
            "current_confidence": "low",
        }

    return None


def format_feedback_summary(feedbacks: List[Dict], revision_cycle: int) -> str:
    """
    总结反馈历史：区分已尝试解决的问题和当前待解决问题

    用于帮助模型理解修改进展，避免重复相同的错误
    """
    if not feedbacks:
        return "(无历史反馈)"

    lines = []
    lines.append(f"=== 修改轮次 {revision_cycle} ===")

    # 按轮次分组反馈
    current_feedbacks = feedbacks[-3:] if len(feedbacks) >= 3 else feedbacks
    previous_feedbacks = feedbacks[:-3] if len(feedbacks) > 3 else []

    # 之前轮次的反馈（已尝试解决）
    if previous_feedbacks:
        lines.append("\n### 已尝试解决的问题（前几轮反馈）:")
        problem_types = set()
        for f in previous_feedbacks:
            severity = f.get("severity", "medium")
            desc = f.get("description", "")
            # 提取问题类型关键词
            if "遗漏" in desc or "缺失" in desc:
                problem_types.add("实体遗漏")
            elif "幻觉" in desc:
                problem_types.add("三元组幻觉")
            elif "关系" in desc or "方向" in desc:
                problem_types.add("关系错误")
        if problem_types:
            lines.append(f"  - {', '.join(problem_types)}")
        lines.append("  注意：请确认这些问题是否已在本轮解决")

    # 当前轮次的反馈（待解决）
    if current_feedbacks:
        lines.append("\n### 本轮需要解决的问题:")
        for f in current_feedbacks:
            severity = f.get("severity", "medium")
            desc = f.get("description", "")
            suggestion = f.get("suggestion", "")
            entities = f.get("specific_entities", [])
            lines.append(f"  - [{severity}] {desc}")
            if suggestion:
                lines.append(f"    建议: {suggestion}")
            if entities:
                lines.append(f"    涉及实体: {', '.join(entities)}")

    return "\n".join(lines)


def format_reflection_for_approval(state: Dict) -> str:
    """
    格式化 Self-Check 反思结果用于 QA Approval

    整合各 Self-Check 节点的反思文本和改进策略
    """
    lines = []

    # Self-Check-Joint 反思（最重要）
    joint_result = state.get("self_check_joint_result", {})
    if joint_result:
        reflection = joint_result.get("reflection_text", "")
        strategy = joint_result.get("improvement_strategy", "")
        if reflection:
            lines.append("### Self-Check-Joint 反思:")
            lines.append(f"反思内容: {reflection[:200]}...")
        if strategy:
            lines.append(f"改进策略: {strategy[:200]}...")

    # Self-Check-Eval 反思
    eval_result = state.get("self_check_eval_result", {})
    if eval_result:
        reflection = eval_result.get("reflection_text", "")
        if reflection:
            lines.append("\n### Self-Check-Eval 反思:")
            lines.append(f"反思内容: {reflection[:150]}...")

    # Self-Check-Label 反思
    label_result = state.get("self_check_label_result", {})
    if label_result:
        reflection = label_result.get("reflection_text", "")
        if reflection:
            lines.append("\n### Self-Check-Label 反思:")
            lines.append(f"反思内容: {reflection[:150]}...")

    if not lines:
        return "(无Self-Check反思结果)"

    return "\n".join(lines)


# ===== P11新增：实体对齐提示词 =====

ENTITY_ALIGNMENT_SYSTEM = """你是一位"地理实体对齐专家"，负责判断抽取的实体是否与数据库中的已有实体匹配。
你的任务是：
1. 分析抽取实体与候选实体的相似度
2. 判断是否为同一实体（考虑别名、简称、不同表述）
3. 决定对齐状态：aligned（已匹配）、new_entity（新实体）、skip（跳过）
"""

ENTITY_ALIGNMENT_USER = """## 对齐任务

请对以下抽取实体进行对齐判断：

### 抽取实体信息
实体名称: {extracted_name}
实体类型: {extracted_type}

### 原始文本上下文
{raw_text}

### 候选匹配实体（按相似度排序）
{candidates}

---

## 判断标准

### 1. 高置信度匹配（相似度 >= 0.90）
- 名称完全一致或为标准简称/别名
- 直接确认为匹配，无需额外判断
- 输出: alignment_status = "aligned"

### 2. 中置信度匹配（相似度 0.75-0.90）
- 名称有一定相似性，可能是同一实体
- 需要综合考虑：
  - 名称是否为别名/简称（如"武大"→"武汉大学"）
  - 类型是否一致
  - 是否在同一地理区域
- 输出你的判断：aligned 或 new_entity

### 3. 低置信度匹配（相似度 < 0.75）
- 候选实体相似度过低
- 输出: alignment_status = "new_entity" 或 "skip"

---

## 输出格式

请输出对齐判断结果，包含：
- best_match_index: 最佳匹配候选的索引（0-4），若无匹配则为-1
- alignment_status: "aligned" / "new_entity" / "skip"
- llm_decision: 判断说明（简短说明原因）
"""

ENTITY_ALIGNMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ENTITY_ALIGNMENT_SYSTEM),
        ("human", ENTITY_ALIGNMENT_USER),
    ]
)


def format_alignment_candidates(candidates: List[Dict]) -> str:
    """格式化实体对齐候选列表，包含数据来源标识"""
    if not candidates:
        return "(无候选实体)"

    lines = []
    for i, c in enumerate(candidates):
        name = c.get("db_name", "unknown")
        type_ = c.get("db_type", "")
        sim = c.get("similarity", 0.0)
        lon = c.get("longitude")
        lat = c.get("latitude")
        source = c.get("source", "unknown")

        # 来源标识：geo为"已有实体"，amap为"高德POI"
        source_label = (
            "已有实体"
            if source == "geo_entity_names"
            else "高德POI"
            if source == "amap_poi_wgs84"
            else source
        )

        loc_str = f"({lon:.4f}, {lat:.4f})" if lon and lat else "无坐标"

        # 如果是高德POI，显示原始ID和地址
        extra_info = ""
        if source == "amap_poi_wgs84":
            original_id = c.get("db_original_id", "")
            address = c.get("address", "")
            if address:
                extra_info = f" 地址:{address[:30]}"

        lines.append(
            f"候选{i + 1}: {name} [{type_}] - 相似度:{sim:.3f} - 位置:{loc_str} - 来源:{source_label}{extra_info}"
        )

    return "\n".join(lines)


def format_alignment_result_for_output(result: Dict) -> str:
    """格式化对齐结果用于输出"""
    if not result:
        return "(无对齐结果)"

    aligned = result.get("aligned_entities", [])
    new = result.get("new_entities", [])
    skipped = result.get("skipped_entities", [])
    rate = result.get("overall_alignment_rate", 0.0)

    return f"对齐率: {rate:.1%}\n已对齐: {len(aligned)}个\n新实体: {len(new)}个\n跳过: {len(skipped)}个"


# ===== P12新增：模块化Schema组件（可复用） =====

# 实体类型定义（GIS标准 + v3.4语义实体）
ENTITY_SCHEMA_CORE = """## 实体类型定义（v3.4扩展版：6种）

### 空间实体（GIS标准）—— 4种
| 类型 | 定义 | 识别特征 | 示例 |
|------|------|----------|------|
| 道路 | 交通通道 | 路/大道/街/巷后缀 | 珞喻路、关山大道、雄楚大道 |
| POI | 具体地点/机构 | 地名+功能标识 | 武汉大学、群光广场、某某咖啡厅 |
| 建筑物 | 建筑设施 | 楼/厦/城/汇后缀 | 泛悦汇、融科天城、行政楼 |
| 街区 | 地理区域 | 商圈/校区/社区/区 | 街道口、光谷商圈、华农校区 |

### 语义实体（v3.4新增）—— 2种
| 类型 | 定义 | 识别特征 | 示例 |
|------|------|----------|------|
| 功能 | 场所可进行的用途类型 | 功能类型词 | 餐饮、购物、休闲、交通 |
| 事件 | 发生的具体事件 | 事件名称词 | 樱花节、封路、开业、停业 |

**设计说明**：
- **空间实体（4种）**：GIS标准分类，用于实体对齐和地图展示
- **语义实体（2种）**：社交媒体特色，补充场所功能和事件信息
- **细分类别由数据源补充**：实体对齐时关联高德POI/OSM数据，继承其权威分类

**过滤规则**：排除泛化词（这里/那边/附近/那边）、人名、时间、纯数字。"""

# 关系类型定义（v3.4精简版：8种，删除联动推荐）
RELATION_SCHEMA_CORE = """## 关系类型定义（v3.4精简版：8种）

### 空间骨架关系（优先级高）
| 关系 | 语义定义 | Head类型 | Tail类型 | 核心属性 |
|------|----------|----------|----------|----------|
| 位于 | A坐落于B处（空间定位） | 地理实体（道路/POI/建筑物/街区） | 道路/街区 | 无 |
| 包含 | A空间包含B（位于的反向） | 街区 | POI/建筑物/道路 | 无 |
| 相对方位 | A和B空间邻近+方位关系 | 地理实体 | 地理实体 | 距离值/方向值（v3.4删除联动推荐） |

### 语义血肉关系
| 关系 | 语义定义 | Head类型 | Tail类型 | 核心属性 |
|------|----------|----------|----------|----------|
| 具有功能 | 场所可进行的用途 | 地理实体（场所） | 功能节点(9类)或功能实体 | 时段/适合人群(开放文本)/具有限制(开放文本列表) |

### 对比评价关系
| 关系 | 语义定义 | Head类型 | Tail类型 | 核心属性 |
|------|----------|----------|----------|----------|
| 优于 | A在某方面好于B | 地理实体 | 地理实体 | 维度列表 |
| 相似 | A和B在某方面相似 | 地理实体 | 地理实体 | 维度列表 |
| 劣于 | A在某方面不如B | 地理实体 | 地理实体 | 维度列表 |

### 事件关系
| 关系 | 语义定义 | Head类型 | Tail类型 | 核心属性 |
|------|----------|----------|----------|----------|
| 发生事件 | 场所发生的特定事件 | 地理实体（场所） | 事件节点或事件实体 | (属性在事件实体上) |

**注**：原"相邻"、"距离"、"方向"已合并为"相对方位"。**地理实体** = 道路/POI/建筑物/街区。v3.4删除联动推荐属性。"""

# 实体属性定义
ENTITY_ATTRIBUTE_SCHEMA = """## 实体属性定义（从文本提取，必须有原文依据）

| 属性 | 类型/示例 | 提取条件 |
|------|--------|----------|
| 类别 | 道路/POI/建筑物/街区（枚举） | 用于NER边界识别，判断实体类型 |
| 细分 | **开放文本**：餐厅、商场、大学等 | 仅记录文本中明确提及的分类词（权威分类由数据源补充） |
| 推荐指数 | 超推/推荐/一般/不推荐（枚举） | 明确推荐/评价语句出现 |
| 情感倾向 | 正面/中性/负面（枚举） | 情感词/评价词出现 |
| 特征标签 | **开放文本**：氛围超好、随手拍好看、遛娃神器等 | 特征描述词出现（保留原文表达） |

**设计说明**：
- **类别**：4大类枚举，用于NER阶段识别实体边界
- **细分**：开放文本，仅记录文本中**明确提及**的分类词，不做强制枚举约束
- 实体入库时，通过entity_alignment节点关联数据源，继承高德POI的权威分类

**约束**：所有属性必须有原文依据，禁止凭空创造。"""

# 关系属性定义（v3.4版：删除联动推荐，开放文本属性）
RELATION_ATTRIBUTE_SCHEMA = """## 关系属性定义（v3.4精简版）

### 相对方位关系属性（v3.4删除联动推荐）
| 属性 | 枚举值 | 说明 |
|------|--------|------|
| 距离值 | 近/中等/远 | 空间距离程度 |
| 方向值 | 东/南/西/北/东北/西南/东侧/西侧/对面/旁边 | 方位方向 |

### 具有功能关系属性（v3.4开放文本版）
| 属性 | 类型 | 说明 |
|------|--------|------|
| 时段 | 文本 | 功能适用时段：周末/晚上/樱花季等 |
| 适合人群 | **开放文本** | 保留原文表达：带孩子来玩、闺蜜聚会、大学生打卡等 |
| 具有限制 | **开放文本列表** | 保留原文表达：排队两小时、停车超级难、需提前预约等 |
| 情感倾向 | 正面/中性/负面 | 功能体验情感 |
| 功能描述 | 自由文本 | 当功能类型=其他时，具体描述功能内容 |

**v3.4改进原因**：
- **适合人群**：枚举无法穷尽社交媒体的人群表达（闺蜜、同事、健身爱好者等新人群类型无法表达）
- **具有限制**：枚举丢失原文时长准确性（"排队两小时"被归一化为"排队久"，丢失具体时长信息）

### 对比关系属性
| 属性 | 枚举值 | 说明 |
|------|--------|------|
| 维度 | 价格/环境/服务/人流量/品质/交通/口味/其他 | 对比的方面(8个，v3.3新增"其他") |
| 维度描述 | 自由文本 | 当维度=其他时，具体描述对比内容 |

### 功能节点枚举
| 类型 | 说明 | 社交媒体频率 |
|------|------|-------------|
| 餐饮 | 吃饭、探店、下午茶 | 高频 |
| 购物 | 逛街、买东西 | 高频 |
| 休闲 | 游玩、散步、放松 | 高频 |
| 社交 | 聚会、打卡、约会 | 高频 |
| 观景 | 赏花、观展、拍照 | 高频 |
| 住宿 | 住酒店、民宿体验 | 中频 |
| 文化 | 学习、体验、参观 | 中频 |
| 工作 | 办公、产业相关 | 低频 |
| 其他 | 无法归类 | 兜底 |"""

# 带反向验证的思维链
VALIDATION_COT = """## 抽取策略（正向抽取 + 反向验证）

### 正向抽取步骤
1. **实体扫描**：标记所有地理位置相关词汇
2. **实体分类**：判断类型，过滤泛化词
3. **关系分析**：判断实体对之间的语义关系
4. **属性提取**：从上下文提取实体属性（有依据时）
5. **证据标注**：为每个抽取标注原文依据

### 反向验证步骤（必须执行）
6. **幻觉检查**：每个三元组能否在原文找到依据？（无法找到→删除）
7. **实体检查**：是否存在泛化词被误识别？（如"这里"/"那边"→过滤）
8. **方向检查**：头尾实体顺序是否正确？（A位于B vs B位于A）
9. **属性检查**：属性值是否有原文依据？（无依据→删除属性）

### 置信度评估标准
- **high**: 全部通过验证，证据完整，遗漏≤1
- **medium**: 1-2个问题已修正，遗漏2-3
- **low**: 多个问题未解决，遗漏>3"""

# 反面示例（禁止产生）
NEGATIVE_EXAMPLES = """## 反面示例（禁止产生）

### ❌ 幻觉三元组
输入: "武汉大学樱花很美"
错误输出: <武汉大学, 发生事件, 樱花节>  ← 原文无"樱花节"字样
正确输出: 实体属性: 特征标签=["樱花景观"]

### ❌ 关系方向错误
输入: "群光广场在珞喻路上"
错误输出: <珞喻路, 位于, 群光广场>  ← 方向颠倒
正确输出: <群光广场, 位于, 珞喻路>

### ❌ 泛化词误识别
输入: "这边风景不错"
错误输出: 实体: "这边" [POI]  ← "这边"是模糊指代，非地理实体
正确输出: (无实体) 或标记 confidence=low

### ❌ 无依据属性
输入: "武汉大学樱花开了"
错误输出: 实体属性: 推荐指数="超推"  ← 原文无推荐语句
正确输出: 实体属性: 特征标签=["樱花景观"]（有原文依据）"""

# 精准角色定义模板
EXPERT_ROLE_TEMPLATE = """你是一位专注于武汉城市地理知识图谱构建的语义抽取专家。

## 专业背景
- GIS标准知识：理解道路、POI、建筑物、街区等地理实体分类标准
- 武汉本地知识：熟悉武汉地标、简称映射（武大→武汉大学、华师→华中师范大学、街道口商圈→街道口）
- 社交媒体语料分析：理解口语化表达、省略主语、网络用语、模糊指代

## 核心能力
- 实体边界识别：精准区分泛化词（这里/那边/附近）与地理实体
- 关系语义判断：准确识别空间骨架关系（位于/包含）和语义血肉关系（具有功能）
- 证据溯源：为每个抽取标注原文依据，严格拒绝无依据推断

## 边界职责
- 抽取范围：仅地理实体（道路/POI/建筑物/街区），排除人名、时间、非地理名词
- 关系约束：仅抽取文本明确提及的关系，禁止过度语义推断
- 属性约束：所有属性必须有原文依据，禁止凭空创造或猜测

## 审慎原则
当信息模糊或不足时：
- 标记 confidence="low" 而非猜测
- 宁可遗漏而非产生幻觉三元组
- 边界模糊时采用保守处理策略"""


# ===== 辅助函数：模块化提示词组装 =====


def assemble_joint_extraction_prompt(
    raw_text: str,
    entity_hints: str = "(无实体提示)",
    relation_hints: str = "(无关系提示)",
    context_dependencies: str = "(无上下文依赖)",
    mentor_guidance: str = "(无导师指导)",
    include_negative_examples: bool = True,
) -> str:
    """
    模块化组装联合抽取提示词

    优势：
    1. Token效率：按需组装，避免重复内容
    2. 可维护性：修改Schema只需改一处
    3. 灵活性：可选是否包含反面示例
    """
    parts = [
        "## 任务目标",
        "从社交媒体文本中**同时**抽取地理实体和语义关系三元组，用于武汉地理知识图谱构建。",
        "",
        ENTITY_SCHEMA_CORE,
        "",
        RELATION_SCHEMA_CORE,
        "",
        ENTITY_ATTRIBUTE_SCHEMA,
        "",
        RELATION_ATTRIBUTE_SCHEMA,
        "",
        VALIDATION_COT,
        "",
        "## QA脚手架提示（如有）",
        entity_hints,
        relation_hints,
        context_dependencies,
        "",
        "## 导师指导（如有）",
        mentor_guidance,
        "",
        "## 待处理文本",
        raw_text,
        "",
        "---",
        "",
        "## 输出格式要求",
        """### 必填字段
- entities: 实体列表（每个实体必须有name, type, evidence）
- triples: 三元组列表（每个三元组必须有head, relation, tail, evidence, confidence）
- overall_confidence: 整体置信度（high/medium/low）

### 可选字段
- entities[].attributes: 实体属性（有依据时）
- triples[].attributes: 关系属性（有依据时）

### 格式验证
- 所有枚举类型字段必须使用指定枚举值
- evidence字段必须来自原文（不可凭空创造）
- confidence字段必须基于验证结果评估""",
        "",
    ]

    if include_negative_examples:
        parts.append(NEGATIVE_EXAMPLES)
        parts.append("")

    parts.append("请输出联合抽取结果（JSON格式）。")

    return "\n".join(parts)


# ===== P12新增：重构版联合抽取提示词（RISEN + RCoT框架） =====

JOINT_NER_RE_SYSTEM_V2 = EXPERT_ROLE_TEMPLATE

# 使用字符串拼接组装提示词，避免format与ChatPromptTemplate变量冲突
_JOINT_PROMPT_PREFIX = """## 任务目标
从社交媒体文本中**同时**抽取地理实体和语义关系三元组，用于武汉地理知识图谱构建。

---"""

_JOINT_EXAMPLES = """## 正面示例

### 示例1：空间关系+包含
输入: "街道口商圈里面有群光广场和银泰城，逛完可以吃饭"
输出:
{{
  "entities": [
    {{\"name\": "街道口商圈", "type": "街区", "category": "商圈", "evidence": "街道口"}},
    {{\"name\": "群光广场", "type": "建筑物", "category": "商业综合体", "evidence": "群光广场"}},
    {{\"name\": "银泰城", "type": "建筑物", "category": "商业综合体", "evidence": "银泰城"}}
  ],
  "triples": [
    {{\"head\": "街道口商圈", "relation": "包含", "tail": "群光广场", "evidence": "里面有群光广场", "confidence": "high"}},
    {{\"head": "街道口商圈", "relation": "包含", "tail": "银泰城", "evidence": "里面有银泰城", "confidence": "high"}}
  ],
  "overall_confidence": "high"
}}

### 示例2：功能关系+属性
输入: "群光广场周末适合带娃逛街，但排队很久"
输出:
{{
  "entities": [
    {{\"name\": "群光广场", "type": "建筑物", "category": "商业综合体", "evidence": "群光广场"}}
  ],
  "triples": [
    {{\"head\": "群光广场", "relation": "具有功能", "tail": "购物",
     "evidence": "适合带娃逛街", "confidence": "high",
     "attributes": {{\"时段\": "周末", "适合人群\": "亲子", "具有限制\": ["排队久"]}}}}
  ],
  "overall_confidence": "high"
}}

### 示例3：相对方位关系
输入: "咖啡厅就在地铁站附近，对面是书店"
输出:
{{
  "entities": [
    {{\"name\": "咖啡厅", "type": "POI", "category": "餐饮", "evidence": "咖啡厅"}},
    {{\"name\": "地铁站", "type": "POI", "category": "交通", "evidence": "地铁站"}},
    {{\"name\": "书店", "type": "POI", "category": "文化", "evidence": "书店"}}
  ],
  "triples": [
    {{\"head\": "咖啡厅", "relation": "相对方位", "tail": "地铁站",
     "evidence": "就在地铁站附近", "confidence": "high",
     "attributes": {{\"距离值\": "近"}}}},
    {{\"head\": "书店", "relation": "相对方位", "tail": "咖啡厅",
     "evidence": "对面是书店", "confidence": "high",
     "attributes": {{\"方向值\": "对面"}}}}
  ],
  "overall_confidence": "high"
}}"""

_JOINT_PROMPT_SUFFIX = """---

## QA脚手架提示（如有）
{entity_hints}
{relation_hints}
{context_dependencies}

## 导师指导（如有）
{mentor_guidance}

## 待处理文本
{raw_text}

---

请输出联合抽取结果（JSON格式）。"""

# 使用字符串拼接组装最终提示词
JOINT_NER_RE_USER_V2 = "\n\n".join(
    [
        _JOINT_PROMPT_PREFIX,
        ENTITY_SCHEMA_CORE,
        "---",
        RELATION_SCHEMA_CORE,
        "---",
        ENTITY_ATTRIBUTE_SCHEMA,
        "---",
        VALIDATION_COT,
        "---",
        _JOINT_EXAMPLES,
        "---",
        NEGATIVE_EXAMPLES,
        _JOINT_PROMPT_SUFFIX,
    ]
)


JOINT_NER_RE_PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        ("system", JOINT_NER_RE_SYSTEM_V2),
        ("human", JOINT_NER_RE_USER_V2),
    ]
)


# ===== P12新增：增强版Self-Check反思机制 =====

SELF_CHECK_JOINT_SYSTEM_V2 = """你是一位"联合抽取校验专家"，负责独立审视Joint Extraction结果并生成结构化反思。

## 校验维度（四维度校验）
- **完整性**: 遗漏实体数量（0/1-2/3+）
- **准确性**: 实体类型判定正确率
- **真实性**: 三元组幻觉率（无原文依据比例）
- **证据性**: 证据匹配率（evidence来自原文比例）

## 反思原则
- 客观评估，不带偏见
- 问题优先级排序（遗漏>幻觉>类型错误）
- 改进策略可执行、具体化"""

SELF_CHECK_JOINT_USER_V2 = """## 校验任务

### 1. 四维度校验（量化评估）

| 维度 | 检查项 | 评分标准 |
|------|--------|----------|
| 完整性 | 遗漏实体数量 | 0=high, 1-2=medium, 3+=low |
| 准确性 | 类型判定错误数 | 0=high, 1-2=medium, 3+=low |
| 真实性 | 幻觉三元组数 | 0=high, 1-2=medium, 3+=low |
| 证据性 | 证据缺失数 | 0=high, 1-2=medium, 3+=low |

### 2. 反思生成（多角度结构化）

请从以下角度生成反思：

**遗漏原因分析**：
- 为什么遗漏了这些实体？（识别难点：简称/别名/边界模糊）
- 是否有QA脚手架提示但未关注？

**幻觉原因分析**：
- 为什么产生了无依据的三元组？（语义推断过度/关系模板滥用）
- 是否有反面示例警示但未注意？

**错误分类**：
- 遗漏 vs 幻觉 vs 类型错误（优先级：遗漏>幻觉>类型）

### 3. 改进策略（可执行、具体化）

| 问题类型 | 改进动作 |
|----------|----------|
| 遗漏实体 | 列出具体实体名称，附原文位置 |
| 幻觉三元组 | 列出具体三元组，标记为rejected |
| 类型错误 | 列出实体+正确类型 |
| 方向错误 | 列出三元组+正确头尾顺序 |

### 4. 重抽建议判定

| 条件 | 建议 |
|------|------|
| 遗漏≤1, 幻觉≤1 | confidence=high, 无需重抽 |
| 遗漏2-3, 幻觉2-3 | confidence=medium, 建议修正 |
| 遗漏>3, 幻觉>3 | confidence=low, 建议重抽 |

---

## 待校验结果
实体: {entities}
三元组: {triples}

## 原始文本
{raw_text}

## QA脚手架提示
{semantic_summary}
{context_dependencies}

## 重试历史（如有）
上一轮反思: {previous_reflection}
改进尝试: {improvement_attempts}

---

请输出校验结果，重点输出：
1. dimension_scores: 四维度评分
2. reflection_text: 结构化反思（遗漏原因+幻觉原因+改进优先级）
3. improvement_strategy: 可执行的改进动作列表
4. confidence: 整体置信度"""

SELF_CHECK_JOINT_PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_JOINT_SYSTEM_V2),
        ("human", SELF_CHECK_JOINT_USER_V2),
    ]
)


# ===== Self-Check反思结果格式化函数（增强版） =====


def format_dimension_scores(scores: Dict) -> str:
    """格式化四维度评分"""
    if not scores:
        return "(无评分)"
    lines = []
    for dim, score in scores.items():
        issues = score.get("issues", 0)
        rating = score.get("rating", "medium")
        lines.append(f"- {dim}: {rating} (问题数: {issues})")
    return "\n".join(lines)


def format_improvement_strategy(strategy: Dict) -> str:
    """格式化改进策略"""
    if not strategy:
        return "(无改进策略)"
    lines = []

    # 遗漏实体补充
    missing = strategy.get("missing_entities", [])
    if missing:
        lines.append("### 遗漏实体补充")
        for e in missing:
            name = e.get("name", "")
            type_ = e.get("type", "")
            evidence = e.get("evidence", "")
            lines.append(f'- 补充: {name} [{type_}] 原文依据: "{evidence}"')

    # 幻觉删除
    rejected = strategy.get("rejected_triples", [])
    if rejected:
        lines.append("### 幻觉三元组删除")
        for t in rejected:
            head = t.get("head", "")
            relation = t.get("relation", "")
            tail = t.get("tail", "")
            lines.append(f"- 删除: <{head}, {relation}, {tail}> (无原文依据)")

    # 类型修正
    type_corrections = strategy.get("type_corrections", [])
    if type_corrections:
        lines.append("### 类型修正")
        for c in type_corrections:
            name = c.get("name", "")
            wrong = c.get("wrong_type", "")
            correct = c.get("correct_type", "")
            lines.append(f"- {name}: {wrong} → {correct}")

    # 方向修正
    direction_corrections = strategy.get("direction_corrections", [])
    if direction_corrections:
        lines.append("### 方向修正")
        for c in direction_corrections:
            old = c.get("old_triple", "")
            new = c.get("correct_triple", "")
            lines.append(f"- {old} → {new}")

    return "\n".join(lines) if lines else "(无需改进)"


# ===== P13新增：优化版提示词（RISEN + CARE + TIDD-EC框架） =====
# Token优化目标：减少50-60% Token消耗，同时提升指令清晰度

# ===== 1. 精简版Schema组件（表格化，Token减少约40%） =====

ENTITY_SCHEMA_TABLE = """## 实体类型（v3.4扩展版：6种）

### 空间实体（GIS标准）—— 4种
| 类型 | 识别特征 | 示例 | 过滤条件 |
|------|----------|------|----------|
| 道路 | 路/大道/街/巷后缀 | 珞喻路、关山大道 | - |
| POI | 地名+功能标识 | 武汉大学、咖啡厅 | - |
| 建筑物 | 楼/厦/城/汇后缀 | 泛悦汇、行政楼 | - |
| 街区 | 商圈/校区/社区 | 街道口、光谷商圈 | - |

### 语义实体（v3.4新增）—— 2种
| 类型 | 识别特征 | 示例 | 说明 |
|------|----------|------|------|
| 功能 | 功能类型词 | 餐饮、购物、交通 | 场所用途类型 |
| 事件 | 事件名称词 | 樱花节、封路、开业 | 发生的具体事件 |

⚠️ **过滤规则**：这里/那边/附近/那边（泛化词）、人名、时间、纯数字"""

RELATION_SCHEMA_TABLE = """## 关系类型（v3.4精简版：8个，删除联动推荐）

| 类别 | 关系 | Head类型 | Tail类型 | 核心属性 |
|------|------|----------|----------|----------|
| 空间骨架 | 位于 | 地理实体（道路/POI/建筑物/街区） | 道路/街区 | 无 |
| 空间骨架 | 包含 | 街区 | POI/建筑物/道路 | 无 |
| 空间骨架 | 相对方位 | 地理实体 | 地理实体 | 距离值/方向值（v3.4删除联动推荐） |
| 语义血肉 | 具有功能 | 地理实体（场所） | 功能节点(9类)或功能实体 | 时段/适合人群(开放文本)/限制(开放文本列表) |
| 对比评价 | 优于 | 地理实体 | 地理实体 | 维度列表 |
| 对比评价 | 相似 | 地理实体 | 地理实体 | 维度列表 |
| 对比评价 | 劣于 | 地理实体 | 地理实体 | 维度列表 |
| 事件 | 发生事件 | 地理实体（场所） | 事件节点或事件实体 | (属性在事件实体上) |

⚠️ **地理实体** = 道路/POI/建筑物/街区（4种空间实体）
⚠️ **功能实体/事件实体不能参与空间关系**（位于/包含/相对方位）"""

FUNCTION_SCHEMA_TABLE = """## 功能节点（v3.4：10大类，新增交通）

| 类型 | 说明 | 频率 |
|------|------|------|
| 餐饮 | 吃饭、探店、下午茶 | 高频 |
| 购物 | 逛街、买东西 | 高频 |
| 休闲 | 游玩、散步、放松 | 高频 |
| 社交 | 聚会、打卡、约会 | 高频 |
| 观景 | 赏花、观展、拍照 | 高频 |
| 交通 | 打车、公交、地铁等出行 | 高频（v3.4新增） |
| 住宿 | 住酒店、民宿体验 | 中频 |
| 文化 | 学习、体验、参观 | 中频 |
| 工作 | 办公、产业相关 | 低频 |
| 其他 | 无法归类 | 兜底"""

ATTRIBUTE_SCHEMA_TABLE = """## 属性约束（v3.4版，必须有原文依据）

### 相对方位属性（v3.4删除联动推荐）
- 距离值: 近/中等/远
- 方向值: 东/南/西/北/东北/西南/东侧/西侧/对面/旁边

### 功能属性（v3.4开放文本版）
- 时段: 周末/晚上/樱花季等（文本）
- 适合人群: **开放文本** - 保留原文表达（带孩子来玩、闺蜜聚会、大学生打卡等）
- 具有限制: **开放文本列表** - 保留原文表达（排队两小时、停车超级难、需提前预约等）
- 情感倾向: 正面/中性/负面

### 对比属性
- 维度: 价格/环境/服务/人流量/品质/交通/口味/其他（多选，v3.3新增"其他")"""


# ===== 2. 约束规则组件（TIDD-EC框架） =====

CONSTRAINT_RULES = """## 必须遵守（Do）

✅ 每个三元组必须有 evidence 字段（原文依据）
✅ 实体类型必须为 6 类之一（道路/POI/建筑物/街区/功能/事件）
✅ 关系类型必须为 8 种之一（见上表）
✅ 属性值必须来自指定枚举列表
✅ 使用"其他"时，必须在描述字段说明具体内容
✅ 关系实体类型约束：
   - 位于：Head=地理实体（道路/POI/建筑物/街区），Tail=道路/街区
   - 包含：Head=街区，Tail=POI/建筑物/道路
   - 相对方位：Head=地理实体，Tail=地理实体
   - 具有功能：Head=地理实体（场所），Tail=功能节点或功能实体
   - 优于/相似/劣于：Head=地理实体，Tail=地理实体
   - 发生事件：Head=地理实体（场所），Tail=事件节点或事件实体

## 禁止行为（Don't）

❌ 生成无原文依据的三元组（幻觉）
❌ 使用泛化词作为实体（这里/那边/附近）
❌ 猜测未出现的属性值
❌ 将推荐指数/情感倾向作为关系而非实体属性
❌ 功能实体或事件实体作为空间关系的头/尾实体（位于/包含/相对方位）"""


# ===== 3. 反向验证CoT（RCoT框架） =====

RCOT_VALIDATION = """## 反向验证步骤（必须执行）

### Step 1: 幻觉检查
对每个三元组：能否在原文找到 evidence？
→ 无法找到 → 删除该三元组

### Step 2: 实体检查
实体是否为泛化词（这里/那边/附近/那边）？
→ 是 → 过滤该实体

### Step 3: 方向检查
头尾顺序是否正确？（A位于B vs B位于A）
→ 不正确 → 修正方向

### Step 4: 属性检查
属性值是否有原文依据？
→ 无依据 → 删除该属性"""


# ===== 4. Pre-Mortem失败预防 =====

PRE_MORTEM_CHECK = """## 预失败分析（Pre-Mortem）

假设本次抽取最终失败，请分析可能原因：

### 高风险模式（必须避免）
| 失败模式 | 检查方法 | 当前问题数 |
|----------|----------|------------|
| 幻觉三元组 | evidence能否找到原文？ | ？ |
| 遗漏实体 | 原文有地名但未抽取？ | ？ |
| 类型错误 | 实体类型判定合理？ | ？ |
| 方向错误 | 头尾顺序正确？ | ？ |

### 已知失败案例
- 原文"武大的樱花很美" → 错误<武汉大学, 发生事件, 樱花节>（幻觉，原文无"樱花节"）
- 原文"这边风景不错" → 错误实体"这边"[POI]（泛化词）
- 原文"群光在珞喻路上" → 错误<珞喻路, 位于, 群光>（方向颠倒）"""


# ===== 5. 优化版联合抽取提示词（RISEN框架） =====

JOINT_NER_RE_SYSTEM_V3 = """你是一位武汉地理知识图谱构建专家。
核心能力：实体边界识别、关系语义判断、证据溯源。
边界职责：仅抽取地理实体，排除人名/时间/非地理名词。
审慎原则：信息模糊时标记confidence=low，宁可遗漏不产生幻觉。"""

JOINT_NER_RE_USER_V3 = (
    """## Role（角色）
你是一位武汉地理知识图谱语义抽取专家。

---

## Instructions（任务）
从社交媒体文本中**同时**抽取地理实体和语义关系三元组。

---

"""
    + ENTITY_SCHEMA_TABLE
    + """

---

"""
    + RELATION_SCHEMA_TABLE
    + """

---

"""
    + FUNCTION_SCHEMA_TABLE
    + """

---

"""
    + CONSTRAINT_RULES
    + """

---

## Steps（思维链）

### 正向抽取
1. 扫描文本，标记地理位置词汇
2. 分类实体：道路/POI/建筑物/街区
3. 分析实体对语义关系，抽取三元组
4. 为每个抽取标注 evidence（原文依据）

"""
    + RCOT_VALIDATION
    + """

---

## End Goal（输出格式）

{{
  "entities": [{{"name": "...", "type": "...", "evidence": "..."}}],
  "triples": [{{"head": "...", "relation": "...", "tail": "...", "evidence": "...", "confidence": "high/medium/low"}}],
  "overall_confidence": "high/medium/low"
}}

---

## Context（可选）

实体提示: {entity_hints}
关系提示: {relation_hints}
导师指导: {mentor_guidance}

---

## Input（待处理文本）

{raw_text}

---

请输出联合抽取结果（JSON格式）。"""
)

JOINT_NER_RE_PROMPT_V3 = ChatPromptTemplate.from_messages(
    [
        ("system", JOINT_NER_RE_SYSTEM_V3),
        ("human", JOINT_NER_RE_USER_V3),
    ]
)


# ===== 6. 优化版Filter提示词（APE框架） =====

FILTER_SYSTEM_V2 = """你是地理文本快速筛选专家。
任务：高效判断文本是否值得处理，跳过无价值文本。"""

FILTER_USER_V2 = """## Action（任务）
判断文本是否包含武汉地理信息。

## Purpose（目的）
筛选无效文本，节省处理成本。

## Expectation（输出）
{
  "is_valid": true/false,
  "confidence": "high/medium/low",
  "skip_reason": "（无效时）原因",
  "has_geo_entity": true/false,
  "is_non_wuhan_region": true/false
}

## 判断规则

**有效文本**：
- 提及武汉地理实体（道路/POI/建筑物/街区）
- 武汉地标：武汉大学、黄鹤楼、东湖、江汉路等
- 无法确定时默认放行（保守策略）

**无效文本**：
- 纯情感表达、<5字符、纯表情
- 明确非武汉地区（北京故宫、西湖等）

## 待筛选文本
{raw_text}

请输出筛选结果（JSON格式）。"""

FILTER_PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        ("system", FILTER_SYSTEM_V2),
        ("human", FILTER_USER_V2),
    ]
)


# ===== 7. 优化版RE提示词（表格化Schema） =====

RE_SYSTEM_V2 = """你是地理语义专家，擅长从文本中提取实体关系。
核心原则：所有关系必须有原文依据（evidence），禁止幻觉。"""

RE_USER_V2 = (
    """## 任务
从文本中抽取实体间的语义关系三元组。

---

"""
    + ENTITY_SCHEMA_TABLE
    + """

---

"""
    + RELATION_SCHEMA_TABLE
    + """

---

"""
    + ATTRIBUTE_SCHEMA_TABLE
    + """

---

"""
    + CONSTRAINT_RULES
    + """

---

## Steps（思维链）

"""
    + RCOT_VALIDATION
    + """

---

## 已识别实体
{entities}

---

## QA脚手架提示
{relation_hints}
{context_dependencies}

---

## 待处理文本
{raw_text}

---

请输出关系抽取结果（JSON格式）。"""
)

RE_PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        ("system", RE_SYSTEM_V2),
        ("human", RE_USER_V2),
    ]
)


# ===== 8. 优化版Self-Check提示词（Pre-Mortem + 四维度评分） =====

SELF_CHECK_JOINT_SYSTEM_V3 = """你是联合抽取校验专家。
任务：四维度量化评估 + Pre-Mortem失败预防 + 结构化反思。"""

SELF_CHECK_JOINT_USER_V3 = (
    """## Pre-Mortem（假设失败分析）

"""
    + PRE_MORTEM_CHECK
    + """

---

## 四维度量化评分

| 维度 | 检查项 | 评分标准 | 当前问题数 |
|------|--------|----------|------------|
| 完整性 | 遗漏实体 | 0=high, 1-2=medium, 3+=low | {missing_count} |
| 准确性 | 类型错误 | 同上 | {type_error_count} |
| 真实性 | 幻觉三元组 | 同上 | {hallucination_count} |
| 证据性 | evidence缺失 | 同上 | {evidence_missing_count} |

---

## RCoT反向验证

"""
    + RCOT_VALIDATION
    + """

---

## 待校验结果
实体: {entities}
三元组: {triples}

---

## 原始文本
{raw_text}

---

## QA脚手架
{semantic_summary}
{context_dependencies}

---

## 重试历史
上一轮反思: {previous_reflection}
改进尝试: {improvement_attempts}

---

请输出校验结果：
1. dimension_scores: 四维度评分
2. reflection_text: 结构化反思
3. improvement_actions: 可执行的改进动作列表
4. confidence: 整体置信度"""
)

SELF_CHECK_JOINT_PROMPT_V3 = ChatPromptTemplate.from_messages(
    [
        ("system", SELF_CHECK_JOINT_SYSTEM_V3),
        ("human", SELF_CHECK_JOINT_USER_V3),
    ]
)


# ===== 9. 优化版Label提示词 =====

LABEL_SYSTEM_V2 = """你是地理知识标注专家。
任务：为实体和关系打上属性标签，必须有原文依据。"""

LABEL_USER_V2 = (
    """## 任务
为实体和关系标注属性（v3.3：特征标签开放文本）。

---

"""
    + ENTITY_SCHEMA_TABLE
    + """

---

## 实体属性（从文本提取）

| 属性 | 类型 | 提取条件 |
|------|------|----------|
| 类别 | 枚举(道路/POI/建筑物/街区) | 用于NER边界识别 |
| 细分 | **开放文本** | 仅记录文本中明确提及的分类词 |
| 特征标签 | **开放文本** | 保留原文表达（氛围超好、遛娃神器等） |
| 推荐指数 | 枚举(超推/推荐/一般/不推荐) | 明确推荐语句出现 |
| 情感倾向 | 枚举(正面/中性/负面) | 情感词出现 |

---

"""
    + ATTRIBUTE_SCHEMA_TABLE
    + """

---

"""
    + CONSTRAINT_RULES
    + """

---

## QA脚手架
{semantic_summary}
{entity_hints}
{relation_hints}

---

## 待标注实体
{entities}

## 待标注关系
{relations}

---

## 原始文本
{raw_text}

---

请输出属性标注结果（JSON格式）。"""
)

LABEL_PROMPT_V2 = ChatPromptTemplate.from_messages(
    [
        ("system", LABEL_SYSTEM_V2),
        ("human", LABEL_USER_V2),
    ]
)


# ===== 10. Token对比统计（供参考） =====

TOKEN_ESTIMATE_V2_V3 = """
## Token优化对比估算

| 提示词版本 | V2估算Token | V3估算Token | 减少比例 |
|------------|-------------|-------------|----------|
| JOINT_NER_RE | ~4000 | ~1500 | **62.5%** |
| RE | ~3000 | ~1200 | **60%** |
| FILTER | ~1500 | ~500 | **67%** |
| SELF_CHECK_JOINT | ~2000 | ~800 | **60%** |
| LABEL | ~2500 | ~1000 | **60%** |

主要优化手段：
1. 表格化Schema（减少40%）
2. RISEN框架结构化（提升清晰度）
3. TIDD-EC Do/Don't列表（集中约束）
4. RCoT反向验证（增加质量）
"""


# ===== 11. 辅助函数：动态组装优化版提示词 =====


def assemble_optimized_joint_prompt(
    raw_text: str,
    entity_hints: str = "(无)",
    relation_hints: str = "(无)",
    mentor_guidance: str = "(无)",
) -> str:
    """
    动态组装优化版联合抽取提示词

    优势：
    1. 模块化组件，易于维护
    2. Token效率高
    3. 结构清晰（RISEN框架）
    """
    # 使用字符串拼接避免format冲突
    parts = [
        "## Role（角色）",
        "你是一位武汉地理知识图谱语义抽取专家。",
        "",
        "---",
        "",
        ENTITY_SCHEMA_TABLE,
        "",
        "---",
        "",
        RELATION_SCHEMA_TABLE,
        "",
        "---",
        "",
        FUNCTION_SCHEMA_TABLE,
        "",
        "---",
        "",
        CONSTRAINT_RULES,
        "",
        "---",
        "",
        "## Steps（思维链）",
        "",
        "### 正向抽取",
        "1. 扫描文本，标记地理位置词汇",
        "2. 分类实体：道路/POI/建筑物/街区",
        "3. 分析实体对语义关系，抽取三元组",
        "4. 为每个抽取标注 evidence（原文依据）",
        "",
        RCOT_VALIDATION,
        "",
        "---",
        "",
        "## End Goal（输出格式）",
        "",
        "{",
        '  "entities": [{"name": "...", "type": "...", "evidence": "..."}],',
        '  "triples": [{"head": "...", "relation": "...", "tail": "...", "evidence": "...", "confidence": "high/medium/low"}],',
        '  "overall_confidence": "high/medium/low"',
        "}",
        "",
        "---",
        "",
        "## Context（可选）",
        f"实体提示: {entity_hints}",
        f"关系提示: {relation_hints}",
        f"导师指导: {mentor_guidance}",
        "",
        "---",
        "",
        "## Input（待处理文本）",
        "",
        raw_text,
        "",
        "---",
        "",
        "请输出联合抽取结果（JSON格式）。",
    ]

    return "\n".join(parts)
