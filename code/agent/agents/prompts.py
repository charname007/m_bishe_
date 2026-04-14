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
- 涉及空间关系：位于、旁边、连接、附近、在...内等
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

FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FILTER_SYSTEM),
    ("human", FILTER_USER),
])


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

NORMALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NORMALIZE_SYSTEM),
    ("human", NORMALIZE_USER),
])


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
  "relation_hints": ["属于", "承载活动", "引发情感", "位于"],
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
  "relation_hints": ["位于", "相邻", "距离"],
  "context_dependencies": ["华中师范大学可简称华师"],
  "overall_confidence": "high",
  "should_skip_detailed_extraction": false
}}

## 待处理文本（已归一化）
{normalized_text}

请输出QA脚手架结果（JSON格式）。"""

QA_SCAFFOLD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SCAFFOLD_SYSTEM),
    ("human", QA_SCAFFOLD_USER),
])


# ===== Step 1: NER 提示词模板 =====

NER_SYSTEM = """你是一位"地理语义专家"，精通城市地理实体识别与社交媒体语料分析。
你的任务是从小红书文本中提取地理知识实体。"""

NER_USER = """## 候选目标
请识别以下类别的实体：
- 道路(Road): 街道、大道、小巷等（如：关山大道）
- POI(Point of Interest): 具体店名、地标、机构（如：武汉大学、某某咖啡厅）
- 建筑物(Building): 具体的楼宇、商场主体（如：泛悦汇）
- 街区(Block): 具有边界感的生活区域（如：街道口、华农校区）

## 思维链(CoT)
1. 首先，识别句中指代具体位置的专有名词
2. 其次，根据上下文判断其实体粒度
3. 最后，将其归入上述四个候选目标之一

## QA脚手架提示（如有）
前置QA分析可能发现以下实体提示，可作为参考：
{entity_hints}

上下文依赖提醒：
{context_dependencies}

## 任务示例
输入: "在洪山区的街道口，泛悦汇三楼的这家书店氛围感拉满。"
输出: {{\"道路\": [], \"POI\": [\"书店\"], \"建筑物\": [\"泛悦汇\"], \"街区\": [\"街道口\"]}}

## 待处理文本
{raw_text}

请输出实体识别结果（JSON格式）。"""

NER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NER_SYSTEM),
    ("human", NER_USER),
])


# ===== Step 2: RE 提示词模板（v2.2改进：18个关系体系） =====

RE_SYSTEM = """你是一位"地理语义专家"，擅长梳理非结构化文本中的语义逻辑。
你精通社交媒体地理文本分析，能够准确提取实体间的关系和属性。"""

RE_USER = """## 候选目标
请识别实体间的以下三元组关系：<头实体, 关系, 尾实体, 属性>

### 空间基础关系（8个）—— 图谱骨架

| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| **位于** | A在B处 | 地理实体 | 无 |
| **相邻** | A和B空间邻近 | 地理实体 | **联动推荐**（布尔） |
| **属于** | A是B的组成部分 | 地理实体 | 无 |
| **连接** | A和B交通连接 | 地理实体 | 无 |
| **距离** | A距离B的远近程度 | 地理实体 | **距离值**：近/中等/远 |
| **方向** | A在B的某方位 | 地理实体 | **方向值**：东/南/西/北/东北/西南/东侧/西侧/对面/旁边 |
| **穿过** | 道路穿越区域 | 区域/街区 | 无 |
| **变化为** | A已变更为B | 地理实体 | **变化时间**（可选） |

### 社交语义关系（6个）—— 图谱血肉

| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| **推荐指数** | 用户推荐程度 | 评价等级节点（超推/推荐/一般/不推荐） | 无 |
| **承载活动** | 场所可进行的活动 | 活动节点 | **时段**+**适合人群**+**具有限制**（可选） |
| **可达方式** | 到达的交通方式 | 交通方式节点（地铁/公交/步行/自驾/骑行/打车） | 无 |
| **消费档次** | 消费水平 | 消费等级节点（平价/中档/高档/奢侈） | 无 |
| **品类特征** | 风格/文化特征 | 特征标签节点（老字号/网红/新潮/文创/ins风/复古） | 无 |
| **引发情感** | 情感体验大类 | 情感节点（正面/中性/负面） | **具体表达**（可选） |

### 对比评价关系（3个）—— 特色

| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| **优于** | A在某方面好于B | 地理实体 | **维度**（列表） |
| **相似** | A和B在某方面相似 | 地理实体 | **维度**（列表） |
| **劣于** | A在某方面不如B | 地理实体 | **维度**（列表） |

### 事件关系（1个）

| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| **发生事件** | 场所发生的特定事件 | 事件节点（LLM归纳命名） | **事件类别**+**状态**+**时间** |

---

## 属性详细说明

### 承载活动关系属性

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **时段** | 时间节点（可选） | 活动最适合的时间 | 周末/晚上/樱花季/春季/夏季 |
| **适合人群** | 人群节点（可选） | 活动适合的人群 | 亲子/宝妈/学生党/情侣/打工人/特种兵/银发族/宠物主/独行者/团建 |
| **具有限制** | 限制节点列表（可选） | 活动的限制条件 | 需预约/排队久/停车难/限流/谢绝宠物/只收现金/时间限制/人数限制/消费门槛/季节限制 |

### 对比关系属性

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **维度** | 列表 | 对比的方面 | 价格/环境/服务/人流量/品质/氛围/交通/停车/口味/性价比 |

### 发生事件关系属性

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **事件类别** | 枚举 | 事件大类 | 自然事件/人文事件/商业事件/社会事件/负面事件 |
| **状态** | 枚举 | 事件状态 | 正在进行/已结束/计划中/周期性 |
| **时间** | 时间节点（可选） | 事件发生时间 | 周末/樱花季/2024年 |

---

## 思维链(CoT)
1. 观察已识别的实体对
2. 分析句中的动词/形容词/名词，判断其反映的关系类型
3. 判断是否需要提取属性（时段、维度、人群、限制等）
4. 若句子存在主语省略，请结合段落背景推断

---

## 任务示例

### 示例1：空间关系
输入文本: "武汉大学在珞喻路上，旁边是群光广场，逛完可以去街道口吃饭"
已知实体: {{\"道路\": [\"珞喻路\"], \"POI\": [\"武汉大学\", \"群光广场\"], \"建筑物\": [], \"街区\": [\"街道口\"]}}
输出:
{{\"triples\": [
    {{\"head\": \"武汉大学\", \"relation\": \"位于\", \"tail\": \"珞喻路\", \"evidence\": \"武汉大学在珞喻路上\"}},
    {{\"head\": \"群光广场\", \"relation\": \"相邻\", \"tail\": \"武汉大学\", \"evidence\": \"旁边是群光广场\", \"attributes\": {{\"联动推荐\": false}}}},
    {{\"head\": \"街道口\", \"relation\": \"相邻\", \"tail\": \"群光广场\", \"evidence\": \"逛完可以去街道口\", \"attributes\": {{\"联动推荐\": true}}}}
  ]
}}

### 示例2：社交语义关系+属性
输入文本: "周末很适合带孩子来玩，但排队很久"
已知实体: {{\"道路\": [], \"POI\": [\"公园\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"公园\", \"relation\": \"承载活动\", \"tail\": \"玩\", \"evidence\": \"很适合带孩子来玩\", \"attributes\": {{\"时段\": \"周末\", \"适合人群\": \"亲子\", \"具有限制\": [\"排队久\"]}}}}
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

### 示例4：情感关系+事件关系
输入文本: "樱花节正在举办，超级治愈，强烈推荐"
已知实体: {{\"道路\": [], \"POI\": [\"武汉大学\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"武汉大学\", \"relation\": \"发生事件\", \"tail\": \"樱花节\", \"evidence\": \"樱花节正在举办\", \"attributes\": {{\"事件类别\": \"人文事件\", \"状态\": \"正在进行\"}}}},
    {{\"head\": \"武汉大学\", \"relation\": \"引发情感\", \"tail\": \"正面\", \"evidence\": \"超级治愈\", \"attributes\": {{\"具体表达\": \"治愈\"}}}},
    {{\"head\": \"武汉大学\", \"relation\": \"推荐指数\", \"tail\": \"超推\", \"evidence\": \"强烈推荐\"}}
  ]
}}

### 示例5：距离+方向关系
输入文本: "咖啡厅就在地铁站附近，对面是书店"
已知实体: {{\"道路\": [], \"POI\": [\"咖啡厅\", \"地铁站\", \"书店\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"咖啡厅\", \"relation\": \"距离\", \"tail\": \"地铁站\", \"evidence\": \"就在地铁站附近\", \"attributes\": {{\"距离值\": \"近\"}}}},
    {{\"head\": \"书店\", \"relation\": \"方向\", \"tail\": \"咖啡厅\", \"evidence\": \"对面是书店\", \"attributes\": {{\"方向值\": \"对面\"}}}}
  ]
}}

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

RE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RE_SYSTEM),
    ("human", RE_USER),
])


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

EVAL_PROMPT_1 = ChatPromptTemplate.from_messages([
    ("system", EVAL_1_SYSTEM),
    ("human", EVAL_1_USER),
])


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

EVAL_PROMPT_2 = ChatPromptTemplate.from_messages([
    ("system", EVAL_2_SYSTEM),
    ("human", EVAL_2_USER),
])


# P2改进：简化的单次评估提示词（合并评分和修正）
EVAL_SIMPLIFIED_SYSTEM = """你是一位"地理语义评审专家"。你的任务是评估三元组并在发现错误时直接修正。"""

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
- 修正关系类型（如：将"位于"改为"属于")
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

EVAL_PROMPT_SIMPLIFIED = ChatPromptTemplate.from_messages([
    ("system", EVAL_SIMPLIFIED_SYSTEM),
    ("human", EVAL_SIMPLIFIED_USER),
])


# ===== Step 4: 属性标注提示词模板（v2.2改进） =====

LABEL_SYSTEM = """你是一位"地理知识管理专家"，精通GIS标准、城市规划术语和社交媒体语义分析。
你的任务是将初步知识片段转化为具备专业语义背景的结构化知识。"""

LABEL_USER = """## 任务描述
请为已识别的实体和关系打上专业属性标签。

---

## 实体属性标注

### 基础分类属性（GIS标准）

| 类别 | 细分枚举 |
|------|---------|
| POI | 餐饮、交通、教育、历史保护、购物、医疗、娱乐、文化、酒店、服务 |
| 建筑物 | 商业综合体、住宅、办公楼、文化设施、教育设施、医疗设施 |
| 街区 | 商圈、校区、社区、行政区、景区 |
| 道路 | 主干道、次干道、支路、小巷、地铁线路 |

### 文本属性（从语料提取）

| 属性 | 枚举值 |
|------|--------|
| 情感标签 | 氛围感、治愈、高级感、温暖、文艺、复古、现代、网红感、小清新、赛博朋克感 |
| 体验评价 | 服务好、环境舒适、商品丰富、性价比高、停车方便、交通便利、人流量适中 |
| 知名度 | 热门、小众、隐藏宝藏、必去、打卡圣地 |

### 元数据属性

| 属性 | 枚举值 | 说明 |
|------|--------|------|
| 来源可信度 | 高/中/低 | 多条文本一致=高，单条有佐证=中，单条无佐证=低 |

---

## 关系属性标注（Label阶段补充）

### 空间关系属性

| 关系 | Label补充属性 |
|------|--------------|
| 位于 | 空间精度[精确/近似/模糊]，语义类型[内部/区域内/附近/周边] |
| 相邻 | 空间精度[精确/近似]，相邻类型[直接相邻/邻近/隔街相望] |
| 属于 | 层级类型[组成部分/行政隶属/功能隶属] |
| 连接 | 连接类型[直达/换乘/途径/沿线]，交通方式[地铁/公交/步行/自驾] |
| 距离 | 空间精度[精确/近似]，距离类型[物理距离/感知距离/步行距离] |
| 方向 | 方向类型[绝对方位/相对方位/定性方位] |
| 穿过 | 穿过类型[横穿/纵穿/穿越] |
| 变化为 | 变化类型[业态变更/功能转变/建筑改造/关闭拆除] |

### 社交语义关系属性

| 关系 | Label补充属性 |
|------|--------------|
| 推荐指数 | 推荐强度[强烈/一般/较弱]，推荐场景[日常/周末/节假日/约会/团建] |
| 承载活动 | 活动类型[体验型/消费型/社交型/休闲型/观赏型]，活动频率[高频/中频/低频/季节性] |
| 可达方式 | 可达程度[直达/换乘/需步行/不便]，交通效率[高效/一般/低效] |
| 消费档次 | 价格区间[补充数值]，消费类型[日常消费/休闲消费/高端消费] |
| 品类特征 | 特征类型[风格特征/文化特征/历史特征/功能特征]，特征显著性[显著/一般/微弱] |
| 引发情感 | 情感强度[强烈/一般/微弱]，情感类型[愉悦型/放松型/感动型/浪漫型/负面型] |

### 对比关系属性

| 关系 | Label补充属性 |
|------|--------------|
| 优于 | 优势程度[明显优势/稍有优势/相当]，对比可靠性[主观对比/客观对比] |
| 相似 | 相似程度[高度相似/部分相似/风格相近]，替代性[可替代/部分替代/不可替代] |
| 劣于 | 劣势程度[明显劣势/稍有劣势/相当]，风险等级[高风险/中风险/低风险] |

### 事件关系属性

| 关系 | Label补充属性 |
|------|--------------|
| 发生事件 | 事件影响度[重大影响/一般影响/微弱影响]，事件持续性[长期事件/短期事件/周期性事件] |

---

## 任务示例

输入实体: ["武汉大学", "群光广场", "街道口"]
输入关系: [
  "<武汉大学, 位于, 珞喻路>",
  "<武汉大学, 承载活动, 拍照> [时段=樱花季, 适合人群=学生党]",
  "<群光广场, 相邻, 街道口> [联动推荐=true]",
  "<群光广场, 消费档次, 中档>",
  "<武汉大学, 引发情感, 正面> [具体表达=治愈]"
]
原始文本: "武汉大学樱花季超治愈，强烈推荐学生党打卡，旁边群光广场可以逛完去街道口吃饭"

输出:
{{\"entities\": {{
    \"武汉大学\": {{
      \"类别\": \"POI\",
      \"细分\": \"教育\",
      \"情感标签\": [\"治愈\", \"浪漫\", \"氛围感\"],
      \"体验评价\": [\"环境优美\", \"历史悠久\"],
      \"知名度\": \"热门\",
      \"来源可信度\": \"高\"}},
    \"群光广场\": {{
      \"类别\": \"建筑物\",
      \"细分\": \"商业综合体\",
      \"情感标签\": [\"现代\", \"热闹\"],
      \"体验评价\": [\"商品丰富\", \"停车方便\"],
      \"知名度\": \"热门\",
      \"来源可信度\": \"高\"}},
    \"街道口\": {{
      \"类别\": \"街区\",
      \"细分\": \"商圈\",
      \"情感标签\": [\"热闹\"],
      \"体验评价\": [\"交通便利\"],
      \"知名度\": \"热门\",
      \"来源可信度\": \"高\"}}
  }},
  \"relations\": {{
    \"<武汉大学, 位于, 珞喻路>\": {{
      \"空间精度\": \"精确\",
      \"语义类型\": \"区域内\",
      \"来源可信度\": \"高\"}},
    \"<武汉大学, 承载活动, 拍照>\": {{
      \"活动类型\": \"体验型\",
      \"活动频率\": \"季节性\",
      \"来源可信度\": \"高\"}},
    \"<群光广场, 相邻, 街道口>\": {{
      \"空间精度\": \"精确\",
      \"相邻类型\": \"直接相邻\",
      \"来源可信度\": \"高\"}},
    \"<群光广场, 消费档次, 中档>\": {{
      \"价格区间\": \"人均50-150\",
      \"消费类型\": \"休闲消费\",
      \"来源可信度\": \"高\"}},
    \"<武汉大学, 引发情感, 正面>\": {{
      \"情感强度\": \"强烈\",
      \"情感类型\": \"愉悦型\",
      \"来源可信度\": \"高\"}}
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

## 待标注关系（含RE阶段抽取的属性）
{relations}

## 原始文本（用于提取情感标签、体验评价）
{raw_text}

请输出属性标注结果（JSON格式）。"""

LABEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", LABEL_SYSTEM),
    ("human", LABEL_USER),
])


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
        attrs = t.get('attributes', {})
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
        evidence = t.get('evidence', '')
        attrs = t.get('attributes', {})

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
            parts.append(f"证据:\"{evidence}\"")

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

SELF_CHECK_NER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SELF_CHECK_NER_SYSTEM),
    ("human", SELF_CHECK_NER_USER),
])


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

SELF_CHECK_RE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SELF_CHECK_RE_SYSTEM),
    ("human", SELF_CHECK_RE_USER),
])


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


# ===== P9新增：联合抽取提示词 =====

JOINT_NER_RE_SYSTEM = """你是一位"地理语义联合抽取专家"，擅长在一次推理中同时识别实体和关系。
你的优势在于：能够全局理解文本，避免实体边界识别错误对关系判定的干扰。"""

JOINT_NER_RE_USER = """## 任务描述
请从文本中**同时**抽取：
1. 地理实体（道路/POI/建筑物/街区）
2. 实体间的语义关系（18种关系类型）
3. 每个抽取的证据依据

## 实体类型定义
| 类型 | 定义 | 示例 |
|------|------|------|
| 道路 | 交通通道 | 珞喻路、关山大道 |
| POI | 具体地点 | 武汉大学、群光广场 |
| 建筑物 | 建筑设施 | 泛悦汇、融科天城 |
| 街区 | 地理区域 | 街道口、光谷商圈 |

## 关系类型（18种）
### 空间基础关系（8个）
- 位于、相邻、属于、连接、距离、方向、穿过、变化为

### 社交语义关系（6个）
- 推荐指数、承载活动、可达方式、消费档次、品类特征、引发情感

### 对比评价关系（3个）
- 优于、相似、劣于

### 事件关系（1个）
- 发生事件

## QA脚手架提示（如有）
{entity_hints}
{relation_hints}
{context_dependencies}

## 联合抽取策略（CoT）
1. **第一步**：扫描文本，识别所有可能的地名、道路、建筑等
2. **第二步**：对识别的实体，判断其类型和类别
3. **第三步**：分析实体之间的语义关系，抽取三元组
4. **第四步**：为每个抽取提供原文依据（evidence）
5. **第五步**：评估整体置信度

## 任务示例

### 示例1：基础联合抽取
输入: "武大的樱花开了，很多人在行政楼前拍照打卡"

输出:
 {{
  "entities": [
    {{\"name\": "武汉大学", "type": "POI", "category": "高校", "aliases": ["武大"], "evidence": "武大"}},
    {{\"name": "行政楼", "type": "建筑物", "category": "教育设施", "aliases": [], "evidence": "行政楼"}}
  ],
  "triples": [
    {{\"head": "行政楼", "relation": "属于", "tail": "武汉大学", "evidence": "行政楼", "confidence": "high"}},
    {{\"head": "武汉大学", "relation": "承载活动", "tail": "拍照打卡", "evidence": "拍照打卡", "confidence": "high", "attributes": {{\"时段": "樱花季"}}}}
  ],
  "entity_relation_mapping": {{
    "武汉大学": ["<行政楼, 属于, 武汉大学>", "<武汉大学, 承载活动, 拍照打卡>"]
  }},
  "overall_confidence": "high"
 }}

### 示例2：复杂语义关系
输入: "群光广场就在珞喻路上，比街道口更热闹，周末适合带娃逛街"

输出:
 {{
  "entities": [
    {{\"name\": "群光广场", \"type\": "建筑物", \"category\": "商业综合体", \"aliases\": [], \"evidence\": "群光广场"}},
    {{\"name\": "珞喻路", \"type\": "道路", \"category\": "主干道", \"aliases\": [], \"evidence\": "珞喻路"}},
    {{\"name\": "街道口", \"type\": "街区", \"category\": "商圈", \"aliases\": [], \"evidence\": "街道口"}}
  ],
  "triples": [
    {{\"head\": "群光广场", \"relation\": "位于", \"tail\": "珞喻路", \"evidence\": "就在珞喻路上", \"confidence\": "high"}},
    {{\"head\": "群光广场", \"relation\": "优于", \"tail\": "街道口", \"evidence\": "比街道口更热闹", \"confidence\": "medium", \"attributes\": {{\"维度\": ["氛围"]}}}}
  ],
  "entity_relation_mapping": {{
    "群光广场": ["<群光广场, 位于, 珞喻路>", "<群光广场, 优于, 街道口>"]
  }},
  "overall_confidence": "high"
 }}

## 待处理文本
{raw_text}

请输出联合抽取结果（JSON格式）。"""

JOINT_NER_RE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", JOINT_NER_RE_SYSTEM),
    ("human", JOINT_NER_RE_USER),
])


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

SELF_CHECK_JOINT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SELF_CHECK_JOINT_SYSTEM),
    ("human", SELF_CHECK_JOINT_USER),
])


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

SELF_CHECK_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SELF_CHECK_QA_SYSTEM),
    ("human", SELF_CHECK_QA_USER),
])


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

SELF_CHECK_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SELF_CHECK_EVAL_SYSTEM),
    ("human", SELF_CHECK_EVAL_USER),
])


# ===== P9新增：Self-Check-Label提示词 =====

SELF_CHECK_LABEL_SYSTEM = """你是一位"标注结果校验专家"，负责审视属性标注结果。
你的任务是：验证属性合理性、检查完整性，并生成反思建议。"""

SELF_CHECK_LABEL_USER = """## 校验任务

### 1. 实体属性验证
- 类别、细分是否正确？
- 情感标签、体验评价是否与原文匹配？
- 知名度判断是否合理？

### 2. 关系属性验证
- 空间精度、语义类型等属性是否正确？
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

SELF_CHECK_LABEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SELF_CHECK_LABEL_SYSTEM),
    ("human", SELF_CHECK_LABEL_USER),
])


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

SELF_CHECK_FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SELF_CHECK_FILTER_SYSTEM),
    ("human", SELF_CHECK_FILTER_USER),
])


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

SELF_CHECK_NORMALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SELF_CHECK_NORMALIZE_SYSTEM),
    ("human", SELF_CHECK_NORMALIZE_USER),
])


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
        lines.append(f"- {e.get('name', '')} [{e.get('type', '')}] 类别:{e.get('category', '')}{alias_str} 证据:\"{evidence}\"")
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
        lines.append(f"- {base} 置信度:{confidence}{attr_str} 证据:\"{evidence}\"")
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
        lines.append(f"- {t_str} SEM:{s.get('SEM', 0)} FAC:{s.get('FAC', 0)} CON:{s.get('CON', 0)}")
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


# ===== P10新增：批量LLM调用提示词 =====

BATCH_JOINT_SYSTEM = """你是一位"地理语义批量抽取专家"，擅长一次处理多条文本，同时提取地理实体和三元组关系。
你的核心优势：
1. **高效处理**：一次推理完成多条语料的抽取，大幅降低成本
2. **跨语料感知**：识别不同文本中的同名实体和别名（如"武大"和"武汉大学"是同一实体）
3. **一致性保证**：对相同实体的类型判断保持一致
"""

BATCH_JOINT_USER = """## 任务描述
请同时处理以下多条语料（共 {batch_size} 条），为每条语料提取：
1. 地理实体（道路/POI/建筑物/街区）
2. 实体间的语义关系三元组
3. 每个抽取的原文依据

---

## 实体类型定义

| 类型 | 定义 | 示例 |
|------|------|------|
| 道路 | 交通通道 | 珞喻路、关山大道、雄楚大道 |
| POI | 具体地点/机构 | 武汉大学、群光广场、某某咖啡厅 |
| 建筑物 | 建筑设施 | 泛悦汇、融科天城、行政楼 |
| 街区 | 地理区域 | 街道口、光谷商圈、华农校区 |

---

## 关系类型（18种）

### 空间基础关系（8个）
- **位于**：A在B处（如：武汉大学 位于 珞喻路）
- **相邻**：A和B空间邻近（如：群光广场 相邻 街道口）
- **属于**：A是B的组成部分（如：行政楼 属于 武汉大学）
- **连接**：A和B交通连接
- **距离**：A距离B的远近（属性：近/中等/远）
- **方向**：A在B的某方位（属性：东/南/西/北/对面/旁边）
- **穿过**：道路穿越区域
- **变化为**：A已变更为B

### 社交语义关系（6个）
- **推荐指数**：用户推荐程度（尾：超推/推荐/一般/不推荐）
- **承载活动**：场所可进行的活动（属性：时段、适合人群、限制）
- **可达方式**：交通方式（尾：地铁/公交/步行/自驾）
- **消费档次**：消费水平（尾：平价/中档/高档/奢侈）
- **品类特征**：风格/文化特征
- **引发情感**：情感体验（尾：正面/中性/负面）

### 对比评价关系（3个）
- **优于**：A在某方面好于B（属性：维度列表）
- **相似**：A和B相似
- **劣于**：A在某方面不如B

### 事件关系（1个）
- **发生事件**：场所发生的特定事件

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
1. `results`: 每条语料的抽取结果（实体、三元组、置信度）
2. `cross_corpus_aliases`: 跨语料发现的别名映射
3. `overall_confidence`: 整体置信度评估

输出JSON格式。
"""

BATCH_JOINT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BATCH_JOINT_SYSTEM),
    ("human", BATCH_JOINT_USER),
])


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

BATCH_SELF_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BATCH_SELF_CHECK_SYSTEM),
    ("human", BATCH_SELF_CHECK_USER),
])


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
        triple_str = ", ".join([f"<{t.get('head', '')}, {t.get('relation', '')}, {t.get('tail', '')}>" for t in triples[:3]])

        lines.append(f"- [{corpus_id}] 置信度:{confidence}\n  实体: {entity_str}\n  三元组: {triple_str}")
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