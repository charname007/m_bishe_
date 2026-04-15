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
  "relation_hints": ["位于", "具有功能", "方位"],
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
  "relation_hints": ["位于", "方位", "包含"],
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


# ===== Step 2: RE 提示词模板（v3.2精简版：8个关系体系） =====

RE_SYSTEM = """你是一位"地理语义专家"，擅长梳理非结构化文本中的语义逻辑。
你精通社交媒体地理文本分析，能够准确提取实体间的关系和属性。"""

RE_USER = """## 候选目标
请识别实体间的以下三元组关系：<头实体, 关系, 尾实体, 属性>

### 空间基础关系（3个）—— 图谱骨架（v3.2精简版）

| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| **位于** | A坐落于B处（空间定位/归属） | 道路/街区/行政区 | 无 |
| **包含** | A空间包含B（位于的反向） | POI/建筑物/道路 | 无 |
| **方位** | A和B空间邻近+方位关系 | 地理实体 | **距离值**+**方向值**+**联动推荐**（可选） |

**注**：原"相邻"、"距离"、"方向"已合并为"方位"关系，通过属性区分。

### 社交语义关系（1个）—— 图谱血肉（v3.2精简版）

| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| **具有功能** | 场所可进行的功能用途 | 功能节点（9大类） | **时段**+**适合人群**+**具有限制**+**情感倾向**（可选） |

**注**：原"承载活动"改为"具有功能"；"推荐指数"和"引发情感"已改为实体属性而非关系。

### 对比评价关系（3个）—— 特色

| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| **优于** | A在某方面好于B | 地理实体 | **维度**（列表） |
| **相似** | A和B在某方面相似 | 地理实体 | **维度**（列表） |
| **劣于** | A在某方面不如B | 地理实体 | **维度**（列表） |

### 事件关系（1个）

| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| **发生事件** | 场所发生的特定事件 | 事件节点（LLM归纳命名） | 无（属性全部在事件节点上） |

---

## 重要说明：实体属性而非关系

以下语义应作为**实体属性**而非三元组关系抽取：

| 属性 | 说明 | 枚举值 |
|------|------|--------|
| **推荐指数** | 整体推荐程度 | 超推/推荐/一般/不推荐 |
| **情感倾向** | 实体整体情感印象 | 正面/中性/负面 |
| **特征标签** | 氛围/定位/体验特征 | 氛围感/网红/文艺/复古/小众/老字号/连锁/文创/高端/热门/宝藏/打卡圣地/服务好/环境好/性价比高/交通便利（约16个） |

**注**：交通方式、交通便利度、消费档次等已删除，由外部数据补充。

---

## 属性详细说明

### 方位关系属性（合并原相邻+距离+方向）

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **距离值** | 枚举（可选） | 距离远近程度 | 近/中等/远 |
| **方向值** | 枚举（可选） | 方位方向 | 东/南/西/北/东北/西南/东侧/西侧/对面/旁边 |
| **联动推荐** | 布尔（可选） | 是否推荐联动游览 | true/false |

### 具有功能关系属性

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **时段** | 文本（可选） | 功能适用时段 | 周末/晚上/樱花季/春季/夏季等 |
| **适合人群** | 枚举（可选） | 功能适合人群 | 亲子/宝妈/学生党/情侣/打工人/特种兵/银发族/宠物主/独行者/团建 |
| **具有限制** | 列表（可选） | 功能限制条件 | 需预约/排队久/停车难/限流/谢绝宠物/只收现金/时间限制/人数限制/消费门槛/季节限制 |
| **情感倾向** | 枚举（可选） | 功能体验情感 | 正面/中性/负面 |

### 功能节点枚举（具有功能的tail，v3.2精简版：9大类）

| 类型 | 说明 | 社交媒体频率 |
|------|------|-------------|
| **餐饮** | 吃饭、探店、下午茶等餐饮活动 | 高频 |
| **购物** | 逛街、买东西等消费活动 | 高频 |
| **休闲** | 游玩、散步、放松等休闲活动 | 高频 |
| **社交** | 聚会、打卡、约会等社交活动 | 高频 |
| **观景** | 赏花、观展、拍照等观赏活动 | 高频 |
| **住宿** | 住酒店、民宿体验等住宿活动 | 中频 |
| **文化** | 学习、体验、参观等文化活动 | 中频 |
| **工作** | 办公、产业等工作相关 | 低频 |
| **其他** | 无法归类的功能 | 兜底 |

### 对比关系属性

| 属性 | 类型 | 说明 | 枚举值 |
|------|------|------|--------|
| **维度** | 列表 | 对比的方面 | 价格/环境/服务/人流量/品质/交通/口味（7个） |

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

### 示例1：空间关系+方位关系
输入文本: "街道口商圈里面有群光广场、银泰城，逛完可以去吃饭"
已知实体: {{\"道路\": [], \"POI\": [\"群光广场\", \"银泰城\"], \"建筑物\": [], \"街区\": [\"街道口商圈\"]}}
输出:
{{\"triples\": [
    {{\"head\": \"街道口商圈\", \"relation\": \"包含\", \"tail\": \"群光广场\", \"evidence\": \"里面有群光广场\"}},
    {{\"head\": \"街道口商圈\", \"relation\": \"包含\", \"tail\": \"银泰城\", \"evidence\": \"里面有银泰城\"}},
    {{\"head\": \"群光广场\", \"relation\": \"方位\", \"tail\": \"银泰城\", \"evidence\": \"一起在商圈里\", \"attributes\": {{\"联动推荐\": true}}}}
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

### 示例5：方位关系（合并距离+方向）
输入文本: "咖啡厅就在地铁站附近，对面是书店"
已知实体: {{\"道路\": [], \"POI\": [\"咖啡厅\", \"地铁站\", \"书店\"], \"建筑物\": [], \"街区\": []}}
输出:
{{\"triples\": [
    {{\"head\": \"咖啡厅\", \"relation\": \"方位\", \"tail\": \"地铁站\", \"evidence\": \"就在地铁站附近\", \"attributes\": {{\"距离值\": \"近\"}}}},
    {{\"head\": \"书店\", \"relation\": \"方位\", \"tail\": \"咖啡厅\", \"evidence\": \"对面是书店\", \"attributes\": {{\"方向值\": \"对面\"}}}}
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


# ===== Step 4: 属性标注提示词模板（v3.0精简版） =====

LABEL_SYSTEM = """你是一位"地理知识管理专家"，精通GIS标准、城市规划术语和社交媒体语义分析。
你的任务是将初步知识片段转化为具备专业语义背景的结构化知识。"""

LABEL_USER = """## 任务描述
请为已识别的实体和关系打上专业属性标签（v3.2精简版）。

---

## 实体属性标注（v3.2精简版，仅5个属性）

### 基础分类属性（必须有原文依据）

| 属性 | 枚举值 |
|------|--------|
| 类别 | 道路、POI、建筑物、街区 |
| 细分 | 餐饮/交通/教育/历史保护/购物/医疗/娱乐/文化/酒店/服务（POI）；商业综合体/住宅/办公楼/文化设施/教育设施/医疗设施（建筑物）；商圈/校区/社区/行政区/景区（街区）；主干道/次干道/支路/小巷/地铁线路（道路） |

### 文本属性（从语料提取，必须有原文依据）

| 属性 | 枚举值 |
|------|--------|
| 特征标签 | 氛围感、网红、文艺、复古、小众、老字号、连锁、文创、高端、热门、宝藏、打卡圣地、服务好、环境好、性价比高、交通便利（多选，约16个） |
| 推荐指数 | 超推、推荐、一般、不推荐 |
| 情感倾向 | 正面、中性、负面 |

**注意**：所有属性必须有原文依据（明确出现、暗示表达、语义推断）。禁止凭空创造（幻觉）。

---

## 关系属性标注（v3.2精简版，与TripleAttributes一致）

| 属性 | 适用关系 | 枚举值/类型 |
|------|----------|-------------|
| 距离值 | 方位 | 近/中等/远 |
| 方向值 | 方位 | 东/南/西/北/东北/西南/东侧/西侧/对面/旁边 |
| 联动推荐 | 方位 | true/false |
| 时段 | 具有功能 | 周末/晚上/樱花季/春季/夏季等（文本） |
| 适合人群 | 具有功能 | 亲子/宝妈/学生党/情侣/打工人/特种兵/银发族/宠物主/独行者/团建 |
| 具有限制 | 具有功能 | 需预约/排队久/停车难/限流/谢绝宠物/只收现金/时间限制/人数限制/消费门槛/季节限制（多选） |
| 情感倾向 | 具有功能 | 正面/中性/负面 |
| 维度 | 优于、相似、劣于 | 价格/环境/服务/人流量/品质/交通/口味（多选，7个） |

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
      \"特征标签\": [\"网红\", \"热门\"],
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


# ===== P9新增：联合抽取提示词（v3.0精简版：13种关系） =====

JOINT_NER_RE_SYSTEM = """你是一位"地理语义联合抽取专家"，擅长在一次推理中同时识别实体和关系。
你的优势在于：能够全局理解文本，避免实体边界识别错误对关系判定的干扰。"""

JOINT_NER_RE_USER = """## 任务描述
请从文本中**同时**抽取：
1. 地理实体（道路/POI/建筑物/街区）
2. 实体间的语义关系（8种关系类型）
3. 每个抽取的证据依据

## 实体类型定义
| 类型 | 定义 | 示例 |
|------|------|------|
| 道路 | 交通通道 | 珞喻路、关山大道 |
| POI | 具体地点 | 武汉大学、群光广场 |
| 建筑物 | 建筑设施 | 泛悦汇、融科天城 |
| 街区 | 地理区域 | 街道口、光谷商圈 |

## 关系类型（v3.2精简版：8种）
### 空间基础关系（3个）—— 图谱骨架
| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| 位于 | A坐落于B处（空间定位/归属） | 道路/街区/行政区 | 无 |
| 包含 | A空间包含B（位于的反向） | POI/建筑物/道路 | 无 |
| 方位 | A和B空间邻近+方位关系 | 地理实体 | **距离值**+**方向值**+**联动推荐**（可选） |

**注**：原"相邻"、"距离"、"方向"已合并为"方位"关系。

### 社交语义关系（1个）—— 图谱血肉
| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| 具有功能 | 场所可进行的功能用途 | 功能节点（9大类） | **时段**+**适合人群**+**具有限制**+**情感倾向**（可选） |

**注**：原"承载活动"改为"具有功能"；"推荐指数"和"引发情感"已改为实体属性。

### 对比评价关系（3个）—— 特色
| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| 优于 | A在某方面好于B | 地理实体 | **维度**（列表） |
| 相似 | A和B在某方面相似 | 地理实体 | **维度**（列表） |
| 劣于 | A在某方面不如B | 地理实体 | **维度**（列表） |

### 事件关系（1个）
| 关系 | 语义定义 | Tail类型 | 关系属性 |
|------|----------|----------|----------|
| 发生事件 | 场所发生的特定事件 | 事件节点 | 无（属性全部在事件节点上） |

---

## 重要说明：实体属性而非关系

以下语义应作为**实体属性**而非三元组关系抽取：

| 属性 | 枚举值 |
|------|--------|
| 推荐指数 | 超推/推荐/一般/不推荐 |
| 情感倾向 | 正面/中性/负面 |
| 特征标签 | 氛围感/网红/文艺/复古/小众/老字号/连锁/文创/高端/热门/宝藏/打卡圣地/服务好/环境好/性价比高/交通便利（约16个） |

**注**：交通方式、交通便利度、消费档次等已删除，由外部数据补充。

---

## 属性详细说明

### 方位关系属性（合并原相邻+距离+方向）
| 属性 | 枚举值 |
|------|--------|
| 距离值 | 近/中等/远 |
| 方向值 | 东/南/西/北/东北/西南/东侧/西侧/对面/旁边 |
| 联动推荐 | true/false |

### 具有功能关系属性
| 属性 | 枚举值 |
|------|--------|
| 时段 | 周末/晚上/樱花季/春季/夏季等 |
| 适合人群 | 亲子/宝妈/学生党/情侣/打工人/特种兵/银发族/宠物主/独行者/团建 |
| 具有限制 | 需预约/排队久/停车难/限流/谢绝宠物/只收现金/时间限制/人数限制/消费门槛/季节限制 |
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
| 维度 | 价格/环境/服务/人流量/品质/交通/口味（7个） |

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
    {{\"head": "群光广场", "relation": "方位", "tail": "银泰城", "evidence": "一起在商圈里", "confidence": "high", "attributes": {{\"联动推荐\": true}}}}
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

### 1. 实体属性验证（Schema v3.2）
- 类别、细分是否正确？（道路/POI/建筑物/街区）
- 特征标签是否与原文匹配？（氛围感/网红/热门等）
- 推荐指数是否合理？（超推/推荐/一般/不推荐）
- 情感倾向是否准确？（正面/中性/负面）

### 2. 关系属性验证（Schema v3.2）
- 方位关系属性：距离值、方向值、联动推荐是否正确？
- 功能关系属性：时段、适合人群、具有限制、情感倾向是否合理？
- 对比关系属性：维度列表是否准确？（价格/环境/服务等7个维度）
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

## 关系类型（v3.2精简版：8种）

### 空间基础关系（3个）—— 图谱骨架
- **位于**：A坐落于B处（如：武汉大学 位于 珞喻路）
- **包含**：A空间包含B（如：街道口 包含 群光广场）
- **方位**：A和B空间邻近+方位关系（属性：距离值/方向值/联动推荐）

**注**：原"相邻"、"距离"、"方向"已合并为"方位"关系。

### 社交语义关系（1个）—— 图谱血肉
- **具有功能**：场所可进行的功能用途（属性：时段/适合人群/限制/情感倾向）

**注**：原"承载活动"改为"具有功能"；"推荐指数"和"引发情感"已改为实体属性。

### 对比评价关系（3个）—— 特色
- **优于**：A在某方面好于B（属性：维度列表）
- **相似**：A和B在某方面相似
- **劣于**：A在某方面不如B

### 事件关系（1个）
- **发生事件**：场所发生的特定事件（属性全部在事件节点上）

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

QA_MENTOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_MENTOR_SYSTEM),
    ("human", QA_MENTOR_USER),
])


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

QA_APPROVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_APPROVAL_SYSTEM),
    ("human", QA_APPROVAL_USER),
])


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

REVISION_JOINT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REVISION_JOINT_SYSTEM),
    ("human", REVISION_JOINT_USER),
])


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

    return f"实体:\n  " + "\n  ".join(entity_lines) + f"\n三元组:\n  " + "\n  ".join(triple_lines)


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

ENTITY_ALIGNMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ENTITY_ALIGNMENT_SYSTEM),
    ("human", ENTITY_ALIGNMENT_USER),
])


def format_alignment_candidates(candidates: List[Dict]) -> str:
    """格式化实体对齐候选列表"""
    if not candidates:
        return "(无候选实体)"

    lines = []
    for i, c in enumerate(candidates):
        name = c.get("db_name", "unknown")
        type_ = c.get("db_type", "")
        sim = c.get("similarity", 0.0)
        lon = c.get("longitude")
        lat = c.get("latitude")
        loc_str = f"({lon:.4f}, {lat:.4f})" if lon and lat else "无坐标"
        lines.append(f"候选{i+1}: {name} [{type_}] - 相似度: {sim:.3f} - 位置: {loc_str}")

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