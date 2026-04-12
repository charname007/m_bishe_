"""
四步骤智能体工作流提示词模板 - 使用LangChain ChatPromptTemplate
P2改进：简化评估提示词，单次评估包含评分和修正
P5改进：添加 Filter 筛选提示词
"""
from langchain_core.prompts import ChatPromptTemplate


# ===== Step 0: Filter 筛选提示词模板（P5新增） =====

FILTER_SYSTEM = """你是一位"地理文本筛选专家"，负责快速判断文本是否值得处理。
你的任务是高效筛选，识别包含地理信息的文本，跳过无价值文本以节省处理成本。"""

FILTER_USER = """## 快速筛选标准

**有效文本（is_valid=true）**：
- 提及地理实体：道路、POI、建筑物、街区、地名等
- 涉及空间关系：位于、旁边、连接、附近、在...内等
- 地理相关活动：逛街、打卡、拍照、游玩等（暗示地点）
- 即使主语省略，但有地理暗示（如"这里的樱花很好看"）

**无效文本（is_valid=false）**：
- 过短文本：少于5个有效字符
- 无地理信息：纯情感表达、无关话题、纯表情/乱码
- 纯抽象内容：时间、数字、无地点的活动描述

## 边界模糊处理
- 如果判断困难，返回 confidence="low"，让后续流程处理
- 有地理暗示但无明确实体时，建议保留（is_valid=true, confidence="low")

## 任务示例

示例1:
输入: "武汉大学在珞喻路上，樱花开了很漂亮"
输出: {{
  "is_valid": true,
  "confidence": "high",
  "has_geo_entity": true,
  "has_spatial_relation": true,
  "geo_entity_hint": "武汉大学、珞喻路"
}}

示例2:
输入: "今天心情不好"
输出: {{
  "is_valid": false,
  "skip_reason": "无地理信息，纯情感表达",
  "confidence": "high",
  "has_geo_entity": false,
  "has_spatial_relation": false
}}

示例3:
输入: "😂😂😂太好笑了"
输出: {{
  "is_valid": false,
  "skip_reason": "过短，纯表情，无语义内容",
  "confidence": "high",
  "has_geo_entity": false,
  "has_spatial_relation": false
}}

示例4:
输入: "这里挺好玩"
输出: {{
  "is_valid": true,
  "confidence": "low",
  "has_geo_entity": false,
  "has_spatial_relation": true,
  "geo_entity_hint": "这里（模糊地点指代）"
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


# ===== Step 2: RE 提示词模板 =====

RE_SYSTEM = """你是一位"地理语义专家"，擅长梳理非结构化文本中的语义逻辑。"""

RE_USER = """## 候选目标
请识别实体间的以下三元组关系：<头实体, 关系, 尾实体>
关系集包含：[连接, 位于, 承载活动, 引发情感, 属于]

关系说明：
- 连接: A和B通过道路/交通连接
- 位于: A在B的内部或附近
- 承载活动: A场所发生B活动
- 引发情感: A引发B情感（如"好看"、"不错"）
- 属于: A属于B的组成部分

## 思维链(CoT)
1. 观察已识别的实体对
2. 分析句中的动词或形容词，判断其反映的是物理位置还是社会意义关联
3. 若句子存在主语省略（如"拍照很好看"），请结合段落背景推断主语

## 任务示例
输入文本: "武汉大学的樱花开了，大家都在行政楼前合影。"
已知实体: {{\"道路\": [], \"POI\": [\"武汉大学\", \"行政楼\"], \"建筑物\": [], \"街区\": []}}
输出: {{\"triples\": [{{\"head\": \"行政楼\", \"relation\": \"属于\", \"tail\": \"武汉大学\", \"evidence\": \"行政楼在武汉大学内\"}}, {{\"head\": \"行政楼\", \"relation\": \"承载活动\", \"tail\": \"合影\", \"evidence\": \"在行政楼前合影\"}}]}}

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

## 待评估三元组
{triples}

## 原始文本
{raw_text}

请输出评估结果（JSON格式），包含评分和可选修正。"""

EVAL_PROMPT_SIMPLIFIED = ChatPromptTemplate.from_messages([
    ("system", EVAL_SIMPLIFIED_SYSTEM),
    ("human", EVAL_SIMPLIFIED_USER),
])


# ===== Step 4: 属性标注提示词模板 =====

LABEL_SYSTEM = """你是一位"地理知识管理专家"。"""

LABEL_USER = """## 任务描述
请为已确定的实体和关系打上细粒度标签。

## 实体标签类别
POI细分: [餐饮, 交通, 教育, 历史保护, 购物, 医疗, 娱乐, 文化, 酒店, 服务]
建筑物细分: [商业综合体, 住宅, 办公楼, 文化设施, 教育设施, 医疗设施]
街区细分: [商圈, 校区, 社区, 行政区, 景区]
道路细分: [主干道, 次干道, 支路, 小巷, 地铁线路]

## 关系标签
活动关系细分: [开发, 保护, 解释, 评价, 体验]
位置关系细分: [内部, 邻近, 沿线, 跨越]
连接关系细分: [直达, 换乘, 途径]

## 任务示例
输入实体: ["武汉大学", "行政楼", "街道口"]
输入关系: [<武汉大学, 包含, 行政楼>, <行政楼, 承载活动, 合影>]

输出:
{{
  "entities": {{
    "武汉大学": {{\"类别\": \"POI\", \"细分\": \"教育\"}},
    "行政楼": {{\"类别\": \"建筑物\", \"细分\": \"教育设施\"}},
    "街道口": {{\"类别\": \"街区\", \"细分\": \"商圈\"}}
  }},
  "relations": {{
    "<武汉大学, 包含, 行政楼>": {{\"类型\": \"位置关系\", \"细分\": \"内部\"}},
    "<行政楼, 承载活动, 合影>": {{\"类型\": \"活动关系\", \"细分\": \"体验\"}}
  }}
}}

## 待标注实体
{entities}

## 待标注关系
{relations}

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


def format_triples(triples: list) -> str:
    """格式化三元组列表用于提示词"""
    if not triples:
        return "(无三元组)"
    lines = []
    for t in triples:
        lines.append(f"- <{t['head']}, {t['relation']}, {t['tail']}>")
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