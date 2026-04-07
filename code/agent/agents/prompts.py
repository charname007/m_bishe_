"""
四步骤智能体工作流提示词模板
使用PydanticOutputParser进行结构化输出
"""
from typing import Dict, List

from .schemas import (
    ner_parser, re_parser,
    eval_first_parser, eval_second_parser,
    label_parser
)


# ===== Step 1: NER 提示词模板 =====
NER_PROMPT = '''# System
你是一位"地理语义专家"，精通城市地理实体识别与社交媒体语料分析。你的任务是从以下小红书文本中提取地理知识实体。

# User
## 候选目标
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
输出: {{"道路": [], "POI": ["书店"], "建筑物": ["泛悦汇"], "街区": ["街道口"]}}

## 输出格式要求
{format_instructions}

## 待处理文本
{raw_text}

# 请输出实体识别结果'''


# ===== Step 2: RE 提示词模板 =====
RE_PROMPT = '''# System
你是一位"地理语义专家"，擅长梳理非结构化文本中的语义逻辑。

# User
## 候选目标
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
已知实体: {{"道路": [], "POI": ["武汉大学", "行政楼"], "建筑物": [], "街区": []}}
输出: {{"triples": [{{"head": "行政楼", "relation": "属于", "tail": "武汉大学", "evidence": "行政楼在武汉大学内"}}, {{"head": "行政楼", "relation": "承载活动", "tail": "合影", "evidence": "在行政楼前合影"}}]}}

## 输出格式要求
{format_instructions}

## 已识别实体
{entities}

## 待处理文本
{raw_text}

# 请输出关系抽取结果'''


# ===== Step 3: 三元组评估提示词模板 =====
EVAL_PROMPT_1 = '''# System
你是一位"地理语义评审专家"。你的任务是对生成的地理关系三元组进行真实性与逻辑性评估。

# User
## 评估维度
- 语义准确性(SEM): 三元组是否准确代表了原文意思？（1-5分）
- 事实真实性(FAC): 是否符合地理常识？（1-5分）
- 一致性(CON): 关系方向是否正确？（1-5分）

评分标准：
- 5分: 完全正确
- 4分: 基本正确，有小瑕疵
- 3分: 可接受
- 2分: 有问题
- 1分: 错误

## 输出格式要求
{format_instructions}

## 待评估三元组
{triples}

## 原始文本
{raw_text}

# 请输出评分结果'''


EVAL_PROMPT_2 = '''# System
你是同一位"地理语义评审专家"，现在进行二次验证。

# User
## 任务
请结合原始语料，重新验证你刚才的评分：
1. 检查是否存在"虚假幻觉"（三元组在原文中不存在依据）
2. 检查是否存在"逻辑偏差"（关系方向或类型错误）

## 任务强调
如果发现错误，请给出修正后的三元组。

## 输出格式要求
{format_instructions}

## 你刚才的评分结果
{previous_scores}

## 原始语料
{raw_text}

# 请输出二次验证结果'''


# ===== Step 4: 属性标注提示词模板 =====
LABEL_PROMPT = '''# System
你是一位"地理知识管理专家"。

# User
## 任务描述
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

## 输出格式要求
{format_instructions}

## 任务示例
输入实体: ["武汉大学", "行政楼", "街道口"]
输入关系: [<武汉大学, 包含, 行政楼>, <行政楼, 承载活动, 合影>]

输出:
{{
  "entities": {{
    "武汉大学": {{"类别": "POI", "细分": "教育"}},
    "行政楼": {{"类别": "建筑物", "细分": "教育设施"}},
    "街道口": {{"类别": "街区", "细分": "商圈"}}
  }},
  "relations": {{
    "<武汉大学, 包含, 行政楼>": {{"类型": "位置关系", "细分": "内部"}},
    "<行政楼, 承载活动, 合影>": {{"类型": "活动关系", "细分": "体验"}}
  }}
}}

## 待标注实体
{entities}

## 待标注关系
{relations}

# 请输出属性标注结果'''


# ===== 批量处理提示词模板 =====
BATCH_NER_RE_PROMPT = '''# System
你是一位"地理语义专家"，正在批量处理多条语料。

# User
## 任务
请对以下每条语料依次执行：
1. 实体识别（道路/POI/建筑物/街区）
2. 关系抽取（连接/位于/承载活动/引发情感/属于）

## 语料列表
{corpus_list}

# 请输出JSON数组，每条语料一个对象'''


# ===== 辅助函数 =====

def format_corpus_list(corpus_list: List[Dict]) -> str:
    """格式化语料列表用于批量提示词"""
    lines = []
    for i, corpus in enumerate(corpus_list, 1):
        lines.append(f"[{i}] ID: {corpus.get('id', 'unknown')}")
        lines.append(f"    文本: {corpus['text']}")
        lines.append("")
    return "\n".join(lines)


def format_entities(entities: Dict[str, List[str]]) -> str:
    """格式化实体字典用于提示词"""
    result = []
    for entity_type, names in entities.items():
        if names:
            result.append(f"- {entity_type}: {', '.join(names)}")
        else:
            result.append(f"- {entity_type}: (无)")
    return "\n".join(result)


def format_triples(triples: List[Dict]) -> str:
    """格式化三元组列表用于提示词"""
    if not triples:
        return "(无三元组)"
    lines = []
    for t in triples:
        lines.append(f"- <{t['head']}, {t['relation']}, {t['tail']}>")
    return "\n".join(lines)


def get_ner_prompt(raw_text: str) -> str:
    """获取完整的NER提示词"""
    return NER_PROMPT.format(
        raw_text=raw_text,
        format_instructions=ner_parser.get_format_instructions()
    )


def get_re_prompt(raw_text: str, entities: Dict[str, List[str]]) -> str:
    """获取完整的RE提示词"""
    return RE_PROMPT.format(
        raw_text=raw_text,
        entities=format_entities(entities),
        format_instructions=re_parser.get_format_instructions()
    )


def get_eval_prompt_1(triples: List[Dict], raw_text: str) -> str:
    """获取完整的第一次评估提示词"""
    return EVAL_PROMPT_1.format(
        triples=triples,
        raw_text=raw_text,
        format_instructions=eval_first_parser.get_format_instructions()
    )


def get_eval_prompt_2(previous_scores: List[Dict], raw_text: str) -> str:
    """获取完整的第二次评估提示词"""
    return EVAL_PROMPT_2.format(
        previous_scores=previous_scores,
        raw_text=raw_text,
        format_instructions=eval_second_parser.get_format_instructions()
    )


def get_label_prompt(entities: List[str], relations: List[Dict]) -> str:
    """获取完整的属性标注提示词"""
    return LABEL_PROMPT.format(
        entities=entities,
        relations=relations,
        format_instructions=label_parser.get_format_instructions()
    )