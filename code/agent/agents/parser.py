"""
LLM响应解析工具 - 使用PydanticOutputParser
"""
from typing import Dict, List, Any
from loguru import logger

from .schemas import (
    ner_parser, re_parser,
    eval_first_parser, eval_second_parser,
    label_parser,
    EntityRecognitionResult, RelationExtractionResult,
    EvalResultFirst, EvalResultSecond,
    LabelResult, Triple, TripleScore
)


def parse_ner_response(response: str) -> Dict[str, List[str]]:
    """解析NER响应 - 使用PydanticOutputParser"""
    try:
        result: EntityRecognitionResult = ner_parser.parse(response)
        return {
            "道路": result.道路,
            "POI": result.POI,
            "建筑物": result.建筑物,
            "街区": result.街区
        }
    except Exception as e:
        logger.warning(f"NER解析失败，尝试降级解析: {e}")
        return _fallback_parse_ner(response)


def _fallback_parse_ner(response: str) -> Dict[str, List[str]]:
    """NER降级解析"""
    import json
    import re

    # 尝试提取JSON
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "道路": data.get("道路", data.get("Road", data.get("road", []))),
                "POI": data.get("POI", data.get("poi", [])),
                "建筑物": data.get("建筑物", data.get("Building", data.get("building", []))),
                "街区": data.get("街区", data.get("Block", data.get("block", [])))
            }
    except Exception:
        pass

    return {"道路": [], "POI": [], "建筑物": [], "街区": []}


def parse_re_response(response: str) -> List[Dict]:
    """解析RE响应 - 使用PydanticOutputParser"""
    try:
        result: RelationExtractionResult = re_parser.parse(response)
        return [
            {
                "head": t.head,
                "relation": t.relation,
                "tail": t.tail,
                "evidence": t.evidence or ""
            }
            for t in result.triples
        ]
    except Exception as e:
        logger.warning(f"RE解析失败，尝试降级解析: {e}")
        return _fallback_parse_re(response)


def _fallback_parse_re(response: str) -> List[Dict]:
    """RE降级解析"""
    import json
    import re

    try:
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            data = json.loads(json_match.group())
            triples = []
            for item in data:
                if isinstance(item, dict):
                    triples.append({
                        "head": item.get("head", ""),
                        "relation": item.get("relation", ""),
                        "tail": item.get("tail", ""),
                        "evidence": item.get("evidence", "")
                    })
            return triples
    except Exception:
        pass

    return []


def parse_eval_response_1(response: str) -> List[Dict]:
    """解析第一次评估响应 - 使用PydanticOutputParser"""
    try:
        result: EvalResultFirst = eval_first_parser.parse(response)
        return [
            {
                "triple": {
                    "head": s.triple.head,
                    "relation": s.triple.relation,
                    "tail": s.triple.tail
                },
                "SEM": s.SEM,
                "FAC": s.FAC,
                "CON": s.CON
            }
            for s in result.scores
        ]
    except Exception as e:
        logger.warning(f"Eval1解析失败，尝试降级解析: {e}")
        return _fallback_parse_eval_1(response)


def _fallback_parse_eval_1(response: str) -> List[Dict]:
    """Eval1降级解析"""
    import json
    import re

    try:
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            data = json.loads(json_match.group())
            scores = []
            for item in data:
                if isinstance(item, dict):
                    scores.append({
                        "triple": item.get("triple", {}),
                        "SEM": item.get("SEM", 3),
                        "FAC": item.get("FAC", 3),
                        "CON": item.get("CON", 3)
                    })
            return scores
    except Exception:
        pass

    return []


def parse_eval_response_2(response: str) -> Dict:
    """解析第二次评估响应（自检）- 使用PydanticOutputParser"""
    try:
        result: EvalResultSecond = eval_second_parser.parse(response)
        return {
            "need_correction": result.need_correction,
            "corrections": [
                {
                    "original": {
                        "head": c.original.head,
                        "relation": c.original.relation,
                        "tail": c.original.tail
                    },
                    "corrected": {
                        "head": c.corrected.head,
                        "relation": c.corrected.relation,
                        "tail": c.corrected.tail
                    },
                    "reason": c.reason
                }
                for c in result.corrections
            ],
            "final_scores": [
                {
                    "triple": {
                        "head": s.triple.head,
                        "relation": s.triple.relation,
                        "tail": s.triple.tail
                    },
                    "SEM": s.SEM,
                    "FAC": s.FAC,
                    "CON": s.CON
                }
                for s in result.final_scores
            ]
        }
    except Exception as e:
        logger.warning(f"Eval2解析失败，尝试降级解析: {e}")
        return _fallback_parse_eval_2(response)


def _fallback_parse_eval_2(response: str) -> Dict:
    """Eval2降级解析"""
    import json
    import re

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "need_correction": data.get("need_correction", False),
                "corrections": data.get("corrections", []),
                "final_scores": data.get("final_scores", [])
            }
    except Exception:
        pass

    return {
        "need_correction": False,
        "corrections": [],
        "final_scores": []
    }


def parse_label_response(response: str) -> Dict:
    """解析属性标注响应 - 使用PydanticOutputParser"""
    try:
        result: LabelResult = label_parser.parse(response)
        return {
            "entities": {
                name: {"类别": attrs.类别, "细分": attrs.细分}
                for name, attrs in result.entities.items()
            },
            "relations": {
                key: {"类型": attrs.类型, "细分": attrs.细分}
                for key, attrs in result.relations.items()
            }
        }
    except Exception as e:
        logger.warning(f"Label解析失败，尝试降级解析: {e}")
        return _fallback_parse_label(response)


def _fallback_parse_label(response: str) -> Dict:
    """Label降级解析"""
    import json
    import re

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())

            # 检查是否是新格式
            if "entities" in data and "relations" in data:
                return {
                    "entities": data.get("entities", {}),
                    "relations": data.get("relations", {})
                }

            # 旧格式：直接是实体字典
            is_entity_format = False
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict) and ("类别" in value or "细分" in value):
                        is_entity_format = True
                        break

            if is_entity_format:
                return {
                    "entities": data,
                    "relations": {}
                }

            return {
                "entities": data,
                "relations": {}
            }
    except Exception:
        pass

    return {"entities": {}, "relations": {}}


def calculate_avg_score(scores: List[Dict]) -> float:
    """计算平均评分"""
    if not scores:
        return 0.0

    total = 0
    count = 0
    for s in scores:
        total += s.get("SEM", 0) + s.get("FAC", 0) + s.get("CON", 0)
        count += 3

    return total / count if count > 0 else 0.0


def is_eval_passed(scores: List[Dict], threshold: float = 3.5) -> bool:
    """判断评估是否通过"""
    avg = calculate_avg_score(scores)
    return avg >= threshold