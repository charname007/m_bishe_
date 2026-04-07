"""
Aggregator Agent - 聚合器
负责合并Worker结果、实体去重、别名发现
"""
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher
from loguru import logger

from .state import DistributedState, PhaseEnum, WorkerResult, CorpusState


class Aggregator:
    """聚合器 - 合并Worker结果"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def aggregate(self, state: DistributedState) -> DistributedState:
        """聚合所有Worker结果"""
        logger.info("开始聚合Worker结果...")

        worker_results = state["worker_results"]

        # 收集所有实体和三元组
        all_entities = defaultdict(list)  # {实体名: [出现信息]}
        all_triples = []

        for worker_result in worker_results:
            for corpus_state in worker_result["results"]:
                corpus_id = corpus_state["corpus_id"]

                # 收集实体
                for entity_type, names in corpus_state["entities"].items():
                    for name in names:
                        all_entities[name].append({
                            "corpus_id": corpus_id,
                            "type": entity_type,
                            "attrs": corpus_state["entity_attrs"].get(name, {})
                        })

                # 收集三元组
                for triple in corpus_state["corrected_triples"]:
                    triple["_corpus_id"] = corpus_id
                    all_triples.append(triple)

        # 实体去重和别名发现
        unique_entities, aliases = self._deduplicate_entities(all_entities)

        # 三元组去重
        unique_triples = self._deduplicate_triples(all_triples)

        state["aggregated_entities"] = unique_entities
        state["aggregated_triples"] = unique_triples
        state["entity_aliases"] = aliases
        state["current_phase"] = PhaseEnum.FINALIZE

        logger.info(f"聚合完成: {len(unique_entities)}个实体, {len(unique_triples)}个三元组")

        return state

    def _deduplicate_entities(self, all_entities: Dict) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """实体去重，发现别名"""
        unique_entities = []
        aliases = {}  # {标准名: [别名列表]}
        processed = set()

        entity_names = list(all_entities.keys())

        for i, name in enumerate(entity_names):
            if name in processed:
                continue

            # 找到所有相似的实体名
            similar_names = [name]
            for j, other_name in enumerate(entity_names):
                if i != j and other_name not in processed:
                    if self._is_similar(name, other_name):
                        similar_names.append(other_name)
                        processed.add(other_name)

            processed.add(name)

            # 选择最长的名称作为标准名
            standard_name = max(similar_names, key=len)

            # 收集所有出现信息
            occurrences = []
            entity_type = None
            entity_attrs = {}  # 收集属性
            for n in similar_names:
                for occ in all_entities[n]:
                    occurrences.append(occ)
                    if entity_type is None:
                        entity_type = occ["type"]
                    # 合并属性
                    if occ.get("attrs"):
                        entity_attrs.update(occ["attrs"])

            # 记录别名
            other_names = [n for n in similar_names if n != standard_name]
            if other_names:
                aliases[standard_name] = other_names

            unique_entities.append({
                "name": standard_name,
                "type": entity_type,
                "category": entity_attrs.get("细分", entity_attrs.get("category", "")),
                "aliases": other_names,
                "occurrence_count": len(occurrences),
                "corpus_ids": list(set(o["corpus_id"] for o in occurrences)),
                "attrs": entity_attrs  # 保留完整属性
            })

        return unique_entities, aliases

    def _deduplicate_triples(self, all_triples: List[Dict]) -> List[Dict]:
        """三元组去重，保留评分字段"""
        seen = set()
        unique_triples = []

        for triple in all_triples:
            key = (triple["head"], triple["relation"], triple["tail"])
            if key not in seen:
                seen.add(key)
                unique_triples.append({
                    "head": triple["head"],
                    "relation": triple["relation"],
                    "tail": triple["tail"],
                    "evidence": triple.get("evidence", ""),
                    "corpus_ids": [triple.get("_corpus_id", "")],
                    # 保留评分字段
                    "sem_score": triple.get("sem_score", 0),
                    "fac_score": triple.get("fac_score", 0),
                    "con_score": triple.get("con_score", 0),
                    "passed_eval": triple.get("passed_eval", True)
                })
            else:
                # 更新corpus_ids
                for t in unique_triples:
                    if (t["head"], t["relation"], t["tail"]) == key:
                        if triple.get("_corpus_id") not in t["corpus_ids"]:
                            t["corpus_ids"].append(triple.get("_corpus_id"))
                        break

        return unique_triples

    def _is_similar(self, name1: str, name2: str) -> bool:
        """判断两个名称是否相似（可能是别名）"""
        # 完全相同
        if name1 == name2:
            return True

        # 长度差异太大（绝对差异超过2个字符，且相对比例小于50%）
        len1, len2 = len(name1), len(name2)
        if abs(len1 - len2) > 2:
            # 但允许常见简称模式（如"武大"="武汉大学"）
            len_ratio = min(len1, len2) / max(len1, len2)
            if len_ratio < 0.5:
                return False

        # 编辑距离相似度
        similarity = SequenceMatcher(None, name1, name2).ratio()
        if similarity >= self.similarity_threshold:
            return True

        # 简称别名模式检查
        # 如 "武汉大学" = "武大"，需要满足：
        # 1. 短名称是长名称的子串，或
        # 2. 短名称是长名称的首字缩写
        if len1 != len2:
            shorter, longer = (name1, name2) if len1 < len2 else (name2, name1)

            # 检查短名称是否在长名称中（允许一定的字符差异）
            if shorter in longer:
                # 额外检查：短名称长度应至少是长名称的一半
                # 避免将"汉"识别为"武汉大学"的别名
                if len(shorter) >= len(longer) * 0.4:
                    return True

            # 检查首字缩写模式（如"武大"是"武汉大学"的缩写）
            # 取长名称每个词的首字
            if len(shorter) >= 2:
                first_chars = ''.join(
                    word[0] for word in longer if word
                )
                if shorter == first_chars:
                    return True

        return False


class AggregatorAgent:
    """
    Aggregator Agent - 用于LangGraph工作流
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.aggregator = Aggregator(similarity_threshold)

    def __call__(self, state: DistributedState) -> Dict:
        """LangGraph节点调用入口"""
        result = self.aggregator.aggregate(state)
        return {
            "aggregated_entities": result["aggregated_entities"],
            "aggregated_triples": result["aggregated_triples"],
            "entity_aliases": result["entity_aliases"],
            "current_phase": result["current_phase"]
        }


def create_aggregator_node(similarity_threshold: float = 0.85):
    """创建Aggregator节点"""
    aggregator_agent = AggregatorAgent(similarity_threshold)

    def aggregator_node(state: DistributedState) -> Dict:
        return aggregator_agent(state)

    return aggregator_node