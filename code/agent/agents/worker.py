"""
Worker Agent - 执行四步骤工作流
"""
import time
import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger

from .state import CorpusState, WorkerResult, StepEnum
from .prompts import (
    get_ner_prompt, get_re_prompt,
    get_eval_prompt_1, get_eval_prompt_2, get_label_prompt
)
from .parser import (
    parse_ner_response, parse_re_response,
    parse_eval_response_1, parse_eval_response_2,
    parse_label_response, calculate_avg_score, is_eval_passed
)


class WorkerAgent:
    """Worker Agent - 处理单条或多条语料"""

    def __init__(self, worker_id: str, llm_client: Any):
        self.worker_id = worker_id
        self.llm = llm_client

    async def process_corpus(self, corpus: Dict) -> CorpusState:
        """处理单条语料 - 四步骤工作流"""
        corpus_id = corpus.get("id", "unknown")
        raw_text = corpus.get("text", "")

        logger.info(f"[{self.worker_id}] 开始处理语料: {corpus_id}")

        state: CorpusState = {
            "corpus_id": corpus_id,
            "raw_text": raw_text,
            "entities": {"道路": [], "POI": [], "建筑物": [], "街区": []},
            "triples": [],
            "eval_scores": [],
            "eval_passed": False,
            "corrected_triples": [],
            "entity_attrs": {},
            "relation_attrs": {},
            "current_step": StepEnum.NER,
            "error": None
        }

        try:
            # Step 1: NER
            state = await self._step_ner(state)

            # Step 2: RE
            state = await self._step_re(state)

            # Step 3: Eval（二次对话）
            state = await self._step_eval(state)

            # Step 4: Label
            state = await self._step_label(state)

        except Exception as e:
            logger.error(f"[{self.worker_id}] 处理语料 {corpus_id} 出错: {e}")
            state["error"] = str(e)

        return state

    async def _step_ner(self, state: CorpusState) -> CorpusState:
        """Step 1: 命名实体识别"""
        logger.debug(f"[{self.worker_id}] 执行NER...")

        prompt = get_ner_prompt(state["raw_text"])
        response = await self._call_llm(prompt)
        entities = parse_ner_response(response)

        state["entities"] = entities
        state["current_step"] = StepEnum.RE

        logger.debug(f"[{self.worker_id}] NER结果: {entities}")
        return state

    async def _step_re(self, state: CorpusState) -> CorpusState:
        """Step 2: 关系抽取"""
        logger.debug(f"[{self.worker_id}] 执行RE...")

        # 检查是否有实体
        total_entities = sum(len(v) for v in state["entities"].values())
        if total_entities == 0:
            logger.debug(f"[{self.worker_id}] 无实体，跳过RE")
            state["current_step"] = StepEnum.EVAL
            return state

        prompt = get_re_prompt(state["raw_text"], state["entities"])
        response = await self._call_llm(prompt)
        triples = parse_re_response(response)

        state["triples"] = triples
        state["current_step"] = StepEnum.EVAL

        logger.debug(f"[{self.worker_id}] RE结果: {len(triples)}个三元组")
        return state

    async def _step_eval(self, state: CorpusState) -> CorpusState:
        """Step 3: 三元组评估（二次对话验证）"""
        logger.debug(f"[{self.worker_id}] 执行Eval...")

        if not state["triples"]:
            logger.debug(f"[{self.worker_id}] 无三元组，跳过Eval")
            state["current_step"] = StepEnum.LABEL
            return state

        # 第一次对话：初步评分
        prompt1 = get_eval_prompt_1(state["triples"], state["raw_text"])
        response1 = await self._call_llm(prompt1)
        scores = parse_eval_response_1(response1)
        state["eval_scores"] = scores

        # 第二次对话：自检验证
        prompt2 = get_eval_prompt_2(scores, state["raw_text"])
        response2 = await self._call_llm(prompt2)
        eval_result = parse_eval_response_2(response2)

        # 更新评分（使用最终评分）
        if eval_result["final_scores"]:
            state["eval_scores"] = eval_result["final_scores"]

        # 创建评分查找字典
        score_map = {}
        for score_item in state["eval_scores"]:
            triple_key = (
                score_item.get("triple", {}).get("head", ""),
                score_item.get("triple", {}).get("relation", ""),
                score_item.get("triple", {}).get("tail", "")
            )
            score_map[triple_key] = {
                "sem_score": score_item.get("SEM", 0),
                "fac_score": score_item.get("FAC", 0),
                "con_score": score_item.get("CON", 0)
            }

        # 应用修正并将评分写入三元组
        if eval_result["need_correction"] and eval_result["corrections"]:
            corrected_triples = self._apply_corrections(
                state["triples"],
                eval_result["corrections"]
            )
        else:
            corrected_triples = state["triples"]

        # 将评分写入每个三元组
        for triple in corrected_triples:
            triple_key = (triple.get("head", ""), triple.get("relation", ""), triple.get("tail", ""))
            scores_for_triple = score_map.get(triple_key, {})
            triple["sem_score"] = scores_for_triple.get("sem_score", 0)
            triple["fac_score"] = scores_for_triple.get("fac_score", 0)
            triple["con_score"] = scores_for_triple.get("con_score", 0)

        state["corrected_triples"] = corrected_triples
        state["eval_passed"] = is_eval_passed(state["eval_scores"])
        state["current_step"] = StepEnum.LABEL

        logger.debug(f"[{self.worker_id}] Eval通过: {state['eval_passed']}")
        return state

    async def _step_label(self, state: CorpusState) -> CorpusState:
        """Step 4: 属性标注"""
        logger.debug(f"[{self.worker_id}] 执行Label...")

        # 收集所有实体名称
        all_entities = []
        for entity_list in state["entities"].values():
            all_entities.extend(entity_list)

        if not all_entities:
            logger.debug(f"[{self.worker_id}] 无实体，跳过Label")
            state["current_step"] = StepEnum.DONE
            return state

        prompt = get_label_prompt(all_entities, state["corrected_triples"])
        response = await self._call_llm(prompt)
        attrs = parse_label_response(response)

        state["entity_attrs"] = attrs.get("entities", {})
        state["relation_attrs"] = attrs.get("relations", {})
        state["current_step"] = StepEnum.DONE

        logger.debug(f"[{self.worker_id}] Label完成")
        return state

    async def _call_llm(self, prompt: str) -> str:
        """调用LLM（异步非阻塞）"""
        # 优先使用异步方法
        if hasattr(self.llm, 'ainvoke'):
            response = await self.llm.ainvoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        elif hasattr(self.llm, 'invoke'):
            # 同步方法包装为异步，避免阻塞事件循环
            import asyncio
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        else:
            raise ValueError("LLM客户端不支持invoke方法")

    def _apply_corrections(self, original_triples: List[Dict],
                           corrections: List[Dict]) -> List[Dict]:
        """应用修正"""
        corrected = list(original_triples)
        for correction in corrections:
            original = correction.get("original", {})
            new_triple = correction.get("corrected", {})

            # 找到并替换原始三元组
            for i, triple in enumerate(corrected):
                if (triple.get("head") == original.get("head") and
                    triple.get("relation") == original.get("relation") and
                    triple.get("tail") == original.get("tail")):
                    corrected[i] = new_triple
                    break

        return corrected

    async def process_batch(self, corpus_list: List[Dict]) -> WorkerResult:
        """批量处理语料（并行）"""
        start_time = time.time()

        # 并行处理所有语料
        tasks = [self.process_corpus(corpus) for corpus in corpus_list]
        results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

        # 分离结果和错误
        results = []
        errors = []
        for i, result in enumerate(results_or_errors):
            if isinstance(result, Exception):
                errors.append(str(result))
                logger.error(f"[{self.worker_id}] 处理语料失败: {result}")
            else:
                results.append(result)

        processing_time = time.time() - start_time

        return WorkerResult(
            worker_id=self.worker_id,
            corpus_ids=[c.get("id", "unknown") for c in corpus_list],
            results=results,
            processing_time=processing_time,
            error="; ".join(errors) if errors else None
        )


async def run_worker(worker_id: str, corpus_list: List[Dict],
                     llm_client: Any) -> WorkerResult:
    """运行单个Worker"""
    worker = WorkerAgent(worker_id, llm_client)
    return await worker.process_batch(corpus_list)