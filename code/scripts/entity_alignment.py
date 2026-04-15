"""
实体对齐脚本
将现有 geo_entity_names 数据与高德地图数据进行匹配关联

匹配策略：
- 语义匹配：使用 embedding 计算名称相似度
- 坐标匹配：计算地理位置距离
- 组合匹配：语义相似度 + 坐标距离 → 综合置信度

输出：
- entity_alignment 表：存储匹配关系
- 统计报告：匹配数量、置信度分布

使用方式：
    python scripts/entity_alignment.py
    python scripts/entity_alignment.py --threshold 0.8  # 调整相似度阈值
    python scripts/entity_alignment.py --geo-threshold 100  # 调整距离阈值(米)
"""
import os
import sys
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger

# 设置 ModelScope 镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
from kg.postgres_client import PostgresClient
from psycopg2.extras import execute_values


# ============================================================
# 实体对齐配置
# ============================================================

class AlignmentConfig:
    """对齐配置"""
    # 语义相似度阈值
    SEMANTIC_THRESHOLD_HIGH = 0.90    # 高置信度
    SEMANTIC_THRESHOLD_MEDIUM = 0.85  # 中置信度
    SEMANTIC_THRESHOLD_LOW = 0.75     # 低置信度（仅作为候选）

    # 地理距离阈值（米）
    GEO_THRESHOLD_HIGH = 50           # 高置信度
    GEO_THRESHOLD_MEDIUM = 100        # 中置信度
    GEO_THRESHOLD_LOW = 200           # 低置信度

    # 综合置信度计算权重
    SEMANTIC_WEIGHT = 0.6
    GEO_WEIGHT = 0.4


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    计算两点间的地理距离（Haversine公式）

    Args:
        lon1, lat1: 第一个点的经纬度
        lon2, lat2: 第二个点的经纬度

    Returns:
        距离（米）
    """
    if None in [lon1, lat1, lon2, lat2]:
        return float('inf')

    # 地球半径（米）
    R = 6371000

    # 转换为弧度
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine公式
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def get_embedding_model(model_name: str = None):
    """加载嵌入模型"""
    from sentence_transformers import SentenceTransformer

    if model_name is None:
        model_name = settings.get_embedding_config()["model"]

    logger.info(f"加载嵌入模型: {model_name}")
    return SentenceTransformer(model_name)


class EntityAligner:
    """实体对齐器"""

    def __init__(
        self,
        pg_client: PostgresClient,
        semantic_threshold: float = AlignmentConfig.SEMANTIC_THRESHOLD_MEDIUM,
        geo_threshold: float = AlignmentConfig.GEO_THRESHOLD_MEDIUM
    ):
        """
        初始化对齐器

        Args:
            pg_client: PostgreSQL客户端
            semantic_threshold: 语义相似度阈值
            geo_threshold: 地理距离阈值（米）
        """
        self.pg_client = pg_client
        self.semantic_threshold = semantic_threshold
        self.geo_threshold = geo_threshold

        # 加载模型
        self.model = get_embedding_model()

        # 缓存数据
        self.existing_entities = {}   # 现有实体（非高德）
        self.amap_entities = {}       # 高德实体
        self.existing_embeddings = {} # 现有实体embedding
        self.amap_embeddings = {}     # 高德实体embedding

    def load_entities(self):
        """从数据库加载实体数据"""
        with self.pg_client.conn.cursor() as cur:
            # 加载现有实体（非高德来源）
            cur.execute("""
                SELECT entity_id, name, type, longitude, latitude, embedding
                FROM geo_entity_names
                WHERE entity_id NOT LIKE 'amap_%'
            """)
            for row in cur.fetchall():
                eid, name, typ, lon, lat, emb = row
                self.existing_entities[eid] = {
                    "name": name, "type": typ, "lon": lon, "lat": lat
                }
                if emb:
                    self.existing_embeddings[eid] = emb

            # 加载高德实体
            cur.execute("""
                SELECT entity_id, name, type, longitude, latitude, embedding
                FROM geo_entity_names
                WHERE entity_id LIKE 'amap_%'
            """)
            for row in cur.fetchall():
                eid, name, typ, lon, lat, emb = row
                self.amap_entities[eid] = {
                    "name": name, "type": typ, "lon": lon, "lat": lat
                }
                if emb:
                    self.amap_embeddings[eid] = emb

        logger.info(f"现有实体: {len(self.existing_entities)} 条（有embedding: {len(self.existing_embeddings)}）")
        logger.info(f"高德实体: {len(self.amap_entities)} 条（有embedding: {len(self.amap_embeddings)}）")

    def compute_missing_embeddings(self):
        """为缺少embedding的实体计算向量"""
        # 检查现有实体
        missing_existing = [
            (eid, ent["name"])
            for eid, ent in self.existing_entities.items()
            if eid not in self.existing_embeddings
        ]

        # 检查高德实体
        missing_amap = [
            (eid, ent["name"])
            for eid, ent in self.amap_entities.items()
            if eid not in self.amap_embeddings
        ]

        total_missing = len(missing_existing) + len(missing_amap)
        if total_missing == 0:
            logger.info("所有实体已有embedding，无需计算")
            return

        logger.info(f"需要计算embedding的实体: {total_missing} 条")

        # 批量计算
        if missing_existing:
            logger.info("计算现有实体embedding...")
            names = [n for _, n in missing_existing]
            embeddings = self.model.encode(names, show_progress_bar=True)

            for (eid, _), emb in zip(missing_existing, embeddings):
                self.existing_embeddings[eid] = emb.tolist()

            # 更新数据库
            self._update_embeddings(missing_existing, embeddings)

        if missing_amap:
            logger.info("计算高德实体embedding...")
            names = [n for _, n in missing_amap]
            embeddings = self.model.encode(names, show_progress_bar=True)

            for (eid, _), emb in zip(missing_amap, embeddings):
                self.amap_embeddings[eid] = emb.tolist()

            # 更新数据库
            self._update_embeddings(missing_amap, embeddings)

    def _update_embeddings(self, entities: List[Tuple], embeddings):
        """更新数据库中的embedding"""
        data = [(eid, emb.tolist()) for (eid, _), emb in zip(entities, embeddings)]

        with self.pg_client.conn.cursor() as cur:
            execute_values(
                cur,
                """
                UPDATE geo_entity_names SET embedding = data.embedding::vector
                FROM (VALUES %s) AS data (id VARCHAR(100), embedding FLOAT[])
                WHERE geo_entity_names.entity_id = data.id
                """,
                data,
                template="(%s, %s)"
            )
            self.pg_client.conn.commit()

        logger.success(f"更新embedding: {len(data)} 条")

    def semantic_similarity(self, emb1, emb2) -> float:
        """计算两个embedding的余弦相似度"""
        import numpy as np

        if emb1 is None or emb2 is None:
            return 0.0

        # 转换为numpy数组
        v1 = np.array(emb1)
        v2 = np.array(emb2)

        # 余弦相似度
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)

        return dot / norm if norm > 0 else 0.0

    def find_matches(self) -> List[Dict]:
        """
        找出所有匹配对

        Returns:
            匹配列表: [{"source_id", "amap_id", "semantic_sim", "geo_dist", "confidence", "match_type"}]
        """
        logger.info("开始匹配...")
        matches = []

        # 获取高德实体的embedding矩阵（用于批量搜索）
        import numpy as np

        if not self.amap_embeddings:
            logger.warning("高德实体无embedding数据，无法进行语义匹配")
            return matches

        amap_ids = list(self.amap_embeddings.keys())
        amap_emb_matrix = np.array([self.amap_embeddings[eid] for eid in amap_ids])

        # 对每个现有实体进行匹配
        for source_eid, source_ent in self.existing_entities.items():
            source_emb = self.existing_embeddings.get(source_eid)
            source_lon = source_ent.get("lon")
            source_lat = source_ent.get("lat")

            best_match = None
            best_confidence = 0

            # 语义相似度搜索（批量计算）
            if source_emb:
                source_vec = np.array(source_emb)
                # 批量计算与所有高德实体的相似度
                similarities = np.dot(amap_emb_matrix, source_vec) / (
                    np.linalg.norm(amap_emb_matrix, axis=1) * np.linalg.norm(source_vec)
                )

                # 获取相似度高于阈值的候选
                for idx, sim in enumerate(similarities):
                    if sim < self.semantic_threshold:
                        continue

                    amap_eid = amap_ids[idx]
                    amap_ent = self.amap_entities[amap_eid]

                    # 计算地理距离
                    geo_dist = haversine_distance(
                        source_lon, source_lat,
                        amap_ent.get("lon"), amap_ent.get("lat")
                    )

                    # 综合置信度计算
                    if geo_dist <= self.geo_threshold:
                        # 语义+坐标匹配
                        confidence = (
                            AlignmentConfig.SEMANTIC_WEIGHT * sim +
                            AlignmentConfig.GEO_WEIGHT * (1 - geo_dist / self.geo_threshold)
                        )
                        match_type = "semantic_geo"
                    else:
                        # 仅语义匹配
                        confidence = sim
                        match_type = "semantic"

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = {
                            "source_id": source_eid,
                            "amap_id": amap_eid,
                            "semantic_sim": round(sim, 4),
                            "geo_dist": round(geo_dist, 2) if geo_dist != float('inf') else None,
                            "confidence": round(confidence, 4),
                            "match_type": match_type,
                        }

            # 坐标匹配（如果语义匹配未找到）
            if best_match is None and source_lon and source_lat:
                for amap_eid, amap_ent in self.amap_entities.items():
                    amap_lon = amap_ent.get("lon")
                    amap_lat = amap_ent.get("lat")

                    if amap_lon and amap_lat:
                        geo_dist = haversine_distance(
                            source_lon, source_lat, amap_lon, amap_lat
                        )

                        if geo_dist <= self.geo_threshold:
                            # 检查名称是否有一定相似度（模糊匹配）
                            source_name = source_ent["name"]
                            amap_name = amap_ent["name"]

                            # 简单名称相似度（包含关系）
                            name_sim = 0.5 if (source_name in amap_name or amap_name in source_name) else 0.3

                            confidence = name_sim * (1 - geo_dist / self.geo_threshold)
                            match_type = "geo_fuzzy"

                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_match = {
                                    "source_id": source_eid,
                                    "amap_id": amap_eid,
                                    "semantic_sim": None,
                                    "geo_dist": round(geo_dist, 2),
                                    "confidence": round(confidence, 4),
                                    "match_type": match_type,
                                }

            if best_match:
                matches.append(best_match)

        logger.info(f"找到匹配: {len(matches)} 对")
        return matches

    def create_alignment_table(self):
        """创建对齐结果表"""
        with self.pg_client.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS entity_alignment (
                    id SERIAL PRIMARY KEY,
                    source_id VARCHAR(100) NOT NULL,
                    amap_id VARCHAR(100) NOT NULL,
                    source_name VARCHAR(200),
                    amap_name VARCHAR(200),
                    semantic_sim FLOAT,
                    geo_dist FLOAT,
                    confidence FLOAT NOT NULL,
                    match_type VARCHAR(20) NOT NULL,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(source_id, amap_id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alignment_source ON entity_alignment(source_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alignment_amap ON entity_alignment(amap_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_alignment_confidence ON entity_alignment(confidence)")
            self.pg_client.conn.commit()

        logger.info("对齐表已创建")

    def save_matches(self, matches: List[Dict]):
        """保存匹配结果到数据库"""
        if not matches:
            logger.warning("无匹配结果，跳过保存")
            return

        # 准备数据
        data = []
        for m in matches:
            source_ent = self.existing_entities.get(m["source_id"], {})
            amap_ent = self.amap_entities.get(m["amap_id"], {})
            data.append((
                m["source_id"],
                m["amap_id"],
                source_ent.get("name", ""),
                amap_ent.get("name", ""),
                m["semantic_sim"],
                m["geo_dist"],
                m["confidence"],
                m["match_type"],
            ))

        # 批量插入
        with self.pg_client.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO entity_alignment
                    (source_id, amap_id, source_name, amap_name, semantic_sim, geo_dist, confidence, match_type)
                VALUES %s
                ON CONFLICT (source_id, amap_id) DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    semantic_sim = EXCLUDED.semantic_sim,
                    geo_dist = EXCLUDED.geo_dist,
                    match_type = EXCLUDED.match_type
                """,
                data,
                template="(%s, %s, %s, %s, %s, %s, %s, %s)"
            )
            self.pg_client.conn.commit()

        logger.success(f"保存匹配结果: {len(data)} 条")

    def generate_report(self, matches: List[Dict]) -> Dict:
        """生成统计报告"""
        if not matches:
            return {"total": 0}

        # 按匹配类型统计
        type_stats = {}
        for m in matches:
            mt = m["match_type"]
            type_stats[mt] = type_stats.get(mt, 0) + 1

        # 按置信度统计
        high_conf = len([m for m in matches if m["confidence"] >= 0.9])
        medium_conf = len([m for m in matches if 0.8 <= m["confidence"] < 0.9])
        low_conf = len([m for m in matches if m["confidence"] < 0.8])

        # 平均值
        avg_semantic = sum(m["semantic_sim"] or 0 for m in matches) / len(matches)
        avg_geo = sum(m["geo_dist"] or 0 for m in matches if m["geo_dist"]) / max(1, len([m for m in matches if m["geo_dist"]]))

        report = {
            "total": len(matches),
            "by_type": type_stats,
            "by_confidence": {
                "high (>=0.9)": high_conf,
                "medium (0.8-0.9)": medium_conf,
                "low (<0.8)": low_conf,
            },
            "avg_semantic_sim": round(avg_semantic, 4),
            "avg_geo_dist": round(avg_geo, 2),
        }

        return report

    def run(self) -> Dict:
        """执行完整的对齐流程"""
        logger.info("=" * 60)
        logger.info("实体对齐流程开始")
        logger.info("=" * 60)

        # 1. 加载实体
        self.load_entities()

        # 2. 计算缺失的embedding
        self.compute_missing_embeddings()

        # 3. 创建对齐表
        self.create_alignment_table()

        # 4. 执行匹配
        matches = self.find_matches()

        # 5. 保存结果
        self.save_matches(matches)

        # 6. 生成报告
        report = self.generate_report(matches)

        # 输出报告
        logger.success("=" * 60)
        logger.success("对齐完成！统计报告：")
        logger.success(f"  总匹配数: {report['total']}")
        if report['total'] > 0:
            logger.success(f"  按类型:")
            for mt, cnt in report['by_type'].items():
                logger.success(f"    {mt}: {cnt}")
            logger.success(f"  按置信度:")
            for level, cnt in report['by_confidence'].items():
                logger.success(f"    {level}: {cnt}")
            logger.success(f"  平均语义相似度: {report['avg_semantic_sim']}")
            logger.success(f"  平均地理距离: {report['avg_geo_dist']}m")

        return report


def main(
    semantic_threshold: float = AlignmentConfig.SEMANTIC_THRESHOLD_MEDIUM,
    geo_threshold: float = AlignmentConfig.GEO_THRESHOLD_MEDIUM
):
    """主函数"""
    # 连接数据库
    pg_config = settings.get_postgres_config()
    pg_client = PostgresClient(**pg_config)

    try:
        aligner = EntityAligner(
            pg_client=pg_client,
            semantic_threshold=semantic_threshold,
            geo_threshold=geo_threshold
        )
        report = aligner.run()
        return report
    finally:
        pg_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实体对齐脚本")
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=AlignmentConfig.SEMANTIC_THRESHOLD_MEDIUM,
        help="语义相似度阈值 (默认0.85)"
    )
    parser.add_argument(
        "--geo-threshold",
        type=float,
        default=AlignmentConfig.GEO_THRESHOLD_MEDIUM,
        help="地理距离阈值(米) (默认100)"
    )

    args = parser.parse_args()
    main(
        semantic_threshold=args.semantic_threshold,
        geo_threshold=args.geo_threshold
    )