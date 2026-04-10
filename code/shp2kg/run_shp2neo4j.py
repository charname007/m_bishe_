#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHP 转 Neo4j 知识图谱脚本

将 shpfiles 文件夹中的 SHP 文件转换为知识图谱并保存到 Neo4j

使用方法:
    cd e:/study/毕设/code
    python shp2kg/run_shp2neo4j.py
"""

import os
import sys
import time

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from shp2kg.shp2kg import (
    GeoEntityLoader,
    SpatialRelationCalculator,
    GeoKnowledgeGraph,
    logger
)

# ============================================================
# 配置
# ============================================================

SHP_DIR = os.path.join(os.path.dirname(__file__), 'shpfiles')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# SHP 文件配置（根据实际字段调整）
SHP_CONFIG = [
    {
        'path': os.path.join(SHP_DIR, 'pois.shp'),
        'type': 'poi',
        'name_field': 'name',
        'type_field': 'amenity'  # POI 类型字段
    },
    {
        'path': os.path.join(SHP_DIR, 'blocks.shp'),
        'type': 'block',
        'name_field': 'name',
        'type_field': 'landuse'
    },
    {
        'path': os.path.join(SHP_DIR, 'roads.shp'),
        'type': 'road',
        'name_field': 'name',
        'type_field': 'highway'  # 道路等级字段
    },
    {
        'path': os.path.join(SHP_DIR, 'buildings.shp'),
        'type': 'building',
        'name_field': 'name',
        'type_field': 'building'
    }
]

# 空间关系计算参数
DISTANCE_THRESHOLD = 300    # 近邻距离阈值（米）- 降低以减少关系数量
SKELETON_INTERVAL = 100     # 骨架点采样间隔（米）- 增大以减少点数
BOUNDARY_INTERVAL = 100     # 边界采样间隔（米）- 增大以减少点数


def check_shp_files():
    """检查 SHP 文件是否存在"""
    logger.info("=" * 60)
    logger.info("Step 1: 检查 SHP 文件")
    logger.info("=" * 60)

    available = []
    total_records = 0

    for config in SHP_CONFIG:
        path = config['path']
        if os.path.exists(path):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(path)
                available.append(config)
                total_records += len(gdf)
                logger.info(f"  [OK] {os.path.basename(path)}: {len(gdf)} 条记录, {gdf.geometry.geom_type.unique()}")
            except Exception as e:
                logger.error(f"  [ERROR] {os.path.basename(path)}: 读取失败 - {e}")
        else:
            logger.warning(f"  [SKIP] {os.path.basename(path)}: 文件不存在")

    if not available:
        logger.error("没有找到任何可用的 SHP 文件！")
        return None, 0

    logger.info(f"\n找到 {len(available)} 个 SHP 文件，共 {total_records} 条记录")
    return available, total_records


def load_shp_files(loader, shp_configs):
    """加载 SHP 文件"""
    logger.info("=" * 60)
    logger.info("Step 2: 加载 SHP 文件")
    logger.info("=" * 60)

    start_time = time.time()

    for config in shp_configs:
        try:
            logger.info(f"  加载 {config['type']}...")
            gdf = loader.load_shp(
                config['path'],
                config['type'],
                name_field=config['name_field']
            )
            logger.info(f"    -> {len(gdf)} 条记录")
        except Exception as e:
            logger.error(f"  加载 {config['type']} 失败: {e}")

    elapsed = time.time() - start_time
    logger.info(f"\n总计加载实体: {len(loader.entities)} 个 (耗时 {elapsed:.1f}s)")
    return loader.entities


def build_knowledge_graph(entities):
    """构建知识图谱"""
    logger.info("=" * 60)
    logger.info("Step 3: 计算空间关系")
    logger.info("=" * 60)

    start_time = time.time()

    # 创建空间关系计算器
    logger.info(f"  参数: 近邻阈值={DISTANCE_THRESHOLD}m, 骨架间隔={SKELETON_INTERVAL}m, 边界间隔={BOUNDARY_INTERVAL}m")

    calc = SpatialRelationCalculator(
        entities,
        distance_threshold=DISTANCE_THRESHOLD,
        skeleton_interval=SKELETON_INTERVAL,
        boundary_interval=BOUNDARY_INTERVAL
    )

    # 计算所有关系
    relations = calc.compute_all()

    elapsed = time.time() - start_time
    logger.info(f"\n空间关系计算完成，耗时 {elapsed:.1f}s")

    logger.info("=" * 60)
    logger.info("Step 4: 构建知识图谱")
    logger.info("=" * 60)

    # 构建知识图谱
    kg = GeoKnowledgeGraph(entities, relations)

    return kg, calc


def export_to_neo4j(kg):
    """导出到 Neo4j"""
    logger.info("=" * 60)
    logger.info("Step 5: 导出到 Neo4j")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # 使用更大的批次大小提高效率
        kg.export_to_neo4j(batch_size=5000)
        elapsed = time.time() - start_time
        logger.info(f"Neo4j 导出成功！耗时 {elapsed:.1f}s")
        return True
    except Exception as e:
        logger.error(f"Neo4j 导出失败: {e}")
        logger.info("\n请检查:")
        logger.info("  1. Neo4j 服务是否正在运行")
        logger.info("  2. config.py 中的连接配置是否正确")
        logger.info("  3. 数据库用户名密码是否正确")
        return False


def export_to_files(kg):
    """导出到文件（JSON-LD 和 RDF）"""
    logger.info("=" * 60)
    logger.info("Step 6: 导出到文件")
    logger.info("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 导出 JSON-LD
    json_path = os.path.join(OUTPUT_DIR, 'geo_kg.json')
    try:
        start_time = time.time()
        kg.export_to_jsonld(json_path)
        elapsed = time.time() - start_time
        logger.info(f"  JSON-LD: {json_path} (耗时 {elapsed:.1f}s)")
    except Exception as e:
        logger.error(f"  JSON-LD 导出失败: {e}")

    # 导出 RDF
    rdf_path = os.path.join(OUTPUT_DIR, 'geo_kg.ttl')
    try:
        start_time = time.time()
        kg.export_to_rdf(rdf_path)
        elapsed = time.time() - start_time
        logger.info(f"  RDF: {rdf_path} (耗时 {elapsed:.1f}s)")
    except Exception as e:
        logger.error(f"  RDF 导出失败: {e}")


def print_summary(entities, kg, neo4j_success):
    """打印总结"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("处理完成！")
    logger.info("=" * 60)

    # 统计实体类型
    from collections import Counter
    type_count = Counter(ent['type'] for ent in entities.values())

    logger.info(f"实体统计:")
    for etype, count in type_count.most_common():
        logger.info(f"  {etype}: {count}")

    logger.info(f"\n总计: {len(entities)} 实体, {len(kg.relations)} 关系")
    logger.info(f"Neo4j 导出: {'成功' if neo4j_success else '失败'}")
    logger.info(f"文件输出: {OUTPUT_DIR}")


def main():
    """主函数"""
    total_start = time.time()

    logger.info("=" * 60)
    logger.info("SHP 转 Neo4j 知识图谱")
    logger.info("=" * 60)
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"SHP 目录: {SHP_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # 1. 检查文件
    shp_configs, total_records = check_shp_files()
    if not shp_configs:
        return

    # 2. 加载数据
    loader = GeoEntityLoader(crs_target="EPSG:4326")
    entities = load_shp_files(loader, shp_configs)

    if not entities:
        logger.error("没有加载到任何实体，退出")
        return

    # 3. 构建知识图谱
    kg, calc = build_knowledge_graph(entities)

    # 4. 导出到 Neo4j
    neo4j_success = export_to_neo4j(kg)

    # 5. 导出到文件（作为备份）
    export_to_files(kg)

    # 6. 总结
    print_summary(entities, kg, neo4j_success)

    total_elapsed = time.time() - total_start
    logger.info(f"\n总耗时: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()