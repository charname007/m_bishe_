"""
地理知识图谱构建器
输入：道路(线)、POI(点)、建筑物(面)、街区(面) 的 SHP 文件
输出：Neo4j 知识图谱 + NetworkX 可视化
"""

import geopandas as gpd
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import nearest_points, transform
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json
import warnings
import sys
import os
from loguru import logger
from pyproj import Transformer
from rtree import index

try:
    from config import settings
except ImportError:
    # 如果没有 config 模块，使用默认配置
    class MockSettings:
        DEBUG = False
        NEO4J_URI = 'bolt://localhost:7687'
        NEO4J_USER = 'neo4j'
        NEO4J_PASSWORD = 'password'
    settings = MockSettings()

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Loguru 日志配置
# ============================================================

# 移除默认 handler
logger.remove()

# 控制台输出（根据 DEBUG 设置日志级别）
logger.add(
    sys.stderr,
    level="DEBUG" if settings.DEBUG else "INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{module}:{function}:{line}</cyan> | <level>{message}</level>",
    colorize=True
)

# 文件日志（自动创建 logs 目录）
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logger.add(
    os.path.join(log_dir, "shp2kg_{time:YYYY-MM-DD}.log"),
    rotation="00:00",  # 每天轮转
    retention="7 days",  # 保留7天
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {module}:{function}:{line} | {message}"
)

# 错误日志专用（保留30天，带完整堆栈）
logger.add(
    os.path.join(log_dir, "error_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="30 days",
    level="ERROR",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {module}:{function}:{line}\n{message}\n{exception}",
    backtrace=True,   # 显示完整调用栈
    diagnose=True     # 显示变量值（仅DEBUG模式建议开启）
)

logger.info("日志系统初始化完成")


# ============================================================
# 第一部分：数据加载与预处理
# ============================================================

class GeoEntityLoader:
    """地理实体加载器"""

    def __init__(self, crs_target="EPSG:4326"):
        self.crs_target = crs_target
        self.entities = {}  # {entity_id: {type, name, geometry, attributes...}}
        self.entity_counter = 0

    def load_shp(self, shp_path, entity_type, name_field='name'):
        """
        加载SHP文件并提取实体
        :param shp_path:    SHP文件路径
        :param entity_type: 实体类型 (road/poi/building/block)
        :param name_field:  名称字段名
        """
        gdf = gpd.read_file(shp_path)

        # 统一坐标系（建议先转投影坐标系计算距离，再转回）
        if gdf.crs and gdf.crs != self.crs_target:
            gdf = gdf.to_crs(self.crs_target)

        logger.info(f"[加载] {entity_type}: {len(gdf)} 个实体, 几何类型: {gdf.geometry.geom_type.unique()}")

        for idx, row in gdf.iterrows():
            entity_id = f"{entity_type}_{self.entity_counter}"
            self.entity_counter += 1

            # 提取名称
            name = "未命名"
            if name_field and name_field in row.index:
                name = str(row[name_field]) if row[name_field] else "未命名"

            # 提取质心（用于三角网构建）
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            centroid = geom.centroid

            # 提取所有非geometry属性
            attributes = {}
            for col in gdf.columns:
                if col != 'geometry':
                    val = row[col]
                    if val is not None and str(val) != 'nan':
                        attributes[col] = str(val)

            self.entities[entity_id] = {
                'id': entity_id,
                'type': entity_type,
                'name': name,
                'geometry': geom,
                'centroid': centroid,
                'centroid_coords': (centroid.x, centroid.y),
                'attributes': attributes
            }

        return gdf

    def load_all(self, shp_list, name_list=None, type_list=None):
        """
        批量加载SHP文件
        :param shp_list: SHP文件路径列表
        :param name_list: 名称字段列表，与shp_list一一对应，默认全部为'name'
        :param type_list: 实体类型列表，与shp_list一一对应，默认从文件名推断
        :return: 实体字典
        """
        import os

        if name_list is None:
            name_list = ['name'] * len(shp_list)
        if type_list is None:
            # 从文件名推断实体类型
            type_list = [os.path.splitext(os.path.basename(shp))[0].lower()
                         for shp in shp_list]

        for shp, name, etype in zip(shp_list, name_list, type_list):
            self.load_shp(shp, etype, name)

        logger.info(f"[总计] 加载实体: {len(self.entities)} 个")
        return self.entities
    
    def all_entities_names(self):
        """返回所有实体的名称列表（改进版）
        用于之后的实体匹配
        （需要到时候导出为一个文件）
        """
        # 定义无效名称集合
        invalid_names = {"未命名", "null", "None"}

        names = []
        for ent in self.entities.values():
            # 使用 .get() 避免 KeyError
            name = ent.get('name')

            # 检查是否存在、是否为字符串、是否不在无效列表中
            if name and isinstance(name, str) and name not in invalid_names:
                names.append(name)

        return names

# ============================================================
# 第二部分：空间关系计算（核心！）
# ============================================================

class SpatialRelationCalculator:
    """空间关系计算器（修正版：统一投影坐标系 + 优化拓扑逻辑）
    
    其中为了解决三角网在面和线要素上的不足，采用了几何方法来判断intersection和touches关系，保证了更准确的拓扑关系计算。"""

    def __init__(self, entities, distance_threshold=500,
                    src_crs="EPSG:4326", auto_proj_crs=True,
                    poi_type_field='amenity', road_level_field='highway',
                    skeleton_interval=50, boundary_interval=50, large_area_threshold=10000):
        """
        空间关系计算器
        
        Args:
            skeleton_interval: 线状实体骨架点采样间隔（米）
            boundary_interval: 面状实体边界采样间隔（米）
            large_area_threshold: 大型面状实体阈值（平方米），超过此阈值启用边界采样
        """
        self.entities = entities
        self.distance_threshold = distance_threshold
        self.relations = []
        self.poi_type_field = poi_type_field
        self.road_level_field = road_level_field
        
        # ✅ 新增参数
        self.skeleton_interval = skeleton_interval
        self.boundary_interval = boundary_interval
        self.large_area_threshold = large_area_threshold

        # ✅ 1. 自动计算投影坐标系 (UTM)
        if auto_proj_crs:
            center_lon = np.mean([e['centroid_coords'][0]
                                    for e in entities.values()])
            zone = int((center_lon + 180) / 6) + 1
            is_southern = np.mean([e['centroid_coords'][1]
                                    for e in entities.values()]) < 0
            # UTM Zone计算: 北半球32601-32660, 南半球32701-32760
            self.dst_crs = f"EPSG:{32700 + zone if is_southern else 32600 + zone}"
            logger.info(f"[坐标系] 自动选择: {self.dst_crs} (单位: 米)")
        else:
            self.dst_crs = src_crs # 如果不自动投影，需确保输入数据已是投影坐标系

        self.transformer = Transformer.from_crs(
            src_crs, self.dst_crs, always_xy=True)

        # ✅ 2. 预计算投影坐标与投影几何
        self.proj_coords = {}
        self.proj_geometries = {}
        
        # ✅ 新增：三角网增强数据结构
        self.triangulation_points = []  # 所有参与三角网的点坐标
        self.point_to_entity = {}       # 点索引 -> 实体ID
        self.point_type = {}            # 点索引 -> 点类型(centroid/skeleton/boundary)
        
        self._prepare_projected_data()

    def _prepare_projected_data(self):
        """
        预计算投影坐标及几何体的投影（增强版）
        
        改进：
        - 线状实体：骨架点采样（表达空间延伸性）
        - 大型面状实体：边界采样（表达边界邻近性）
        """
        logger.info("  [预处理] 正在投影几何体并采样...")
        
        point_idx = 0
        
        for eid, ent in self.entities.items():
            geom = ent['geometry']
            
            if geom is None or geom.is_empty:
                continue
            
            # 投影几何体
            proj_geom = transform(self.transformer.transform, geom)
            self.proj_geometries[eid] = proj_geom
            
            # 根据几何类型添加三角网参与点
            geom_type = geom.geom_type
            
            if geom_type == 'Point':
                # 点状实体：仅用质心
                centroid = proj_geom
                coords = (centroid.x, centroid.y)
                self.proj_coords[eid] = np.array(coords)
                self.triangulation_points.append(coords)
                self.point_to_entity[point_idx] = eid
                self.point_type[point_idx] = 'centroid'
                point_idx += 1
            
            elif geom_type in ['LineString', 'MultiLineString']:
                # 线状实体：质心 + 骨架点
                centroid = proj_geom.centroid
                coords = (centroid.x, centroid.y)
                self.proj_coords[eid] = np.array(coords)
                
                # 添加质心
                self.triangulation_points.append(coords)
                self.point_to_entity[point_idx] = eid
                self.point_type[point_idx] = 'centroid'
                point_idx += 1
                
                # 添加骨架点
                skeleton_pts = self._sample_line_points(proj_geom)
                for pt in skeleton_pts:
                    self.triangulation_points.append(pt)
                    self.point_to_entity[point_idx] = eid
                    self.point_type[point_idx] = 'skeleton'
                    point_idx += 1
            
            elif geom_type in ['Polygon', 'MultiPolygon']:
                # 面状实体
                centroid = proj_geom.centroid
                coords = (centroid.x, centroid.y)
                self.proj_coords[eid] = np.array(coords)
                
                # 添加质心
                self.triangulation_points.append(coords)
                self.point_to_entity[point_idx] = eid
                self.point_type[point_idx] = 'centroid'
                point_idx += 1
                
                # 大型面状实体：添加边界采样点
                area = proj_geom.area
                if area > self.large_area_threshold:
                    boundary_pts = self._sample_boundary_points(proj_geom)
                    for pt in boundary_pts:
                        self.triangulation_points.append(pt)
                        self.point_to_entity[point_idx] = eid
                        self.point_type[point_idx] = 'boundary'
                        point_idx += 1
        
        # 统计信息
        skeleton_count = sum(1 for t in self.point_type.values() if t == 'skeleton')
        boundary_count = sum(1 for t in self.point_type.values() if t == 'boundary')
        centroid_count = sum(1 for t in self.point_type.values() if t == 'centroid')
        
        logger.info(f"  [预处理] 投影完成。三角网参与点: {len(self.triangulation_points)} 个")
        logger.info(f"    - 质心点: {centroid_count}")
        logger.info(f"    - 骨架点: {skeleton_count}")
        logger.info(f"    - 边界点: {boundary_count}")
    
    def _sample_line_points(self, proj_geom):
        """
        线状实体骨架点采样
        
        Args:
            proj_geom: 投影后的几何体
        
        Returns:
            骨架点坐标列表 [(x, y), ...]
        """
        points = []
        
        if proj_geom.geom_type == 'MultiLineString':
            # 多线段：对每条线采样
            for line in proj_geom.geoms:
                points.extend(self._sample_single_line(line))
        else:
            points.extend(self._sample_single_line(proj_geom))
        
        return points
    
    def _sample_single_line(self, line_geom):
        """
        单线段骨架点采样

        修复：处理各种退化线段情况
        - 空线段（无坐标点）
        - 单点线段（只有一个坐标点）
        - 零长度线段（多个坐标点但长度为0，如所有点相同）
        """
        coords = list(line_geom.coords)

        # 边界检查1：空线段
        if len(coords) == 0:
            logger.warning("发现空线段，跳过采样")
            return []

        # 边界检查2：单点线段
        if len(coords) < 2:
            return [(coords[0][0], coords[0][1])]

        total_length = line_geom.length

        # 边界检查3：零长度线段（起点终点相同）
        if total_length == 0 or coords[0] == coords[-1]:
            # 检查是否所有点都相同
            first_pt = coords[0]
            all_same = all(pt == first_pt for pt in coords)
            if all_same:
                return [(first_pt[0], first_pt[1])]
            # 否则尝试返回第一个不同的点
            for pt in coords:
                if pt != first_pt:
                    return [(first_pt[0], first_pt[1]), (pt[0], pt[1])]
            return [(first_pt[0], first_pt[1])]

        # 短线段：仅返回起点和终点（确保不相同）
        if total_length < self.skeleton_interval:
            start_pt = coords[0]
            end_pt = coords[-1]
            if start_pt == end_pt:
                return [(start_pt[0], start_pt[1])]
            return [(start_pt[0], start_pt[1]), (end_pt[0], end_pt[1])]

        n_points = int(total_length / self.skeleton_interval)
        points = []

        for i in range(n_points + 1):
            dist = i * self.skeleton_interval
            if dist <= total_length:
                pt = line_geom.interpolate(dist)
                points.append((pt.x, pt.y))

        # 确保包含终点（避免重复）
        end_pt = line_geom.interpolate(total_length)
        end_coords = (end_pt.x, end_pt.y)
        if points and points[-1] != end_coords:
            points.append(end_coords)

        return points
    
    def _sample_boundary_points(self, proj_geom):
        """
        面状实体边界采样
        
        改进：MultiPolygon采样所有面的边界（不再只采样最大面）
        
        Args:
            proj_geom: 投影后的几何体
        
        Returns:
            边界采样点坐标列表 [(x, y), ...]
        """
        points = []
        
        if proj_geom.geom_type == 'MultiPolygon':
            # 多面：采样所有面的外边界
            for polygon in proj_geom.geoms:
                exterior = polygon.exterior
                total_length = exterior.length
                n_points = int(total_length / self.boundary_interval)
                
                for i in range(n_points + 1):
                    dist = i * self.boundary_interval
                    if dist <= total_length:
                        pt = exterior.interpolate(dist)
                        points.append((pt.x, pt.y))
        else:
            exterior = proj_geom.exterior
            total_length = exterior.length
            n_points = int(total_length / self.boundary_interval)
            
            for i in range(n_points + 1):
                dist = i * self.boundary_interval
                if dist <= total_length:
                    pt = exterior.interpolate(dist)
                    points.append((pt.x, pt.y))
        
        return points

    # def calc_topological_relations(self):
    #     """✅ 修正版：使用投影几何 + 优化判断逻辑"""
    #     print("\n[拓扑关系] 计算中...")

    #     idx = index.Index()
    #     entity_list = list(self.entities.values())

    #     # 容差设为1米（因为现在是投影坐标系，单位是米）
    #     tolerance_m = 2.0

    #     # 构建R树索引
    #     for i, ent in enumerate(entity_list):
    #         eid = ent['id']
    #         # 使用投影后的几何体计算边界
    #         bounds = self.proj_geometries[eid].buffer(tolerance_m).bounds
    #         idx.insert(i, bounds)

    #     relation_count = 0
    #     processed_pairs = set()

    #     for i, ent_a in enumerate(entity_list):
    #         eid_a = ent_a['id']
    #         geom_a = self.proj_geometries[eid_a]

    #         # R树查询候选集
    #         candidates = list(idx.intersection(geom_a.buffer(tolerance_m).bounds))

    #         for j in candidates:
    #             if j <= i: continue # 避免重复计算

    #             ent_b = entity_list[j]
    #             eid_b = ent_b['id']
    #             geom_b = self.proj_geometries[eid_b]

    #             # 唯一标识符防止重复
    #             pair_key = (min(eid_a, eid_b), max(eid_a, eid_b))
    #             if pair_key in processed_pairs: continue
    #             processed_pairs.add(pair_key)

    #             try:
    #                 # ✅ 逻辑优化：优先判断特殊情况，再判断一般情况
    #                 # 1. 接触 - 必须最先判断，否则会被intersects吞掉
    #                 if geom_a.touches(geom_b):
    #                     self.relations.append((
    #                         eid_a, 'adjacent_to', eid_b,
    #                         {'relation_type': 'topological'}
    #                     ))
    #                     relation_count += 1

    #                 # 2. 包含 - 互斥关系
    #                 elif geom_a.contains(geom_b):
    #                     self.relations.append((
    #                         eid_a, 'contains', eid_b,
    #                         {'relation_type': 'topological'}
    #                     ))
    #                     self.relations.append((
    #                         eid_b, 'within', eid_a,
    #                         {'relation_type': 'topological'}
    #                     ))
    #                     relation_count += 2

    #                 elif geom_b.contains(geom_a):
    #                     self.relations.append((
    #                         eid_b, 'contains', eid_a,
    #                         {'relation_type': 'topological'}
    #                     ))
    #                     self.relations.append((
    #                         eid_a, 'within', eid_b,
    #                         {'relation_type': 'topological'}
    #                     ))
    #                     relation_count += 2

    #                 # 3. 相交 - 剩余情况
    #                 elif geom_a.intersects(geom_b):
    #                     rel_type = 'topological'
    #                     rel_name = 'intersects'

    #                     # 语义细化：道路穿过街区
    #                     if ent_a['type'] == 'road' and ent_b['type'] == 'block':
    #                         rel_name = 'passes_through'
    #                     elif ent_b['type'] == 'road' and ent_a['type'] == 'block':
    #                         # 交换主语，保持 'road passes_through block' 的语义
    #                         eid_a, eid_b = eid_b, eid_a
    #                         rel_name = 'passes_through'

    #                     self.relations.append((
    #                         eid_a, rel_name, eid_b,
    #                         {'relation_type': rel_type}
    #                     ))
    #                     relation_count += 1

    #             except Exception as e:
    #                 print(f"  ! 拓扑计算错误 {eid_a}-{eid_b}: {e}")
    #                 continue

    #     print(f"  → 拓扑关系: {relation_count} 条")

    def calc_topological_relations(self):
        """
        ✅ 改进版：解决缝隙漏判与长线性能问题
        - 使用 buffer 代替 touches，容错数据缝隙
        - 保持 R树 索引优化
        """
        logger.info("\n[拓扑关系] 计算中（容错模式）...")

        idx = index.Index()
        entity_list = list(self.entities.values())

        # 1. 构建R树索引
        # 注意：这里我们直接存储几何体的bounds，不预先buffer，以保持索引最小化
        for i, ent in enumerate(entity_list):
            eid = ent['id']
            geom = self.proj_geometries[eid]
            idx.insert(i, geom.bounds)

        relation_count = 0
        processed_pairs = set()

        # 2. 定义容差半径（米）
        # 用于处理数据缝隙，将“接近”视为“邻接”
        adjacency_tolerance = 2.0

        for i, ent_a in enumerate(entity_list):
            eid_a = ent_a['id']
            geom_a = self.proj_geometries[eid_a]

            # ✅ 优化：查询时对bounds进行buffer，扩大搜索范围防止漏判
            # 但索引内部仍然使用原始bounds，保证索引效率
            search_bounds = geom_a.buffer(adjacency_tolerance).bounds
            
            #candidates是经过索引后实体周围的候选集，后续再通过几何方法精确判断关系
            #，因此不需要遍历全部，所以时间复杂度不是n^2，而是n*log(n)（索引查询）+ k（候选集大小，远小于n）。
            
            candidates = list(idx.intersection(search_bounds))

            for j in candidates:
                if j <= i: continue

                ent_b = entity_list[j]
                eid_b = ent_b['id']
                geom_b = self.proj_geometries[eid_b]

                pair_key = (min(eid_a, eid_b), max(eid_a, eid_b))
                if pair_key in processed_pairs: continue
                processed_pairs.add(pair_key)

                try:
                    # ✅ 核心逻辑修正：基于距离判定拓扑关系

                    # 计算两几何体的最小距离
                    dist = geom_a.distance(geom_b)

                    # 1. 包含关系 (不受容差影响，严格判定)
                    # 注意：contains判定本身很严格，如果因为缝隙导致包含关系识别不出，
                    # 通常需要数据层面的清洗，这里暂保持严格逻辑
                    if geom_a.contains(geom_b):
                        self._add_relation(eid_a, 'contains', eid_b)
                        self._add_relation(eid_b, 'within', eid_a)
                        relation_count += 2

                    elif geom_b.contains(geom_a):
                        self._add_relation(eid_b, 'contains', eid_a)
                        self._add_relation(eid_a, 'within', eid_b)
                        relation_count += 2

                    # 2. 邻接/相交关系 (引入容差)
                    # 如果距离小于容差，认为它们是物理上相邻的
                    elif dist <= adjacency_tolerance:
                        # 进一步区分是“接触”还是“交叉”
                        # 使用 buffer 进行实际几何运算
                        # 如果 A缓冲后与 B 相交，但 A 与 B 不相交 -> 判定为 Adjacent
                        if geom_a.buffer(adjacency_tolerance).intersects(geom_b):
                            # 如果原始几何体本身就相交，那肯定是 Intersects
                            if geom_a.intersects(geom_b):
                                rel_name = 'intersects'
                                # 道路穿过街区逻辑保留
                                if (ent_a['type'] == 'road' and ent_b['type'] == 'block') or \
                                    (ent_b['type'] == 'road' and ent_a['type'] == 'block'):
                                        rel_name = 'passes_through'

                                self._add_relation(eid_a, rel_name, eid_b)
                                relation_count += 1
                            else:
                                # 原始几何不相交，但缓冲后相交 -> 视为 Adjacent (解决了缝隙问题)
                                self._add_relation(eid_a, 'adjacent_to', eid_b)
                                relation_count += 1

                except Exception as e:
                    logger.error(f"  ! 拓扑计算错误 {eid_a}-{eid_b}: {e}")
                    continue

        logger.info(f"  → 拓扑关系: {relation_count} 条")

    def _add_relation(self, head, rel, tail, props=None):
        """辅助方法：添加关系"""
        if props is None:
            props = {'relation_type': 'topological'}
        self.relations.append((head, rel, tail, props))
        
        
    def calc_delaunay_neighbors(self):
        """
        ✅ 增强版：基于骨架点和边界采样点构建三角网

        改进：
        - 线状实体骨架点参与三角网，捕获沿路邻近的POI
        - 大型面状实体边界点参与三角网，捕获边界邻近关系
        - 骨架点连接数量作为邻近强度指标
        - 去除重复点避免Delaunay计算错误
        """
        logger.info("[三角网近邻] 构建Delaunay三角网（增强版）...")

        if len(self.triangulation_points) < 3:
            logger.warning("  → 参与点数量不足3个，无法构建三角网")
            return

        # ✅ 去除重复点（考虑浮点精度）
        unique_points = []
        seen_coords = set()
        old_to_new_idx = {}  # 旧索引 -> 新索引映射
        new_point_to_entity = {}
        new_point_type = {}

        for old_idx, pt in enumerate(self.triangulation_points):
            # 使用6位小数精度避免浮点误差
            pt_key = (round(pt[0], 6), round(pt[1], 6))
            if pt_key not in seen_coords:
                seen_coords.add(pt_key)
                new_idx = len(unique_points)
                unique_points.append(pt)
                old_to_new_idx[old_idx] = new_idx

                # 迁移属性映射
                eid = self.point_to_entity.get(old_idx)
                pt_type = self.point_type.get(old_idx, 'centroid')
                if eid:
                    new_point_to_entity[new_idx] = eid
                    new_point_type[new_idx] = pt_type

        # 更新映射
        self._unique_point_to_entity = new_point_to_entity
        self._unique_point_type = new_point_type

        if len(unique_points) < 3:
            logger.warning(f"  → 去重后点数量不足3个（{len(unique_points)}个），无法构建三角网")
            return

        logger.info(f"  → 原始点: {len(self.triangulation_points)}，去重后: {len(unique_points)}")

        # 使用去重后的点构建三角网
        points_arr = np.array(unique_points)
        tri = Delaunay(points_arr)

        logger.info(f"  → 三角形数量: {len(tri.simplices)}")
        logger.info(f"  → 参与点数量: {len(unique_points)}")

        # 统计实体间的骨架点连接强度
        entity_connection_strength = {}
        entity_min_distance = {}  # 实体间的最小距离

        for simplex in tri.simplices:
            for k in range(3):
                i = simplex[k]
                j = simplex[(k + 1) % 3]

                eid_a = new_point_to_entity.get(i)
                eid_b = new_point_to_entity.get(j)

                if eid_a is None or eid_b is None:
                    continue

                # 不同实体的点连接 → 实体邻近
                if eid_a != eid_b:
                    pair_key = (min(eid_a, eid_b), max(eid_a, eid_b))

                    # 计算点间距离
                    dist = np.linalg.norm(points_arr[i] - points_arr[j])

                    # 更新连接强度
                    entity_connection_strength[pair_key] = \
                        entity_connection_strength.get(pair_key, 0) + 1

                    # 更新最小距离
                    if pair_key not in entity_min_distance or \
                       dist < entity_min_distance[pair_key]:
                        entity_min_distance[pair_key] = dist

        # 生成邻近关系
        relation_count = 0
        for (eid_a, eid_b), strength in entity_connection_strength.items():
            min_dist = entity_min_distance.get((eid_a, eid_b), 0)

            if min_dist <= self.distance_threshold:
                # 确定邻近方法（根据点类型）
                pt_types_a = set()
                pt_types_b = set()
                for idx, eid in new_point_to_entity.items():
                    if eid == eid_a:
                        pt_types_a.add(new_point_type.get(idx, 'centroid'))
                    elif eid == eid_b:
                        pt_types_b.add(new_point_type.get(idx, 'centroid'))

                method = 'delaunay_enhanced'
                if 'skeleton' in pt_types_a or 'skeleton' in pt_types_b:
                    method = 'delaunay_skeleton'
                elif 'boundary' in pt_types_a or 'boundary' in pt_types_b:
                    method = 'delaunay_boundary'

                self.relations.append((
                    eid_a, 'near', eid_b,
                    {
                        'relation_type': 'proximity',
                        'distance_m': round(min_dist, 2),
                        'strength': strength,  # 骨架点连接数量
                        'method': method
                    }
                ))
                relation_count += 1

        logger.info(f"  → 近邻关系（≤{self.distance_threshold}m）: {relation_count} 条")

        # 保存三角网用于可视化
        self._delaunay_tri = tri
        self._delaunay_points = points_arr

    def calc_direction_relations(self):
        """
        增强版：基于拓扑邻接 + 三角网邻近 计算方向

        改进：
        - 线状实体使用最近点角度而非质心角度
        - 点状+点状、点状+面状仍使用质心角度
        - 添加投影坐标存在性检查，避免KeyError
        """
        logger.info("[方向关系] 计算中...")

        # 收集所有空间上接近的关系
        # 1. 来自三角网的近邻
        near_pairs = set([(r[0], r[2]) for r in self.relations if r[1] == 'near'])

        # 2. 来自拓扑计算的邻接
        adjacent_pairs = set([(r[0], r[2]) for r in self.relations if r[1] == 'adjacent_to'])

        # 3. 来自拓扑计算的相交/穿过
        intersect_pairs = set([(r[0], r[2]) for r in self.relations if r[1] in ['intersects', 'passes_through']])

        # 合并所有需要计算方向的对
        all_spatial_pairs = near_pairs.union(adjacent_pairs).union(intersect_pairs)

        direction_count = 0
        skipped_count = 0
        for eid_a, eid_b in all_spatial_pairs:
            ent_a = self.entities.get(eid_a)
            ent_b = self.entities.get(eid_b)

            if not ent_a or not ent_b:
                skipped_count += 1
                continue

            geom_a = self.proj_geometries.get(eid_a)
            geom_b = self.proj_geometries.get(eid_b)

            if not geom_a or not geom_b:
                skipped_count += 1
                continue

            # 根据几何类型选择角度计算策略
            geom_type_a = geom_a.geom_type
            geom_type_b = geom_b.geom_type

            # 线状实体 + 点状实体 → 最近点角度
            if geom_type_a in ['LineString', 'MultiLineString'] and geom_type_b == 'Point':
                result = self._calc_direction_line_to_point(geom_a, geom_b)
                angle_a = result['angle']
                direction_a = result['direction']
                method = 'nearest_point'
            elif geom_type_b in ['LineString', 'MultiLineString'] and geom_type_a == 'Point':
                # 反向情况
                result = self._calc_direction_line_to_point(geom_b, geom_a)
                angle_a = (result['angle'] + 180) % 360  # 反向角度
                direction_a = self._get_opposite_direction(result['direction'])
                method = 'nearest_point'
            else:
                # 其他情况 → 质心角度
                # ✅ 添加投影坐标存在性检查
                if eid_a not in self.proj_coords or eid_b not in self.proj_coords:
                    skipped_count += 1
                    logger.debug(f"跳过方向计算: {eid_a} 或 {eid_b} 无投影坐标")
                    continue

                p_a = self.proj_coords[eid_a]
                p_b = self.proj_coords[eid_b]

                dx = p_b[0] - p_a[0]
                dy = p_b[1] - p_a[1]

                angle_a = np.degrees(np.arctan2(dx, dy)) % 360
                direction_a = self._angle_to_direction(angle_a)
                method = 'centroid'

            # 添加正向关系
            self.relations.append((
                eid_a, f'{direction_a}_of', eid_b,
                {'relation_type': 'directional', 'angle': round(angle_a, 1), 'method': method}
            ))
            direction_count += 1

            # 添加反向关系
            opposite_direction = self._get_opposite_direction(direction_a)
            opposite_angle = (angle_a + 180) % 360
            self.relations.append((
                eid_b, f'{opposite_direction}_of', eid_a,
                {'relation_type': 'directional', 'angle': round(opposite_angle, 1), 'method': method}
            ))
            direction_count += 1

        logger.info(f"  → 方向关系: {direction_count} 条，跳过: {skipped_count} 对")
    
    def _calc_direction_line_to_point(self, line_geom, point_geom):
        """
        计算点相对于线状实体的方向（基于最近点）
        
        Args:
            line_geom: 投影后的线状几何体
            point_geom: 投影后的点状几何体
        
        Returns:
            {'angle': 角度, 'direction': 方向名称}
        """
        # 找到点在线上的最近投影点
        distance_along_line = line_geom.project(point_geom)
        nearest_pt = line_geom.interpolate(distance_along_line)
        
        # 计算点到最近点的方向向量
        dx = point_geom.x - nearest_pt.x
        dy = point_geom.y - nearest_pt.y
        
        # 角度计算：北=0°，顺时针
        angle = np.degrees(np.arctan2(dx, dy)) % 360
        direction = self._angle_to_direction(angle)
        
        return {'angle': angle, 'direction': direction}

    @staticmethod
    def _angle_to_direction(angle):
        """八方向转换"""
        if angle >= 337.5 or angle < 22.5: return 'north'
        elif angle < 67.5: return 'northeast'
        elif angle < 112.5: return 'east'
        elif angle < 157.5: return 'southeast'
        elif angle < 202.5: return 'south'
        elif angle < 247.5: return 'southwest'
        elif angle < 292.5: return 'west'
        else: return 'northwest'

    @staticmethod
    def _get_opposite_direction(direction):
        """获取相反方向"""
        opposites = {
            'north': 'south', 'northeast': 'southwest',
            'east': 'west', 'southeast': 'northwest',
            'south': 'north', 'southwest': 'northeast',
            'west': 'east', 'northwest': 'southeast'
        }
        return opposites.get(direction, 'north')

    def calc_semantic_relations(self):
        """语义关系推断"""
        logger.info("[语义关系] 推断中...")
        count = 0

        # 1. 基础属性分类
        for eid, ent in self.entities.items():
            if ent['type'] == 'poi':
                poi_type = ent['attributes'].get(self.poi_type_field, '')
                if poi_type:
                    self.relations.append((eid, 'has_category', f"category_{poi_type}",
                                            {'relation_type': 'semantic'}))
                    count += 1

            if ent['type'] == 'road':
                road_level = ent['attributes'].get(self.road_level_field, '')
                if road_level:
                    self.relations.append((eid, 'has_road_level', f"level_{road_level}",
                                            {'relation_type': 'semantic'}))
                    count += 1

        # 2. 基于拓扑的语义推断
        # 使用列表副本遍历，避免在迭代时修改列表
        for head, rel, tail, props in list(self.relations):
            if rel == 'contains':
                head_ent = self.entities.get(head)
                tail_ent = self.entities.get(tail)
                if not head_ent or not tail_ent: continue

                head_type = head_ent['type']
                tail_type = tail_ent['type']

                if head_type == 'block' and tail_type == 'building':
                    self.relations.append((head, 'block_has_building', tail,
                                            {'relation_type': 'semantic', 'inferred': True}))
                    count += 1
                elif head_type == 'block' and tail_type == 'poi':
                    self.relations.append((head, 'block_has_poi', tail,
                                            {'relation_type': 'semantic', 'inferred': True}))
                    count += 1
                elif head_type == 'building' and tail_type == 'poi':
                    self.relations.append((head, 'building_has_poi', tail,
                                            {'relation_type': 'semantic', 'inferred': True}))
                    count += 1

        logger.info(f"  → 语义关系: {count} 条")

    def compute_all(self):
        """执行全量计算"""
        self.calc_topological_relations()
        self.calc_delaunay_neighbors()
        self.calc_direction_relations()
        self.calc_semantic_relations()

        logger.info("=" * 60)
        logger.info(f"[总计] 关系总数: {len(self.relations)} 条")
        type_count = Counter(r[1] for r in self.relations)
        for rtype, cnt in type_count.most_common():
            logger.info(f"  {rtype:30s}: {cnt:5d}")

        return self.relations
# ============================================================
# 第三部分：知识图谱构建与存储
# ============================================================

class GeoKnowledgeGraph:
    """地理知识图谱"""

    def __init__(self, entities, relations):
        self.entities = entities
        self.relations = relations
        self.graph = nx.MultiDiGraph()
        self._build_graph()

    def _build_graph(self):
        """构建NetworkX图"""
        # 添加实体节点
        for eid, ent in self.entities.items():
            self.graph.add_node(eid, **{
                'name': ent['name'],
                'type': ent['type'],
                'lon': ent['centroid_coords'][0],
                'lat': ent['centroid_coords'][1],
                **{k: v for k, v in ent['attributes'].items()
                   if isinstance(v, (str, int, float))}
            })

        # 添加关系边
        for head, rel, tail, props in self.relations:
            self.graph.add_edge(head, tail, relation=rel, **props)

        logger.info(f"[知识图谱] 节点: {self.graph.number_of_nodes()}, 边: {self.graph.number_of_edges()}")

    def export_to_neo4j(self, uri=None, user=None, password=None, batch_size=1000):
        """
        导出到Neo4j图数据库

        Args:
            uri: Neo4j连接URI，默认使用settings配置
            user: 用户名
            password: 密码
            batch_size: 批量创建时的批次大小
        """
        # 使用settings配置作为默认值
        uri = uri or getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687')
        user = user or getattr(settings, 'NEO4J_USER', 'neo4j')
        password = password or getattr(settings, 'NEO4J_PASSWORD', 'password')

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))

        try:
            with driver.session() as session:
                logger.info("[Neo4j] 使用原生Cypher批量导入")

                # 清空数据库
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("[Neo4j] 清空数据库")

                # 创建通用索引（entity_id在所有节点上）
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.entity_id)")
                # 为每种实体类型创建索引
                index_labels = ['Road', 'Poi', 'Building', 'Block']
                for label in index_labels:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.entity_id)")
                logger.info(f"[Neo4j] 创建索引完成，等待索引就绪...")
                # 等待索引生效
                session.run("CALL db.awaitIndexes(300)")

                # 1. 批量创建实体节点
                batch_data = []
                for eid, ent in self.entities.items():
                    label = ent['type'].capitalize()
                    geom_type = ent['geometry'].geom_type
                    geom = ent['geometry']

                    # 基础属性
                    node_props = {
                        'entity_id': eid,
                        'name': ent['name'],
                        'entity_type': ent['type'],
                        'geom_type': geom_type
                    }

                    # 根据几何类型添加特定属性
                    if geom_type == 'Point':
                        node_props.update({
                            'longitude': float(geom.x),
                            'latitude': float(geom.y)
                        })
                    elif geom_type in ['LineString', 'MultiLineString']:
                        centroid = geom.centroid
                        node_props.update({
                            'longitude': float(centroid.x),
                            'latitude': float(centroid.y),
                            'length': float(geom.length),
                            'geometry': geom.wkt
                        })
                    elif geom_type in ['Polygon', 'MultiPolygon']:
                        centroid = geom.centroid
                        node_props.update({
                            'longitude': float(centroid.x),
                            'latitude': float(centroid.y),
                            'area': float(geom.area),
                            'geometry': geom.wkt
                        })

                    batch_data.append({'label': label, 'props': node_props})

                # 批量创建实体节点（使用原生Cypher，不依赖APOC）
                if batch_data:
                    # 按标签分组批量创建，同时添加Node标签便于通用索引查询
                    for label in set(row['label'] for row in batch_data):
                        label_batch = [row['props'] for row in batch_data if row['label'] == label]
                        total_batches = (len(label_batch) + batch_size - 1) // batch_size
                        for batch_idx, i in enumerate(range(0, len(label_batch), batch_size)):
                            sub_batch = label_batch[i:i + batch_size]
                            # 添加Node标签作为通用标签
                            cypher = f"""
                            UNWIND $batch AS props
                            CREATE (n:Node:{label})
                            SET n = props
                            """
                            session.run(cypher, {'batch': sub_batch})
                            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                                logger.info(f"[Neo4j] {label}节点进度: {min(i+batch_size, len(label_batch))}/{len(label_batch)}")

                    logger.info(f"[Neo4j] 创建实体节点完成: {len(batch_data)} 个")

                # 2. 创建属性节点（每个实体一个AttributeNode）
                attr_batch = []
                for eid, ent in self.entities.items():
                    attrs = {}
                    for k, v in ent['attributes'].items():
                        if v is not None and isinstance(v, (str, int, float)):
                            safe_key = k.replace(' ', '_').replace('-', '_').replace('.', '_')
                            attrs[safe_key] = v

                    if attrs:
                        attr_batch.append({'eid': eid, 'attrs': attrs})

                if attr_batch:
                    attr_total = len(attr_batch)
                    logger.info(f"[Neo4j] 开始创建 {attr_total} 个属性节点...")
                    for i in range(0, len(attr_batch), batch_size):
                        sub_batch = attr_batch[i:i + batch_size]
                        cypher = """
                        UNWIND $batch AS row
                        MATCH (e:Node {entity_id: row.eid})
                        CREATE (a:AttributeNode)
                        SET a = row.attrs
                        CREATE (e)-[:HAS_ATTRIBUTES]->(a)
                        """
                        session.run(cypher, {'batch': sub_batch})
                        if (i // batch_size) % 10 == 0 or i + batch_size >= attr_total:
                            logger.info(f"[Neo4j] 属性节点进度: {min(i+batch_size, attr_total)}/{attr_total}")

                    logger.info(f"[Neo4j] 创建属性节点完成: {attr_total} 个")

                # 3. 批量创建关系（大幅提升性能）
                rel_batch = []
                skipped_semantic = 0
                for head, rel, tail, props in self.relations:
                    # 跳过非实体关系（如 category_xxx, level_xxx）
                    if tail.startswith('category_') or tail.startswith('level_'):
                        skipped_semantic += 1
                        continue

                    props_clean = {k: v for k, v in props.items()
                                   if isinstance(v, (str, int, float))}
                    rel_batch.append({
                        'head': head,
                        'rel': rel.upper(),
                        'tail': tail,
                        'props': props_clean
                    })

                if skipped_semantic > 0:
                    logger.info(f"[Neo4j] 跳过非实体语义关系: {skipped_semantic} 条")

                if rel_batch:
                    logger.info(f"[Neo4j] 开始创建 {len(rel_batch)} 条关系...")
                    # 使用原生Cypher批量创建关系，不依赖APOC
                    rel_types = set(r['rel'] for r in rel_batch)
                    total_created = 0
                    for rel_type in rel_types:
                        type_batch = [r for r in rel_batch if r['rel'] == rel_type]
                        type_total = len(type_batch)
                        logger.info(f"[Neo4j] 关系类型 {rel_type}: {type_total} 条")
                        for i in range(0, len(type_batch), batch_size):
                            sub_batch = type_batch[i:i + batch_size]
                            cypher = f"""
                            UNWIND $batch AS row
                            MATCH (a:Node {{entity_id: row.head}})
                            MATCH (b:Node {{entity_id: row.tail}})
                            CREATE (a)-[r:{rel_type}]->(b)
                            SET r = row.props
                            """
                            session.run(cypher, {'batch': sub_batch})
                            total_created += len(sub_batch)
                            # 每10批次或完成时输出进度
                            if (i // batch_size) % 10 == 0 or i + batch_size >= type_total:
                                logger.info(f"[Neo4j] {rel_type}进度: {min(i+batch_size, type_total)}/{type_total} (总计: {total_created}/{len(rel_batch)})")

                    logger.info(f"[Neo4j] 创建关系完成: {len(rel_batch)} 条")

            logger.info("[Neo4j] 导出完成!")

        except Exception as e:
            logger.error(f"[Neo4j] 导出失败: {e}")
            raise
        finally:
            driver.close()

    # ----- 3.2 导出为 RDF/Turtle -----
    def export_to_rdf(self, output_path="geo_kg.ttl"):
        """导出为RDF Turtle格式"""
        lines = [
            "@prefix geo: <http://geo.example.org/> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "@prefix geosparql: <http://www.opengis.net/ont/geosparql#> .",
            ""
        ]

        # 实体
        for eid, ent in self.entities.items():
            safe_id = eid.replace(' ', '_')
            lines.append(
                f'geo:{safe_id} rdf:type geo:{ent["type"].capitalize()} ;')
            lines.append(f'    rdfs:label "{ent["name"]}" ;')
            lines.append(
                f'    geo:longitude "{ent["centroid_coords"][0]}"^^xsd:double ;')
            lines.append(
                f'    geo:latitude "{ent["centroid_coords"][1]}"^^xsd:double .')
            lines.append("")

        # 关系
        for head, rel, tail, props in self.relations:
            safe_head = head.replace(' ', '_')
            safe_tail = tail.replace(' ', '_')
            lines.append(f'geo:{safe_head} geo:{rel} geo:{safe_tail} .')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"[RDF] 导出到 {output_path}")

    # ----- 3.3 导出为 JSON-LD -----
    def export_to_jsonld(self, output_path="geo_kg.json"):
        """导出为JSON-LD格式"""
        nodes = []
        for eid, ent in self.entities.items():
            nodes.append({
                "@id": f"geo:{eid}",
                "@type": f"geo:{ent['type'].capitalize()}",
                "name": ent['name'],
                "longitude": ent['centroid_coords'][0],
                "latitude": ent['centroid_coords'][1],
                "attributes": ent['attributes']
            })

        edges = []
        for head, rel, tail, props in self.relations:
            edges.append({
                "source": f"geo:{head}",
                "relation": rel,
                "target": f"geo:{tail}",
                "properties": {k: v for k, v in props.items()
                               if isinstance(v, (str, int, float))}
            })

        kg_data = {
            "@context": {
                "geo": "http://geo.example.org/",
                "name": "http://schema.org/name"
            },
            "nodes": nodes,
            "edges": edges
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)

        logger.info(f"[JSON-LD] 导出到 {output_path}")


# ============================================================
# 第四部分：可视化
# ============================================================

class GeoKGVisualizer:
    """地理知识图谱可视化"""

    def __init__(self, entities, relations, spatial_calc=None):
        self.entities = entities
        self.relations = relations
        self.spatial_calc = spatial_calc

    def plot_delaunay_triangulation(self, save_path="delaunay.png"):
        """可视化Delaunay三角网（增强版）"""
        if not self.spatial_calc or not hasattr(self.spatial_calc, '_delaunay_tri'):
            logger.warning("请先计算三角网近邻")
            return

        tri = self.spatial_calc._delaunay_tri
        points = self.spatial_calc._delaunay_points
        # 优先使用去重后的映射（如果存在）
        point_to_entity = getattr(self.spatial_calc, '_unique_point_to_entity',
                                   self.spatial_calc.point_to_entity)
        point_type = getattr(self.spatial_calc, '_unique_point_type',
                             self.spatial_calc.point_type)

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # 绘制三角网
        ax.triplot(points[:, 0], points[:, 1], tri.simplices,
                   color='lightgray', linewidth=0.3, alpha=0.5)

        # 按实体类型和点类型着色
        type_colors = {
            'road': ('#e74c3c', '道路'),
            'poi': ('#2ecc71', 'POI'),
            'building': ('#3498db', '建筑物'),
            'block': ('#f39c12', '街区')
        }
        
        # 点类型标记
        pt_type_markers = {
            'centroid': 'o',   # 圆形 - 质心
            'skeleton': '^',   # 三角形 - 骨架点
            'boundary': 's'    # 方形 - 边界点
        }
        
        # 按类型分组绘制
        for etype, (color, label) in type_colors.items():
            for pt_type, marker in pt_type_markers.items():
                # 找到匹配的点索引
                indices = [idx for idx, eid in point_to_entity.items()
                          if self.entities.get(eid, {}).get('type') == etype
                          and point_type.get(idx) == pt_type]
                
                if indices:
                    pts = points[indices]
                    # 骨架点和边界点稍小
                    size = 30 if pt_type == 'centroid' else 15
                    alpha = 0.7 if pt_type == 'centroid' else 0.5
                    
                    ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker,
                               s=size, alpha=alpha, zorder=5)
            
            # 添加图例（仅显示实体类型）
            ax.scatter([], [], c=color, marker='o', s=30, label=label)

        # 添加点类型图例
        for pt_type, marker in pt_type_markers.items():
            ax.scatter([], [], c='gray', marker=marker, s=20, 
                       label=f'{pt_type}点', alpha=0.5)

        # 高亮近邻关系边（使用实体质心连线）
        near_relations = [(r[0], r[2], r[3].get('method', 'delaunay'))
                          for r in self.relations if r[1] == 'near']
        
        # 按方法类型着色连线
        method_colors = {
            'delaunay_skeleton': '#e67e22',  # 橙色 - 骨架点连接
            'delaunay_boundary': '#9b59b6',  # 紫色 - 边界点连接
            'delaunay_enhanced': '#3498db',  # 蓝色 - 增强版
            'delaunay': '#bdc3c7'            # 灰色 - 原版
        }
        
        for eid_a, eid_b, method in near_relations[:200]:
            # 使用实体质心连线
            if eid_a in self.entities and eid_b in self.entities:
                pa = self.spatial_calc.proj_coords[eid_a]
                pb = self.spatial_calc.proj_coords[eid_b]
                color = method_colors.get(method, '#bdc3c7')
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                        color=color, linewidth=0.8, alpha=0.4)

        ax.set_title('Delaunay三角网（增强版）— 骨架点+边界采样', fontsize=14)
        ax.legend(fontsize=10, loc='upper left')
        ax.set_xlabel('投影坐标 X (米)')
        ax.set_ylabel('投影坐标 Y (米)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        logger.info(f"[可视化] 三角网图保存到 {save_path}")

    def plot_knowledge_graph(self, max_nodes=200, save_path="kg_graph.png"):
        """可视化知识图谱（抽样）"""
        G = nx.DiGraph()

        # 抽样节点
        sampled_entities = dict(list(self.entities.items())[:max_nodes])
        sampled_ids = set(sampled_entities.keys())

        for eid, ent in sampled_entities.items():
            G.add_node(eid, **{'type': ent['type'], 'name': ent['name']})

        # 只添加两端都在抽样中的关系
        for head, rel, tail, props in self.relations:
            if head in sampled_ids and tail in sampled_ids:
                if rel in ['near', 'contains', 'within', 'passes_through',
                           'adjacent_to', 'intersects']:
                    G.add_edge(head, tail, relation=rel)

        fig, ax = plt.subplots(figsize=(16, 12))

        # 使用质心坐标作为布局
        pos = {}
        for eid in G.nodes():
            if eid in self.entities:
                coords = self.entities[eid]['centroid_coords']
                pos[eid] = coords

        type_colors = {
            'road': '#e74c3c',
            'poi': '#2ecc71',
            'building': '#3498db',
            'block': '#f39c12'
        }

        node_colors = [type_colors.get(self.entities.get(n, {}).get('type', ''), '#95a5a6')
                       for n in G.nodes()]

        # 边颜色
        edge_colors_map = {
            'near': '#bdc3c7',
            'contains': '#8e44ad',
            'within': '#8e44ad',
            'passes_through': '#e74c3c',
            'adjacent_to': '#2ecc71',
            'intersects': '#f39c12'
        }
        edge_colors = [edge_colors_map.get(G.edges[e].get('relation', ''), '#bdc3c7')
                       for e in G.edges()]

        nx.draw(G, pos, ax=ax,
                node_color=node_colors, node_size=20,
                edge_color=edge_colors, width=0.5, alpha=0.6,
                arrows=True, arrowsize=5)

        # 图例
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=c, label=l.upper())
                          for l, c in type_colors.items()]
        ax.legend(handles=legend_patches, fontsize=12, loc='upper left')

        ax.set_title('地理知识图谱可视化', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        logger.info(f"[可视化] 知识图谱保存到 {save_path}")


# ============================================================
# 第五部分：图谱查询接口
# ============================================================

class GeoKGQuery:
    """知识图谱查询"""

    def __init__(self, kg: GeoKnowledgeGraph):
        self.kg = kg
        self.graph = kg.graph
        self.entities = kg.entities

    def find_neighbors(self, entity_id, relation_type=None, max_hops=1):
        """查找某实体的邻居"""
        results = []
        if entity_id not in self.graph:
            return results

        for u, v, data in self.graph.out_edges(entity_id, data=True):
            if relation_type is None or data.get('relation') == relation_type:
                results.append({
                    'entity': v,
                    'name': self.entities.get(v, {}).get('name', v),
                    'relation': data.get('relation'),
                    'distance': data.get('distance_m', None)
                })
        return results

    def find_within_block(self, block_id):
        """查找某街区内的所有实体"""
        return self.find_neighbors(block_id, 'contains')

    def find_nearby_pois(self, entity_id, max_distance=300):
        """查找附近的POI"""
        neighbors = self.find_neighbors(entity_id, 'near')
        pois = [n for n in neighbors
                if self.entities.get(n['entity'], {}).get('type') == 'poi'
                and (n['distance'] is None or n['distance'] <= max_distance)]
        return pois

    def shortest_path(self, source_id, target_id):
        """两个实体间的最短路径"""
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_entity_info(self, entity_id):
        """获取实体完整信息"""
        if entity_id not in self.entities:
            return None

        ent = self.entities[entity_id]
        in_rels = [(u, d.get('relation')) for u, v, d in
                   self.graph.in_edges(entity_id, data=True)]
        out_rels = [(v, d.get('relation')) for u, v, d in
                    self.graph.out_edges(entity_id, data=True)]

        return {
            'entity': ent,
            'in_relations': in_rels[:20],
            'out_relations': out_rels[:20]
        }


# ============================================================
# 主程序入口
# ============================================================

def main():
    """
    主程序 —— 运行完整的地理知识图谱构建流程
    """

    # ===== 第1步：加载数据 =====
    logger.info("=" * 60)
    logger.info("第1步：加载SHP文件")
    logger.info("=" * 60)

    loader = GeoEntityLoader(crs_target="EPSG:4326")

    # !!!! 修改为你的SHP文件路径 !!!!
    loader.load_shp("data/road.shp",     "road",     name_field="name")
    loader.load_shp("data/poi.shp",      "poi",      name_field="name")
    loader.load_shp("data/building.shp", "building", name_field="name")
    loader.load_shp("data/block.shp",    "block",    name_field="name")

    entities = loader.entities

    # ===== 第2步：计算空间关系 =====
    logger.info("=" * 60)
    logger.info("第2步：计算空间关系（含Delaunay三角网）")
    logger.info("=" * 60)

    calc = SpatialRelationCalculator(
        entities,
        distance_threshold=500  # 500米内算近邻
    )
    relations = calc.compute_all()

    # ===== 第3步：构建知识图谱 =====
    logger.info("=" * 60)
    logger.info("第3步：构建知识图谱")
    logger.info("=" * 60)

    kg = GeoKnowledgeGraph(entities, relations)

    # ===== 第4步：导出 =====
    logger.info("=" * 60)
    logger.info("第4步：导出知识图谱")
    logger.info("=" * 60)

    kg.export_to_jsonld("output/geo_kg.json")
    kg.export_to_rdf("output/geo_kg.ttl")
    # kg.export_to_neo4j()  # 需要先启动Neo4j

    # ===== 第5步：可视化 =====
    logger.info("=" * 60)
    logger.info("第5步：可视化")
    logger.info("=" * 60)

    viz = GeoKGVisualizer(entities, relations, calc)
    viz.plot_delaunay_triangulation("output/delaunay.png")
    viz.plot_knowledge_graph(max_nodes=300, save_path="output/kg_graph.png")

    # ===== 第6步：示例查询 =====
    logger.info("=" * 60)
    logger.info("第6步：示例查询")
    logger.info("=" * 60)

    query = GeoKGQuery(kg)

    # 查询第一个POI的邻居
    poi_ids = [eid for eid, ent in entities.items() if ent['type'] == 'poi']
    if poi_ids:
        sample_poi = poi_ids[0]
        logger.info(f"查询 {entities[sample_poi]['name']} 的近邻:")
        neighbors = query.find_neighbors(sample_poi, 'near')
        for n in neighbors[:5]:
            logger.info(f"  → {n['name']} (距离: {n['distance']}m, 关系: {n['relation']})")

    # 查询第一个街区包含的实体
    block_ids = [eid for eid, ent in entities.items() if ent['type']
                 == 'block']
    if block_ids:
        sample_block = block_ids[0]
        logger.info(f"查询 {entities[sample_block]['name']} 包含的实体:")
        contained = query.find_within_block(sample_block)
        for c in contained[:5]:
            logger.info(f"  → {c['name']} ({c['relation']})")


if __name__ == "__main__":
    main()
