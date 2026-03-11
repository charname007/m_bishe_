"""
地理知识图谱构建器
输入：道路(线)、POI(点)、建筑物(面)、街区(面) 的 SHP 文件
输出：Neo4j 知识图谱 + NetworkX 可视化
"""

import geopandas as gpd
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import nearest_points
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import warnings
import sys
from loguru import logger
from config import settings
from shapely.ops import transform

import numpy as np
from pyproj import Transformer
from shapely.ops import transform
from scipy.spatial import Delaunay
from rtree import index
from collections import Counter

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
import os
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

        print(f"\n[总计] 加载实体: {len(self.entities)} 个")
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

# class SpatialRelationCalculator:
#     """空间关系计算器 —— 拓扑 + 距离 + 方向 + 三角网近邻"""

#     def __init__(self, entities, distance_threshold=500):
#         """
#         :param entities: 实体字典
#         :param distance_threshold: 近邻距离阈值(米)
#         """
#         self.entities = entities
#         self.distance_threshold = distance_threshold
#         self.relations = []  # [(head_id, relation, tail_id, properties)]

#     # ----- 2.1 拓扑关系（基于DE-9IM模型）-----
#     def calc_topological_relations(self):
#         """
#         计算拓扑关系：包含、相交、相邻
#         重点：
#           - 街区 包含 建筑物/POI
#           - 道路 穿过 街区
#           - 建筑物 临近 道路
#         """
#         print("\n[拓扑关系] 计算中...")
#         entity_list = list(self.entities.values())
#         n = len(entity_list)

#         # 建立空间索引加速
#         from rtree import index
#         idx = index.Index()
#         for i, ent in enumerate(entity_list):
#             idx.insert(i, ent['geometry'].bounds)

#         relation_count = 0

#         for i, ent_a in enumerate(entity_list):
#             # 用空间索引筛选候选
#             candidates = list(idx.intersection(
#                 ent_a['geometry'].buffer(0.001).bounds))

#             for j in candidates:
#                 if j <= i:
#                     continue
#                 ent_b = entity_list[j]

#                 geom_a = ent_a['geometry']
#                 geom_b = ent_b['geometry']

#                 try:
#                     # ---- 包含关系 ----
#                     if geom_a.contains(geom_b):
#                         self.relations.append((
#                             ent_a['id'], 'contains', ent_b['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         self.relations.append((
#                             ent_b['id'], 'within', ent_a['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         relation_count += 2

#                     elif geom_b.contains(geom_a):
#                         self.relations.append((
#                             ent_b['id'], 'contains', ent_a['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         self.relations.append((
#                             ent_a['id'], 'within', ent_b['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         relation_count += 2

#                     # ---- 相交关系 ----
#                     elif geom_a.intersects(geom_b):
#                         # 道路穿过街区
#                         if ent_a['type'] == 'road' and ent_b['type'] == 'block':
#                             self.relations.append((
#                                 ent_a['id'], 'passes_through', ent_b['id'],
#                                 {'relation_type': 'topological'}
#                             ))
#                         elif ent_b['type'] == 'road' and ent_a['type'] == 'block':
#                             self.relations.append((
#                                 ent_b['id'], 'passes_through', ent_a['id'],
#                                 {'relation_type': 'topological'}
#                             ))
#                         else:
#                             self.relations.append((
#                                 ent_a['id'], 'intersects', ent_b['id'],
#                                 {'relation_type': 'topological'}
#                             ))
#                         relation_count += 1

#                     # ---- 相邻关系（共享边界）----
#                     elif geom_a.touches(geom_b):
#                         self.relations.append((
#                             ent_a['id'], 'adjacent_to', ent_b['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         relation_count += 1

#                 except Exception as e:
#                     continue

#         print(f"  → 拓扑关系: {relation_count} 条")

#     # ----- 2.2 Delaunay三角网近邻关系（重点！）-----
#     def calc_delaunay_neighbors(self):
#         """
#         使用Delaunay三角剖分计算空间近邻关系

#         原理：
#         1. 提取所有实体质心坐标
#         2. 构建Delaunay三角网
#         3. 三角网中共享边的实体即为"空间近邻"
#         4. 过滤距离过远的近邻（阈值控制）

#         ┌──────────────────────────────────────┐
#         │  为什么用三角网而不是KNN?              │
#         │  - 三角网保证无交叉、覆盖凸包          │
#         │  - 自动适应密度变化                    │
#         │  - 数学性质好（最大化最小角）           │
#         │  - 生成的邻居关系更符合地理直觉         │
#         └──────────────────────────────────────┘
#         """
#         print("\n[三角网近邻] 构建Delaunay三角网...")

#         # Step 1: 提取所有质心
#         entity_ids = list(self.entities.keys())
#         points = np.array([self.entities[eid]['centroid_coords']
#                           for eid in entity_ids])

#         if len(points) < 3:
#             print("  → 实体数量不足3个，无法构建三角网")
#             return

#         # Step 2: 构建Delaunay三角网
#         tri = Delaunay(points)
#         print(f"  → 三角形数量: {len(tri.simplices)}")

#         # Step 3: 提取三角网的边（即近邻对）
#         neighbor_pairs = set()
#         for simplex in tri.simplices:
#             # 每个三角形有3条边 → 3对近邻
#             for k in range(3):
#                 i = simplex[k]
#                 j = simplex[(k + 1) % 3]
#                 pair = (min(i, j), max(i, j))
#                 neighbor_pairs.add(pair)

#         print(f"  → 候选近邻对: {len(neighbor_pairs)}")

#         # Step 4: 过滤并生成关系
#         relation_count = 0

#         # 转投影坐标系计算真实距离（米）
#         # 武汉大约在 EPSG:32649 (UTM Zone 49N)
#         from pyproj import Transformer
#         transformer = Transformer.from_crs(
#             "EPSG:4326", "EPSG:32649", always_xy=True)

#         # for i, j in neighbor_pairs:
#         #     eid_a = entity_ids[i]
#         #     eid_b = entity_ids[j]


#         #     # 计算真实距离（米）
#         #     xa, ya = points[i]
#         #     xb, yb = points[j]
#         #     xa_m, ya_m = transformer.transform(xa, ya)
#         #     xb_m, yb_m = transformer.transform(xb, yb)
#         #     distance = np.sqrt((xa_m - xb_m) ** 2 + (ya_m - yb_m) ** 2)
#         points_m = np.array([transformer.transform(lon, lat) for lon, lat in points])

#         for i, j in neighbor_pairs:
#             distance = np.linalg.norm(points_m[i] - points_m[j])
#             # 距离阈值过滤
#             if distance <= self.distance_threshold:
#                 self.relations.append((
#                     eid_a, 'near', eid_b,
#                     {
#                         'relation_type': 'proximity',
#                         'distance_m': round(distance, 2),
#                         'method': 'delaunay'
#                     }
#                 ))
#                 relation_count += 1

#         print(f"  → 近邻关系（≤{self.distance_threshold}m）: {relation_count} 条")

#         # 保存三角网供可视化
#         self._delaunay_tri = tri
#         self._delaunay_points = points
#         self._delaunay_ids = entity_ids

#     # ----- 2.3 方向关系 -----
#     def calc_direction_relations(self, only_for='near'):
#         """
#         计算方向关系（八方向模型）
#         只对已有近邻关系的实体对计算方向
#         """
#         print("\n[方向关系] 计算中...")

#         # 收集已有near关系的实体对
#         near_pairs = [(r[0], r[2]) for r in self.relations if r[1] == 'near']

#         from pyproj import Transformer
#         transformer = Transformer.from_crs(
#             "EPSG:4326", "EPSG:32649", always_xy=True)

#         direction_count = 0
#         for eid_a, eid_b in near_pairs:
#             ca = self.entities[eid_a]['centroid_coords']
#             cb = self.entities[eid_b]['centroid_coords']

#             # 投影坐标
#             xa, ya = transformer.transform(ca[0], ca[1])
#             xb, yb = transformer.transform(cb[0], cb[1])

#             dx = xb - xa
#             dy = yb - ya

#             # 计算角度（北为0°，顺时针）
#             angle = np.degrees(np.arctan2(dx, dy)) % 360

#             # 八方向
#             direction = self._angle_to_direction(angle)

#             self.relations.append((
#                 eid_a, f'{direction}_of', eid_b,
#                 {
#                     'relation_type': 'directional',
#                     'angle': round(angle, 1)
#                 }
#             ))
#             direction_count += 1

#         print(f"  → 方向关系: {direction_count} 条")

#     @staticmethod
#     def _angle_to_direction(angle):
#         """角度转八方向"""
#         directions = [
#             (337.5, 360, 'south'), (0, 22.5, 'south'),
#             (22.5, 67.5, 'southwest'),
#             (67.5, 112.5, 'west'),
#             (112.5, 157.5, 'northwest'),
#             (157.5, 202.5, 'north'),
#             (202.5, 247.5, 'northeast'),
#             (247.5, 292.5, 'east'),
#             (292.5, 337.5, 'southeast'),
#         ]
#         for low, high, d in directions:
#             if low <= angle < high:
#                 return d
#         return 'north'

#     # ----- 2.4 语义关系（基于属性推断）-----
#     def calc_semantic_relations(self):
#         """根据属性推断语义关系"""
#         print("\n[语义关系] 推断中...")
#         count = 0

#         # POI 属于某类型 (本体层关系)
#         for eid, ent in self.entities.items():
#             if ent['type'] == 'poi':
#                 poi_type = ent['attributes'].get(
#                     'type', ent['attributes'].get('fclass', ''))
#                 if poi_type:
#                     self.relations.append((
#                         eid, 'has_category', f"category_{poi_type}",
#                         {'relation_type': 'semantic', 'category': poi_type}
#                     ))
#                     count += 1

#             # 道路等级
#             if ent['type'] == 'road':
#                 road_level = ent['attributes'].get('fclass',
#                                                    ent['attributes'].get('highway', ''))
#                 if road_level:
#                     self.relations.append((
#                         eid, 'has_road_level', f"level_{road_level}",
#                         {'relation_type': 'semantic', 'level': road_level}
#                     ))
#                     count += 1

#         print(f"  → 语义关系: {count} 条")

#     def compute_all(self):
#         """计算所有空间关系"""
#         self.calc_topological_relations()
#         self.calc_delaunay_neighbors()
#         self.calc_direction_relations()
#         self.calc_semantic_relations()

#         print(f"\n{'='*50}")
#         print(f"[总计] 关系总数: {len(self.relations)} 条")

#         # 统计各类型
#         type_count = defaultdict(int)
#         for r in self.relations:
#             type_count[r[1]] += 1
#         for rtype, cnt in sorted(type_count.items(), key=lambda x: -x[1]):
#             print(f"  {rtype}: {cnt}")

#         return self.relations

# class SpatialRelationCalculator:
#     """空间关系计算器（改进版）"""

#     def __init__(self, entities, distance_threshold=500,
#                  src_crs="EPSG:4326", auto_proj_crs=True,poi_type_field='amenity', road_level_field='highway'):
#         self.entities = entities
#         self.distance_threshold = distance_threshold
#         self.relations = []
#         self.poi_type_field = poi_type_field
#         self.road_level_field = road_level_field

#         # ✅ 改进：自动计算投影坐标系
#         if auto_proj_crs:
#             center_lon = np.mean([e['centroid_coords'][0]
#                                  for e in entities.values()])
#             zone = int((center_lon + 180) / 6) + 1
#             is_southern = np.mean([e['centroid_coords'][1]
#                                   for e in entities.values()]) < 0
#             dst_crs = f"EPSG:{32700 + zone if is_southern else 32600 + zone}"
#             print(f"[坐标系] 自动选择: {dst_crs}")

#         from pyproj import Transformer

#         self.transformer = Transformer.from_crs(
#             src_crs, dst_crs, always_xy=True)

#         # ✅ 预计算投影坐标
#         self._prepare_projected_coords()

#     def _prepare_projected_coords(self):
#         """预计算所有点的投影坐标及投影几何"""
#         self.proj_coords = {}
#         self.proj_geometries = {} # 新增：存储投影后的几何对象

#         for eid, ent in self.entities.items():
#             lon, lat = ent['centroid_coords']
#             x, y = self.transformer.transform(lon, lat)
#             self.proj_coords[eid] = np.array([x, y])

#             # ✅ 关键修改：投影几何对象
#             # 注意：如果geometry很复杂，这一步会比较耗时，但为了保证准确性是必须的
#             self.proj_geometries[eid] = transform(self.transformer.transform, ent['geometry'])

#     def calc_topological_relations(self):
#         """✅ 保持原有逻辑，只添加容差处理"""
#         print("\n[拓扑关系] 计算中...")
#         from rtree import index

#         idx = index.Index()
#         entity_list = list(self.entities.values())
#         tolerance = 0.00001  # ✅ 地理容差

#         for i, ent_a in enumerate(entity_list):
#             # 使用投影后的几何
#             geom_a = self.proj_geometries[ent_a['id']]
#             # 容差可以直接设为 1.0 (米)
#             bounds = geom_a.buffer(1.0).bounds
#             idx.insert(i, bounds)

#         relation_count = 0
#         for i, ent_a in enumerate(entity_list):
#             candidates = list(idx.intersection(
#                 ent_a['geometry'].buffer(tolerance).bounds))

#             for j in candidates:
#                 if j <= i:
#                     continue

#                 ent_b = entity_list[j]

#                 try:
#                     if ent_a['geometry'].contains(ent_b['geometry']):
#                         self.relations.append((
#                             ent_a['id'], 'contains', ent_b['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         self.relations.append((
#                             ent_b['id'], 'within', ent_a['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         relation_count += 2

#                     elif ent_b['geometry'].contains(ent_a['geometry']):
#                         self.relations.append((
#                             ent_b['id'], 'contains', ent_a['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         self.relations.append((
#                             ent_a['id'], 'within', ent_b['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         relation_count += 2

#                     elif ent_a['geometry'].intersects(ent_b['geometry']):
#                         rel_type = 'topological'
#                         if ent_a['type'] == 'road' and ent_b['type'] == 'block':
#                             rel = 'passes_through'
#                         elif ent_b['type'] == 'road' and ent_a['type'] == 'block':
#                             ent_a, ent_b = ent_b, ent_a
#                             rel = 'passes_through'
#                         else:
#                             rel = 'intersects'

#                         self.relations.append((
#                             ent_a['id'], rel, ent_b['id'],
#                             {'relation_type': rel_type}
#                         ))
#                         relation_count += 1

#                     elif ent_a['geometry'].touches(ent_b['geometry']):
#                         self.relations.append((
#                             ent_a['id'], 'adjacent_to', ent_b['id'],
#                             {'relation_type': 'topological'}
#                         ))
#                         relation_count += 1

#                 except Exception as e:
#                     print(f"  ! 拓扑计算错误 {ent_a['id']}-{ent_b['id']}: {e}")
#                     continue

#         print(f"  → 拓扑关系: {relation_count} 条")

#     def calc_delaunay_neighbors(self):
#         """✅ 优化版：预计算投影坐标"""
#         print("\n[三角网近邻] 构建Delaunay三角网...")

#         entity_ids = list(self.entities.keys())
#         points_m = np.array([self.proj_coords[eid] for eid in entity_ids])

#         if len(points_m) < 3:
#             print("  → 实体数量不足3个，无法构建三角网")
#             return

#         tri = Delaunay(points_m)
#         print(f"  → 三角形数量: {len(tri.simplices)}")

#         # 提取近邻对
#         neighbor_pairs = set()
#         for simplex in tri.simplices:
#             for k in range(3):
#                 i = simplex[k]
#                 j = simplex[(k + 1) % 3]
#                 pair = (min(i, j), max(i, j))
#                 neighbor_pairs.add(pair)

#         print(f"  → 候选近邻对: {len(neighbor_pairs)}")

#         # ✅ 直接用预计算的投影坐标
#         relation_count = 0
#         for i, j in neighbor_pairs:
#             eid_a = entity_ids[i]
#             eid_b = entity_ids[j]

#             distance = np.linalg.norm(points_m[i] - points_m[j])

#             if distance <= self.distance_threshold:
#                 self.relations.append((
#                     eid_a, 'near', eid_b,
#                     {
#                         'relation_type': 'proximity',
#                         'distance_m': round(distance, 2),
#                         'method': 'delaunay'
#                     }
#                 ))
#                 relation_count += 1

#         print(f"  → 近邻关系（≤{self.distance_threshold}m）: {relation_count} 条")

#         # 保存供可视化
#         self._delaunay_tri = tri
#         self._delaunay_points = points_m
#         self._delaunay_ids = entity_ids

#     def calc_direction_relations(self):
#         """✅ 修正八方向"""
#         print("\n[方向关系] 计算中...")

#         near_pairs = [(r[0], r[2]) for r in self.relations if r[1] == 'near']

#         direction_count = 0
#         for eid_a, eid_b in near_pairs:
#             p_a = self.proj_coords[eid_a]
#             p_b = self.proj_coords[eid_b]

#             dx = p_b[0] - p_a[0]
#             dy = p_b[1] - p_a[1]

#             # 北=0°，顺时针
#             angle = np.degrees(np.arctan2(dx, dy)) % 360
#             direction = self._angle_to_direction(angle)

#             self.relations.append((
#                 eid_a, f'{direction}_of', eid_b,
#                 {
#                     'relation_type': 'directional',
#                     'angle': round(angle, 1)
#                 }
#             ))
#             direction_count += 1
            
#             # 反向关系：B在A的相反方向
#             opposite_direction = self._get_opposite_direction(direction)
#             opposite_angle = (angle + 180) % 360

#             self.relations.append((
#                 eid_b, f'{opposite_direction}_of', eid_a,
#                 {
#                     'relation_type': 'directional',
#                     'angle': round(opposite_angle, 1)
#                 }
#             ))
#             direction_count += 1

#         print(f"  → 方向关系: {direction_count} 条")

#     @staticmethod
#     def _angle_to_direction(angle):
#         """✅ 修正的八方向"""
#         directions = [
#             (337.5, 360, 'north'), (0, 22.5, 'north'),
#             (22.5, 67.5, 'northeast'),
#             (67.5, 112.5, 'east'),
#             (112.5, 157.5, 'southeast'),
#             (157.5, 202.5, 'south'),
#             (202.5, 247.5, 'southwest'),
#             (247.5, 292.5, 'west'),
#             (292.5, 337.5, 'northwest'),
#         ]
#         for low, high, d in directions:
#             if low <= angle < high:
#                 return d
#         return 'north'

#     @staticmethod
#     def _get_opposite_direction(direction):
#         """获取相反方向"""
#         opposite_map = {
#             'north': 'south',
#             'northeast': 'southwest',
#             'east': 'west',
#             'southeast': 'northwest',
#             'south': 'north',
#             'southwest': 'northeast',
#             'west': 'east',
#             'northwest': 'southeast'
#         }
#         return opposite_map.get(direction, 'north')

#     def calc_semantic_relations(self):
#         """✅ 增强版"""
#         print("\n[语义关系] 推断中...")
#         count = 0

#         # 1. 属性分类
#         for eid, ent in self.entities.items():
#             if ent['type'] == 'poi':
#                 poi_type = ent['attributes'].get(self.poi_type_field, '')
#                 if poi_type:
#                     self.relations.append((
#                         eid, 'has_category', f"category_{poi_type}",
#                         {'relation_type': 'semantic', 'category': poi_type}
#                     ))
#                     count += 1

#             if ent['type'] == 'road':
#                 road_level = ent['attributes'].get(self.road_level_field, '')
#                 if road_level:
#                     self.relations.append((
#                         eid, 'has_road_level', f"level_{road_level}",
#                         {'relation_type': 'semantic', 'level': road_level}
#                     ))
#                     count += 1
                    
                    
#         # 2. 关系推断

#         for head, rel, tail, props in list(self.relations):
#             if rel == 'contains':
#                 head_ent = self.entities.get(head)
#                 tail_ent = self.entities.get(tail)

#                 if not head_ent or not tail_ent:
#                     continue

#                 head_type = head_ent['type']
#                 tail_type = tail_ent['type']

#                 # 街区包含建筑物
#                 if head_type == 'block' and tail_type == 'building':
#                     self.relations.append((
#                         head, 'block_has_building', tail,
#                         {'relation_type': 'semantic', 'inferred': True}
#                     ))
#                     count += 1

#                 # 街区包含POI
#                 elif head_type == 'block' and tail_type == 'poi':
#                     self.relations.append((
#                         head, 'block_has_poi', tail,
#                         {'relation_type': 'semantic', 'inferred': True}
#                     ))
#                     count += 1

#                 # 建筑物包含POI
#                 elif head_type == 'building' and tail_type == 'poi':
#                     self.relations.append((
#                         head, 'building_has_poi', tail,
#                         {'relation_type': 'semantic', 'inferred': True}
#                     ))
#                     count += 1

#         print(f"  → 语义关系: {count} 条")

#     def compute_all(self):
#         """计算所有关系"""
#         self.calc_topological_relations()
#         self.calc_delaunay_neighbors()
#         self.calc_direction_relations()
#         self.calc_semantic_relations()

#         print(f"\n{'='*60}")
#         print(f"[总计] 关系总数: {len(self.relations)} 条")

#         from collections import Counter
#         type_count = Counter(r[1] for r in self.relations)
#         for rtype, cnt in type_count.most_common():
#             print(f"  {rtype:30s}: {cnt:5d}")

#         return self.relations
    
    



class SpatialRelationCalculator:
    """空间关系计算器（修正版：统一投影坐标系 + 优化拓扑逻辑）
    
    其中为了解决三角网在面和线要素上的不足，采用了几何方法来判断intersection和touches关系，保证了更准确的拓扑关系计算。"""

    def __init__(self, entities, distance_threshold=500,
                    src_crs="EPSG:4326", auto_proj_crs=True,
                    poi_type_field='amenity', road_level_field='highway'):
        self.entities = entities
        self.distance_threshold = distance_threshold
        self.relations = []
        self.poi_type_field = poi_type_field
        self.road_level_field = road_level_field

        # ✅ 1. 自动计算投影坐标系 (UTM)
        if auto_proj_crs:
            center_lon = np.mean([e['centroid_coords'][0]
                                    for e in entities.values()])
            zone = int((center_lon + 180) / 6) + 1
            is_southern = np.mean([e['centroid_coords'][1]
                                    for e in entities.values()]) < 0
            # UTM Zone计算: 北半球32601-32660, 南半球32701-32760
            self.dst_crs = f"EPSG:{32700 + zone if is_southern else 32600 + zone}"
            print(f"[坐标系] 自动选择: {self.dst_crs} (单位: 米)")
        else:
            self.dst_crs = src_crs # 如果不自动投影，需确保输入数据已是投影坐标系

        self.transformer = Transformer.from_crs(
            src_crs, self.dst_crs, always_xy=True)

        # ✅ 2. 预计算投影坐标与投影几何
        self.proj_coords = {}
        self.proj_geometries = {}
        self._prepare_projected_data()

    def _prepare_projected_data(self):
        """预计算所有点的投影坐标及几何体的投影"""
        print("  [预处理] 正在投影几何体...")
        for eid, ent in self.entities.items():
            lon, lat = ent['centroid_coords']
            x, y = self.transformer.transform(lon, lat)
            self.proj_coords[eid] = np.array([x, y])

            # 关键修复：将原始几何体投影到UTM坐标系
            # 这样后续计算的buffer和距离单位都是'米'
            self.proj_geometries[eid] = transform(self.transformer.transform, ent['geometry'])
        print("  [预处理] 投影完成。")

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
        print("\n[拓扑关系] 计算中（容错模式）...")

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
                    print(f"  ! 拓扑计算错误 {eid_a}-{eid_b}: {e}")
                    continue

        print(f"  → 拓扑关系: {relation_count} 条")

    def _add_relation(self, head, rel, tail, props=None):
        """辅助方法：添加关系"""
        if props is None:
            props = {'relation_type': 'topological'}
        self.relations.append((head, rel, tail, props))
        
        
    def calc_delaunay_neighbors(self):
        """✅ 优化版：基于投影坐标构建三角网"""
        print("\n[三角网近邻] 构建Delaunay三角网...")

        entity_ids = list(self.entities.keys())
        # 使用预计算的投影坐标 (单位: 米)
        points_m = np.array([self.proj_coords[eid] for eid in entity_ids])

        if len(points_m) < 3:
            print("  → 实体数量不足3个，无法构建三角网")
            return

        tri = Delaunay(points_m)
        print(f"  → 三角形数量: {len(tri.simplices)}")

        neighbor_pairs = set()
        for simplex in tri.simplices:
            for k in range(3):
                i = simplex[k]
                j = simplex[(k + 1) % 3]
                pair = (min(i, j), max(i, j))
                neighbor_pairs.add(pair)

        relation_count = 0
        for i, j in neighbor_pairs:
            eid_a = entity_ids[i]
            eid_b = entity_ids[j]

            # 计算欧氏距离 (单位: 米)
            distance = np.linalg.norm(points_m[i] - points_m[j])

            if distance <= self.distance_threshold:
                self.relations.append((
                    eid_a, 'near', eid_b,
                    {
                        'relation_type': 'proximity',
                        'distance_m': round(distance, 2),
                        'method': 'delaunay'
                    }
                ))
                relation_count += 1

        print(f"  → 近邻关系（≤{self.distance_threshold}m）: {relation_count} 条")
        self._delaunay_tri = tri

    def calc_direction_relations(self):
        """修正版：基于拓扑邻接 + 三角网邻近 计算方向"""
        print("\n[方向关系] 计算中...")

        # ✅ 修正：收集所有空间上接近的关系，不仅仅是三角网生成的 'near'
        # 1. 来自三角网的近邻
        near_pairs = set([(r[0], r[2]) for r in self.relations if r[1] == 'near'])

        # 2. 来自拓扑计算的邻接 -> 它们也是空间邻居，需要计算方向！
        adjacent_pairs = set([(r[0], r[2]) for r in self.relations if r[1] == 'adjacent_to'])

        # 3. 来自拓扑计算的相交/穿过
        intersect_pairs = set([(r[0], r[2]) for r in self.relations if r[1] in ['intersects', 'passes_through']])

        # 合并所有需要计算方向的对
        all_spatial_pairs = near_pairs.union(adjacent_pairs).union(intersect_pairs)

        direction_count = 0
        for eid_a, eid_b in all_spatial_pairs:
            p_a = self.proj_coords[eid_a]
            p_b = self.proj_coords[eid_b]

            dx = p_b[0] - p_a[0]
            dy = p_b[1] - p_a[1]

            # 角度计算：北=0°，顺时针
            angle = np.degrees(np.arctan2(dx, dy)) % 360
            direction = self._angle_to_direction(angle)

            self.relations.append((
                eid_a, f'{direction}_of', eid_b,
                {'relation_type': 'directional', 'angle': round(angle, 1)}
            ))
            direction_count += 1

            # 反向关系
            opposite_direction = self._get_opposite_direction(direction)
            opposite_angle = (angle + 180) % 360
            self.relations.append((
                eid_b, f'{opposite_direction}_of', eid_a,
                {'relation_type': 'directional', 'angle': round(opposite_angle, 1)}
            ))
            direction_count += 1

        print(f"  → 方向关系: {direction_count} 条")

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
        print("\n[语义关系] 推断中...")
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

        print(f"  → 语义关系: {count} 条")

    def compute_all(self):
        """执行全量计算"""
        self.calc_topological_relations()
        self.calc_delaunay_neighbors()
        self.calc_direction_relations()
        self.calc_semantic_relations()

        print(f"\n{'='*60}")
        print(f"[总计] 关系总数: {len(self.relations)} 条")
        type_count = Counter(r[1] for r in self.relations)
        for rtype, cnt in type_count.most_common():
            print(f"  {rtype:30s}: {cnt:5d}")

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

        print(f"\n[知识图谱] 节点: {self.graph.number_of_nodes()}, "
              f"边: {self.graph.number_of_edges()}")

    # ----- 3.1 导出为 Neo4j -----
    def export_to_neo4j(self, uri="bolt://localhost:7687",
                        user="neo4j", password="password"):
        """导出到Neo4j图数据库"""
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            # 清空数据库
            session.run("MATCH (n) DETACH DELETE n")

            # 3. 创建索引
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Road) ON (n.entity_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Poi) ON (n.entity_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Building) ON (n.entity_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Block) ON (n.entity_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Attribute) ON (n.name, n.value)")

            # 创建节点
            # for eid, ent in self.entities.items():
            #     label = ent['type'].capitalize()  # Road, Poi, Building, Block

            #     # 根据几何类型选择不同的存储策略
            #     geom_type = ent['geometry'].geom_type
            #     geom = ent['geometry']

            #     # 基础属性
            #     node_props = {
            #         'entity_id': eid,
            #         'name': ent['name'],
            #         'entity_type': ent['type'],
            #         'geom_type': geom_type
            #     }

            #     # 根据几何类型添加特定属性
            #     if geom_type == 'Point':
            #         # 点要素：直接存储坐标
            #         node_props.update({
            #             'longitude': geom.x,
            #             'latitude': geom.y
            #         })
            #     elif geom_type in ['LineString', 'MultiLineString']:
            #         # 线要素：存储质心和几何信息
            #         centroid = geom.centroid
            #         node_props.update({
            #             'longitude': centroid.x,
            #             'latitude': centroid.y,
            #             'length': geom.length,
            #             'geometry': geom.wkt  # Well-Known Text格式
            #         })
            #     elif geom_type in ['Polygon', 'MultiPolygon']:
            #         # 面要素：存储质心、面积和几何信息
            #         centroid = geom.centroid
            #         node_props.update({
            #             'longitude': centroid.x,
            #             'latitude': centroid.y,
            #             'area': geom.area,
            #             'geometry': geom.wkt
            #         })

            #     # 使用参数化查询，避免SQL注入
            #     cypher = f"""
            #     CREATE (n:{label} $props)
            #     """
            #     session.run(cypher, {'props': node_props})
            # 1. 先在 Python 端把所有数据准备好
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
                        'longitude': geom.x,
                        'latitude': geom.y
                    })
                elif geom_type in ['LineString', 'MultiLineString']:
                    centroid = geom.centroid
                    node_props.update({
                        'longitude': centroid.x,
                        'latitude': centroid.y,
                        'length': geom.length,
                        'geometry': geom.wkt
                    })
                elif geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = geom.centroid
                    node_props.update({
                        'longitude': centroid.x,
                        'latitude': centroid.y,
                        'area': geom.area,
                        'geometry': geom.wkt
                    })

                # 将处理好的数据加入列表
                batch_data.append({
                    'label': label,
                    'props': node_props
                })

            # 2. 批量提交（需要 APOC 插件支持）
            # 如果没有 APOC，请看下方的“备用方案”
            if batch_data:
                cypher = """
                UNWIND $batch AS row
                CALL apoc.create.node([row.label], row.props) YIELD node
                RETURN count(node)
                """
                session.run(cypher, {'batch': batch_data})
                logger.info(f"Batch created {len(batch_data)} entities.")
                
                
                
            # 2. 创建属性节点和关系（统一 Attribute 标签，避免标签爆炸）
            # for eid, ent in self.entities.items():
            #     # 构建属性列表
            #     attributes = [
            #         {'name': k, 'value': v, 'type': type(v).__name__}
            #         for k, v in ent['attributes'].items()
            #         if v and isinstance(v, (str, int, float))
            #     ]

            #     if not attributes:
            #         logger.debug(f"Entity {eid} has no valid attributes")
            #         continue

            #     try:
            #         # 简化 Cypher：直接 MERGE 属性节点和关系
            #         cypher = """
            #         MATCH (f {entity_id: $eid})
            #         UNWIND $attributes AS attr
            #         MERGE (a:Attribute {name: attr.name, value: attr.value})
            #         ON CREATE SET a.type = attr.type
            #         MERGE (f)-[:HAS_ATTRIBUTE]->(a)
            #         """
            #         session.run(cypher, {'eid': eid, 'attributes': attributes})
            #         logger.debug(f"Created {len(attributes)} attributes for {eid}")

            #     except Exception as e:
            #         logger.error(f"Failed to create attributes for {eid}: {e}")
                    
            # 准备批量数据
            batch_data = []
            for eid, ent in self.entities.items():
                for k, v in ent['attributes'].items():
                    if v and isinstance(v, (str, int, float)):
                        batch_data.append({
                            'eid': eid,
                            'name': k,
                            'value': v,
                            'type': type(v).__name__
                        })

            # 批量提交（一次性处理所有属性）
            if batch_data:
                # 注意：这里修改了 MERGE 逻辑，加入了 entity_id，确保属性私有化
                # 同时也演示了如何批量处理，避免 for 循环
                cypher = """
                UNWIND $batch AS row
                MATCH (f {entity_id: row.eid})
                MERGE (a:Attribute {
                    entity_id: row.eid,  -- 关键：加入实体ID隔离
                    name: row.name
                })
                ON CREATE SET a.value = row.value, a.type = row.type
                ON MATCH SET a.value = row.value  // 更新值
                MERGE (f)-[:HAS_ATTRIBUTE]->(a)
                """
                session.run(cypher, {'batch': batch_data})
                logger.info(f"Batch processed {len(batch_data)} attributes.")

            # 4. 创建关系
            for head, rel, tail, props in self.relations:
                rel_type = rel.upper()
                props_clean = {k: v for k, v in props.items()
                                if isinstance(v, (str, int, float))}

                cypher = f"""
                MATCH (a {{entity_id: $head}})
                MATCH (b {{entity_id: $tail}})
                CREATE (a)-[r:{rel_type} $props]->(b)
                """
                session.run(cypher, {
                    'head': head, 'tail': tail, 'props': props_clean
                })

            driver.close()
            print("[Neo4j] 导出完成!")

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

        print(f"[RDF] 导出到 {output_path}")

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

        print(f"[JSON-LD] 导出到 {output_path}")


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
        """可视化Delaunay三角网"""
        if not self.spatial_calc or not hasattr(self.spatial_calc, '_delaunay_tri'):
            print("请先计算三角网近邻")
            return

        tri = self.spatial_calc._delaunay_tri
        points = self.spatial_calc._delaunay_points
        entity_ids = self.spatial_calc._delaunay_ids

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # 绘制三角网
        ax.triplot(points[:, 0], points[:, 1], tri.simplices,
                   color='lightgray', linewidth=0.3, alpha=0.5)

        # 按类型着色
        type_colors = {
            'road': ('#e74c3c', 's', '道路'),
            'poi': ('#2ecc71', '^', 'POI'),
            'building': ('#3498db', 'o', '建筑物'),
            'block': ('#f39c12', 'D', '街区')
        }

        for etype, (color, marker, label) in type_colors.items():
            mask = [self.entities[eid]['type'] == etype for eid in entity_ids]
            pts = points[mask]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker,
                           s=30, label=label, zorder=5, alpha=0.7)

        # 高亮近邻关系边
        near_relations = [(r[0], r[2])
                          for r in self.relations if r[1] == 'near']
        id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

        for eid_a, eid_b in near_relations[:200]:  # 限制显示数量
            if eid_a in id_to_idx and eid_b in id_to_idx:
                pa = points[id_to_idx[eid_a]]
                pb = points[id_to_idx[eid_b]]
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                        color='#e67e22', linewidth=0.8, alpha=0.4)

        ax.set_title('Delaunay三角网 — 地理实体空间近邻关系', fontsize=14)
        ax.legend(fontsize=12)
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[可视化] 三角网图保存到 {save_path}")

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
        print(f"[可视化] 知识图谱保存到 {save_path}")


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
    print("=" * 60)
    print("第1步：加载SHP文件")
    print("=" * 60)

    loader = GeoEntityLoader(crs_target="EPSG:4326")

    # !!!! 修改为你的SHP文件路径 !!!!
    loader.load_shp("data/road.shp",     "road",     name_field="name")
    loader.load_shp("data/poi.shp",      "poi",      name_field="name")
    loader.load_shp("data/building.shp", "building", name_field="name")
    loader.load_shp("data/block.shp",    "block",    name_field="name")

    entities = loader.entities

    # ===== 第2步：计算空间关系 =====
    print("\n" + "=" * 60)
    print("第2步：计算空间关系（含Delaunay三角网）")
    print("=" * 60)

    calc = SpatialRelationCalculator(
        entities,
        distance_threshold=500  # 500米内算近邻
    )
    relations = calc.compute_all()

    # ===== 第3步：构建知识图谱 =====
    print("\n" + "=" * 60)
    print("第3步：构建知识图谱")
    print("=" * 60)

    kg = GeoKnowledgeGraph(entities, relations)

    # ===== 第4步：导出 =====
    print("\n" + "=" * 60)
    print("第4步：导出知识图谱")
    print("=" * 60)

    kg.export_to_jsonld("output/geo_kg.json")
    kg.export_to_rdf("output/geo_kg.ttl")
    # kg.export_to_neo4j()  # 需要先启动Neo4j

    # ===== 第5步：可视化 =====
    print("\n" + "=" * 60)
    print("第5步：可视化")
    print("=" * 60)

    viz = GeoKGVisualizer(entities, relations, calc)
    viz.plot_delaunay_triangulation("output/delaunay.png")
    viz.plot_knowledge_graph(max_nodes=300, save_path="output/kg_graph.png")

    # ===== 第6步：示例查询 =====
    print("\n" + "=" * 60)
    print("第6步：示例查询")
    print("=" * 60)

    query = GeoKGQuery(kg)

    # 查询第一个POI的邻居
    poi_ids = [eid for eid, ent in entities.items() if ent['type'] == 'poi']
    if poi_ids:
        sample_poi = poi_ids[0]
        print(f"\n查询 {entities[sample_poi]['name']} 的近邻:")
        neighbors = query.find_neighbors(sample_poi, 'near')
        for n in neighbors[:5]:
            print(
                f"  → {n['name']} (距离: {n['distance']}m, 关系: {n['relation']})")

    # 查询第一个街区包含的实体
    block_ids = [eid for eid, ent in entities.items() if ent['type']
                 == 'block']
    if block_ids:
        sample_block = block_ids[0]
        print(f"\n查询 {entities[sample_block]['name']} 包含的实体:")
        contained = query.find_within_block(sample_block)
        for c in contained[:5]:
            print(f"  → {c['name']} ({c['relation']})")


if __name__ == "__main__":
    main()
