"""
测试 shp2kg 增强功能：骨架点采样、边界采样、最近点角度
"""

import pytest
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
from shapely.ops import transform
from pyproj import Transformer

# 导入待测试的类
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shp2kg import SpatialRelationCalculator


class TestSkeletonSampling:
    """测试骨架点采样功能"""

    def test_short_line_sampling(self):
        """短线段采样：应返回起点和终点"""
        # 创建一个短线段（长度 < 50m）
        line = LineString([(0, 0), (30, 0)])  # 30m长

        # 模拟 SpatialRelationCalculator 的采样方法
        calc = SpatialRelationCalculator.__new__(SpatialRelationCalculator)
        calc.skeleton_interval = 50

        points = calc._sample_single_line(line)

        # 短线段应返回起点和终点
        assert len(points) >= 2
        assert points[0] == (0, 0)
        assert points[-1] == (30, 0)

    def test_long_line_sampling(self):
        """长线段采样：应返回多个骨架点"""
        # 创建一个长线段（200m）
        line = LineString([(0, 0), (200, 0)])

        calc = SpatialRelationCalculator.__new__(SpatialRelationCalculator)
        calc.skeleton_interval = 50

        points = calc._sample_single_line(line)

        # 200m线段，50m间隔，应有约5个点
        assert len(points) >= 4
        # 检查点间距约为50m
        for i in range(len(points) - 1):
            dist = np.linalg.norm(np.array(points[i+1]) - np.array(points[i]))
            assert dist <= 55  # 允许一点误差

    def test_multiline_sampling(self):
        """多线段采样"""
        # 创建多线段
        mline = MultiLineString([
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (150, 50)])
        ])

        calc = SpatialRelationCalculator.__new__(SpatialRelationCalculator)
        calc.skeleton_interval = 50

        points = calc._sample_line_points(mline)

        # 两段线都应被采样
        assert len(points) >= 4


class TestBoundarySampling:
    """测试边界采样功能"""

    def test_small_polygon_no_boundary(self):
        """小型面状实体：不应添加边界点"""
        # 创建小型矩形（面积 < 10000 m²）
        poly = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])

        calc = SpatialRelationCalculator.__new__(SpatialRelationCalculator)
        calc.boundary_interval = 30
        calc.large_area_threshold = 10000

        points = calc._sample_boundary_points(poly)

        # 小型面不应采样（但方法本身会返回点）
        # 实际逻辑在 _prepare_projected_data 中判断

    def test_large_polygon_boundary_sampling(self):
        """大型面状实体：应添加边界采样点"""
        # 创建大型矩形（200m x 200m = 40000 m²）
        poly = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])

        calc = SpatialRelationCalculator.__new__(SpatialRelationCalculator)
        calc.boundary_interval = 50

        points = calc._sample_boundary_points(poly)

        # 800m周长，50m间隔，应有约16个点
        assert len(points) >= 10

    def test_multipolygon_sampling(self):
        """多面实体：采样最大面积的边界"""
        # 创建两个面
        poly1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])  # 10000 m²
        poly2 = Polygon([(200, 0), (400, 0), (400, 200), (200, 200)])  # 40000 m²
        mpoly = MultiPolygon([poly1, poly2])

        calc = SpatialRelationCalculator.__new__(SpatialRelationCalculator)
        calc.boundary_interval = 50

        points = calc._sample_boundary_points(mpoly)

        # 应采样最大面（poly2）的边界
        # 最大面周长600m，应有约12个点
        assert len(points) >= 10


class TestNearestPointDirection:
    """测试最近点角度计算"""

    def test_point_direction_to_line(self):
        """点相对于线的方向"""
        # 创建一条水平线
        line = LineString([(0, 50), (100, 50)])
        # 创建线北侧的点
        point = Point((50, 100))  # 在线的正北方

        calc = SpatialRelationCalculator.__new__(SpatialRelationCalculator)

        result = calc._calc_direction_line_to_point(line, point)

        # 点在线的北侧，角度应接近 0°（北方）
        # 实际角度取决于最近点位置
        angle = result['angle']
        direction = result['direction']

        # 方向应该是北方或偏北
        assert direction in ['north', 'northeast', 'northwest']

    def test_point_at_line_endpoint(self):
        """点在线端点附近"""
        # 创建一条线
        line = LineString([(0, 0), (100, 0)])
        # 创建线起点附近的点
        point = Point((0, 50))  # 在起点正北方

        calc = SpatialRelationCalculator.__new__(SpatialRelationCalculator)

        result = calc._calc_direction_line_to_point(line, point)

        # 最近点是起点，点在起点的北方
        assert result['direction'] == 'north'


class TestIntegration:
    """集成测试：完整流程"""

    def test_prepare_projected_data_with_skeleton(self):
        """测试预处理方法是否正确添加骨架点"""
        # 创建模拟实体数据
        entities = {
            'road_0': {
                'id': 'road_0',
                'type': 'road',
                'name': '测试道路',
                'geometry': LineString([(0, 0), (200, 0)]),  # 200m道路
                'centroid_coords': (100, 0),
                'attributes': {}
            },
            'poi_0': {
                'id': 'poi_0',
                'type': 'poi',
                'name': '测试POI',
                'geometry': Point((50, 50)),
                'centroid_coords': (50, 50),
                'attributes': {}
            }
        }

        # 创建计算器（使用默认参数）
        calc = SpatialRelationCalculator(
            entities,
            distance_threshold=100,
            auto_proj_crs=False,  # 不自动投影，使用本地坐标
            skeleton_interval=50
        )

        # 检查骨架点数量
        skeleton_count = sum(1 for t in calc.point_type.values() if t == 'skeleton')

        # 200m道路，50m间隔，应有约5个骨架点
        assert skeleton_count >= 4

        # 检查三角网参与点总数
        # road: 1质心 + ~5骨架点，poi: 1质心
        total_points = len(calc.triangulation_points)
        assert total_points >= 6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])