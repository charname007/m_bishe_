"""
提取洪山区外边界坐标
从包含内部区域边界的shapefile中提取外边界

使用方式：
    python scripts/extract_boundary.py
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union, linemerge
import json


def analyze_shapefile(shp_path: str):
    """分析shapefile结构"""
    print(f"\n读取: {shp_path}")
    gdf = gpd.read_file(shp_path)

    print("=" * 60)
    print("数据概览:")
    print(f"  记录数: {len(gdf)}")
    print(f"  列: {list(gdf.columns)}")
    print(f"  几何类型: {gdf.geometry.geom_type.unique()}")
    print(f"  CRS: {gdf.crs}")

    print("\n属性数据:")
    if 'geometry' in gdf.columns:
        print(gdf.drop(columns='geometry'))
    else:
        print(gdf)

    # 分析几何结构
    print("\n几何分析:")
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        geom_type = geom.geom_type

        if geom_type == 'Polygon':
            # 检查是否有内环（孔洞）
            rings_count = len(geom.interior) if hasattr(geom, 'interior') else 0
            print(f"  [{idx}] Polygon: 外边界点数={len(geom.exterior.coords)}, 内环数={rings_count}")
        elif geom_type == 'MultiPolygon':
            poly_count = len(geom.geoms)
            print(f"  [{idx}] MultiPolygon: 包含 {poly_count} 个Polygon")
            for i, poly in enumerate(geom.geoms):
                rings_count = len(poly.interior) if hasattr(poly, 'interior') else 0
                print(f"    子Polygon[{i}]: 外边界点数={len(poly.exterior.coords)}, 内环数={rings_count}")
        elif geom_type == 'LineString':
            print(f"  [{idx}] LineString: 点数={len(geom.coords)}")
        elif geom_type == 'MultiLineString':
            print(f"  [{idx}] MultiLineString: 包含 {len(geom.geoms)} 条线")

    return gdf


def extract_outer_boundary(gdf, output_path: str = None):
    """
    提取外边界（排除内部区域边界）

    策略：
    1. 如果是Polygon类型，取最大的外边界（通常是行政区域外边界）
    2. 如果是MultiPolygon，合并后取最大面积的面
    3. 如果是LineString/MultiLineString，需要判断哪些是外边界

    Args:
        gdf: GeoDataFrame
        output_path: 输出文件路径

    Returns:
        外边界坐标列表 [(lon, lat), ...]
    """
    print("\n" + "=" * 60)
    print("提取外边界:")

    all_geoms = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        all_geoms.append(geom)

    # 确定几何类型
    geom_types = set(g.geom_type for g in all_geoms)

    outer_coords = []

    if 'Polygon' in geom_types or 'MultiPolygon' in geom_types:
        # 面状数据：找最大面积的外边界
        print("  策略: 找最大面积的Polygon外边界")

        # 合并所有Polygon
        polygons = []
        for g in all_geoms:
            if g.geom_type == 'Polygon':
                polygons.append(g)
            elif g.geom_type == 'MultiPolygon':
                polygons.extend(g.geoms)

        # 计算每个Polygon的面积
        areas = [(p, p.area, len(p.exterior.coords)) for p in polygons]
        areas.sort(key=lambda x: x[1], reverse=True)

        # 取最大的Polygon（通常是外边界）
        largest = areas[0]
        largest_poly, largest_area, largest_points = largest

        print(f"  最大Polygon: 面积={largest_area:.6f}, 点数={largest_points}")
        print(f"  内环数: {len(largest_poly.interiors) if hasattr(largest_poly, 'interiors') else 0}")

        # 取外边界坐标
        outer_coords = list(largest_poly.exterior.coords)

        # 检查是否有其他较大的Polygon（可能是内部区域）
        if len(areas) > 1:
            print(f"\n  其他Polygon (可能为内部区域):")
            for i, (p, area, pts) in enumerate(areas[1:6]):  # 显示前5个
                print(f"    [{i+1}] 面积={area:.6f}, 点数={pts}")

    elif 'LineString' in geom_types or 'MultiLineString' in geom_types:
        # 线状数据：需要合并线段形成闭合边界
        print("  策略: 合并所有LineString形成闭合边界")

        # 收集所有线段
        lines = []
        for g in all_geoms:
            if g.geom_type == 'LineString':
                lines.append(g)
            elif g.geom_type == 'MultiLineString':
                lines.extend(g.geoms)

        print(f"  收集线段数: {len(lines)}")

        # 合并为一条线
        merged = linemerge(unary_union(lines))

        if merged.geom_type == 'LineString':
            print(f"  合并后: LineString, 点数={len(merged.coords)}")
            coords = list(merged.coords)

            # 检查是否闭合
            if coords[0] != coords[-1]:
                print("  警告: 边界未闭合，将手动闭合")
                coords.append(coords[0])  # 手动闭合

            outer_coords = coords

        elif merged.geom_type == 'MultiLineString':
            print(f"  合并后: MultiLineString, 包含 {len(merged.geoms)} 条线")
            print("  线段未完全连接，尝试找最长闭合路径...")

            # 尝试找最长的一条线作为外边界
            lengths = [(l, l.length, len(l.coords)) for l in merged.geoms]
            lengths.sort(key=lambda x: x[1], reverse=True)

            for l, length, pts in lengths[:5]:
                coords = list(l.coords)
                is_closed = coords[0] == coords[-1]
                print(f"    线段: 长度={length:.6f}, 点数={pts}, 闭合={is_closed}")

                # 取最长的一条（假设是外边界）
                if length > 0.2:  # 长度阈值
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])  # 手动闭合
                    outer_coords = coords
                    break

        else:
            print(f"  合并后几何类型: {merged.geom_type}")
            # 尝试提取坐标
            try:
                outer_coords = list(merged.coords)
                if outer_coords[0] != outer_coords[-1]:
                    outer_coords.append(outer_coords[0])
            except:
                print("  无法提取坐标")

    else:
        print(f"  未知的几何类型: {geom_types}")

    if outer_coords:
        print(f"\n  外边界坐标数: {len(outer_coords)}")
        print(f"  坐标范围:")
        lons = [c[0] for c in outer_coords]
        lats = [c[1] for c in outer_coords]
        print(f"    经度: {min(lons):.6f} ~ {max(lons):.6f}")
        print(f"    纬度: {min(lats):.6f} ~ {max(lats):.6f}")

        # 保存结果
        if output_path:
            # 保存为JSON
            boundary_data = {
                "name": "洪山区外边界",
                "coordinates": outer_coords,
                "point_count": len(outer_coords),
                "lon_range": [min(lons), max(lons)],
                "lat_range": [min(lats), max(lats)],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(boundary_data, f, ensure_ascii=False, indent=2)

            print(f"\n  已保存到: {output_path}")

            # 生成高德API格式（坐标字符串）
            amap_coords = "|".join([f"{lon},{lat}" for lon, lat in outer_coords])
            amap_format_path = output_path.replace(".json", "_amap.txt")
            with open(amap_format_path, "w", encoding="utf-8") as f:
                f.write(amap_coords)
            print(f"  高德格式已保存到: {amap_format_path}")

            # 保存为新的shapefile（仅外边界）
            shp_output_path = output_path.replace(".json", ".shp")
            outer_poly = Polygon(outer_coords)
            outer_gdf = gpd.GeoDataFrame(
                {"name": ["洪山区外边界"]},
                geometry=[outer_poly],
                crs=gdf.crs
            )
            outer_gdf.to_file(shp_output_path)
            print(f"  Shapefile已保存到: {shp_output_path}")

    return outer_coords


def main():
    """主函数"""
    shp_path = Path(__file__).parent.parent / "shp2kg" / "shpfiles" / "hsq_boundary.shp"
    output_path = Path(__file__).parent.parent / "data" / "hsq_outer_boundary.json"

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 分析shapefile
    gdf = analyze_shapefile(str(shp_path))

    # 提取外边界
    outer_coords = extract_outer_boundary(gdf, str(output_path))

    return outer_coords


if __name__ == "__main__":
    main()