"""
简化洪山区外边界坐标
使用Douglas-Peucker算法减少坐标点数量，适应高德API限制

高德API polygon参数限制：
- 坐标点数量建议不超过500个
- 字符串长度有限制

使用方式：
    python scripts/simplify_boundary.py
    python scripts/simplify_boundary.py --tolerance 0.001  # 调整简化程度
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from shapely.geometry import Polygon, LineString


def douglas_peucker_simplify(coords: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    """
    使用Douglas-Peucker算法简化坐标序列

    Args:
        coords: 原始坐标列表 [(lon, lat), ...]
        tolerance: 简化容差（单位与坐标相同，如经纬度）

    Returns:
        简化后的坐标列表
    """
    # 创建LineString
    line = LineString(coords)

    # 使用shapely的simplify方法（Douglas-Peucker算法）
    simplified_line = line.simplify(tolerance, preserve_topology=True)

    # 提取简化后的坐标
    simplified_coords = list(simplified_line.coords)

    # 确保闭合（首尾坐标相同）
    if simplified_coords[0] != simplified_coords[-1]:
        simplified_coords.append(simplified_coords[0])

    return simplified_coords


def adaptive_simplify(coords: List[Tuple[float, float]], target_points: int = 400) -> List[Tuple[float, float]]:
    """
    自适应简化：逐步增加容差直到达到目标点数

    Args:
        coords: 原始坐标列表
        target_points: 目标点数量

    Returns:
        简化后的坐标列表
    """
    # 初始容差估算
    # 计算边界总长度
    line = LineString(coords)
    total_length = line.length

    # 初始容差：总长度的1%
    tolerance = total_length * 0.01

    best_coords = coords
    prev_count = len(coords)

    print(f"原始点数: {len(coords)}, 边界长度: {total_length:.4f}")
    print(f"目标点数: {target_points}")
    print("=" * 60)

    # 逐步调整容差
    iteration = 0
    while len(best_coords) > target_points and iteration < 20:
        iteration += 1
        simplified = douglas_peucker_simplify(coords, tolerance)
        count = len(simplified)

        print(f" 迭代 {iteration}: 容差={tolerance:.6f}, 点数={count}")

        if count < target_points:
            # 过度简化，减小容差
            tolerance *= 0.5
        elif count == prev_count:
            # 无法继续简化，退出
            break
        else:
            # 继续简化
            best_coords = simplified
            prev_count = count
            tolerance *= 1.5  # 增加容差以进一步简化

    print("=" * 60)
    print(f"最终点数: {len(best_coords)}")

    return best_coords


def calculate_boundary_metrics(coords: List[Tuple[float, float]]) -> dict:
    """计算边界相关指标"""
    line = LineString(coords)

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    return {
        "point_count": len(coords),
        "total_length": round(line.length, 4),
        "lon_range": [round(min(lons), 6), round(max(lons), 6)],
        "lat_range": [round(min(lats), 6), round(max(lats), 6)],
        "avg_point_distance": round(line.length / max(1, len(coords) - 1), 6),
    }


def save_boundary(coords: List[Tuple[float, float]], output_dir: Path, suffix: str = ""):
    """保存边界到多种格式"""
    # JSON格式
    json_path = output_dir / f"hsq_boundary{suffix}.json"
    metrics = calculate_boundary_metrics(coords)

    boundary_data = {
        "name": f"洪山区边界{suffix}",
        "coordinates": [[lon, lat] for lon, lat in coords],
        **metrics
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(boundary_data, f, ensure_ascii=False, indent=2)

    print(f"JSON保存: {json_path}")

    # 高德格式
    amap_path = output_dir / f"hsq_boundary{suffix}_amap.txt"
    amap_coords = "|".join([f"{lon},{lat}" for lon, lat in coords])
    with open(amap_path, "w", encoding="utf-8") as f:
        f.write(amap_coords)

    print(f"高德格式保存: {amap_path} ({len(amap_coords)} 字符)")

    # Shapefile格式
    from geopandas import GeoDataFrame
    shp_path = output_dir / f"hsq_boundary{suffix}.shp"

    polygon = Polygon(coords)
    gdf = GeoDataFrame(
        {"name": [f"洪山区边界{suffix}"], "point_count": [len(coords)]},
        geometry=[polygon],
        crs="EPSG:4326"
    )
    gdf.to_file(shp_path)

    print(f"Shapefile保存: {shp_path}")

    return json_path, amap_path, shp_path


def main(tolerance: float = None, target_points: int = 400):
    """主函数"""
    # 读取原始边界
    input_path = Path(__file__).parent.parent / "data" / "hsq_outer_boundary.json"
    output_dir = Path(__file__).parent.parent / "data"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    coords = [(c[0], c[1]) for c in data["coordinates"]]
    print(f"\n原始边界信息:")
    print(f"  点数: {len(coords)}")
    print(f"  经度范围: {data['lon_range']}")
    print(f"  纬度范围: {data['lat_range']}")

    # 简化边界
    print("\n开始简化...")
    if tolerance:
        # 使用指定容差
        simplified = douglas_peucker_simplify(coords, tolerance)
    else:
        # 自适应简化到目标点数
        simplified = adaptive_simplify(coords, target_points)

    # 显示简化后的信息
    metrics = calculate_boundary_metrics(simplified)
    print(f"\n简化后边界信息:")
    print(f"  点数: {metrics['point_count']}")
    print(f"  边界长度: {metrics['total_length']}")
    print(f"  经度范围: {metrics['lon_range']}")
    print(f"  纬度范围: {metrics['lat_range']}")
    print(f"  平均点间距: {metrics['avg_point_distance']}")

    # 保存结果
    print("\n保存文件...")
    suffix = f"_simplified_{metrics['point_count']}"
    save_boundary(simplified, output_dir, suffix)

    # 生成多个版本供选择
    print("\n" + "=" * 60)
    print("生成多个简化版本供选择:")
    print("=" * 60)

    versions = [
        ("_s200", 200),   # 最简版
        ("_s300", 300),   # 中简版
        ("_s400", 400),   # 适简版
        ("_s500", 500),   # 较详细版
    ]

    for suffix_name, target in versions:
        print(f"\n目标 {target} 点:")
        version_coords = adaptive_simplify(coords, target)
        version_metrics = calculate_boundary_metrics(version_coords)
        print(f"  实际点数: {version_metrics['point_count']}")
        save_boundary(version_coords, output_dir, suffix_name)

    print("\n完成！")

    return simplified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简化洪山区边界坐标")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="简化容差（经纬度单位），不指定则自适应"
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=400,
        help="目标点数量（自适应模式下使用）"
    )

    args = parser.parse_args()
    main(tolerance=args.tolerance, target_points=args.target_points)