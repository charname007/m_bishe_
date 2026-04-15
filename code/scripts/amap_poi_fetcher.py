"""
高德地图POI数据获取脚本（支持动态划分搜索）
获取洪山区的POI数据，补充到 geo_entity_names 表

核心功能：
- 多边形搜索：使用边界坐标搜索区域内的POI
- 动态划分：当结果达到上限(500)时，自动将区域4等分递归搜索
- 断点续传：进度保存，中断后可继续

使用方式：
    python scripts/amap_poi_fetcher.py
    python scripts/amap_poi_fetcher.py --types 050000,060000
    python scripts/amap_poi_fetcher.py --boundary data/hsq_boundary_s200_amap.txt
"""
import os
import sys
import time
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode
import urllib.request
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
import psycopg2
from psycopg2.extras import execute_values


# ============================================================
# 高德POI分类码（大类）
# ============================================================

AMAP_POI_TYPES = {
    "050000": "餐饮服务",
    "060000": "购物服务",
    "070000": "生活服务",
    "080000": "体育休闲服务",
    "090000": "医疗保健服务",
    "100000": "住宿服务",
    "110000": "风景名胜",
    "120000": "商务住宅",
    "130000": "政府机构及社会团体",
    "140000": "科教文化服务",
    "150000": "交通设施服务",
    "160000": "金融保险服务",
    "170000": "公司企业",
}

# API限制
MAX_POI_PER_QUERY = 500    # 单次查询最大返回数
MAX_PAGE_SIZE = 25         # 每页最大返回数
REQUEST_DELAY = 0.5        # 请求间隔（秒），增加以避免频率限制
MAX_RECURSION_DEPTH = 6    # 最大递归深度


class AmapPOIFetcher:
    """高德地图POI数据获取器（支持动态划分搜索）"""

    API_BASE_POLYGON = "https://restapi.amap.com/v3/place/polygon"
    API_BASE_TEXT = "https://restapi.amap.com/v3/place/text"

    def __init__(
        self,
        api_key: str,
        boundary_path: str = None,
        types: Optional[List[str]] = None
    ):
        """
        初始化POI获取器

        Args:
            api_key: 高德API Key
            boundary_path: 边界坐标文件路径（高德格式）
            types: 要获取的POI分类码列表
        """
        self.api_key = api_key
        self.types = types or list(AMAP_POI_TYPES.keys())

        # 加载边界坐标
        self.boundary_coords = None
        self.boundary_polygon_str = None
        if boundary_path:
            self._load_boundary(boundary_path)
        else:
            # 使用默认边界文件
            default_boundary = Path(__file__).parent.parent / "data" / "hsq_boundary_s200_amap.txt"
            if default_boundary.exists():
                self._load_boundary(str(default_boundary))

        # 计算边界范围（用于动态划分）
        self.boundary_bounds = None
        if self.boundary_coords:
            self._calculate_bounds()

        # 进度跟踪
        self.progress_file = Path(__file__).parent.parent / "amap_progress.json"
        self.progress = self._load_progress()

        # 统计
        self.total_fetched = 0
        self.total_saved = 0
        self.api_calls = 0

    def _load_boundary(self, path: str):
        """加载边界坐标文件"""
        with open(path, "r", encoding="utf-8") as f:
            self.boundary_polygon_str = f.read().strip()

        # 解析坐标列表
        coords = []
        for pair in self.boundary_polygon_str.split("|"):
            if "," in pair:
                lon, lat = pair.split(",")
                coords.append((float(lon), float(lat)))

        self.boundary_coords = coords
        logger.info(f"加载边界: {len(coords)} 个坐标点")

    def _calculate_bounds(self):
        """计算边界范围"""
        lons = [c[0] for c in self.boundary_coords]
        lats = [c[1] for c in self.boundary_coords]
        self.boundary_bounds = {
            "min_lng": min(lons),
            "max_lng": max(lons),
            "min_lat": min(lats),
            "max_lat": max(lats),
            "center_lng": (min(lons) + max(lons)) / 2,
            "center_lat": (min(lats) + max(lats)) / 2,
        }
        logger.info(f"边界范围: lng[{self.boundary_bounds['min_lng']:.4f}-{self.boundary_bounds['max_lng']:.4f}], "
                   f"lat[{self.boundary_bounds['min_lat']:.4f}-{self.boundary_bounds['max_lat']:.4f}]")

    def _load_progress(self) -> Dict:
        """加载进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"completed_types": {}, "last_update": None}

    def _save_progress(self):
        """保存进度"""
        self.progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=2)

    def _fetch_api(self, url: str, params: dict) -> Tuple[List[Dict], int]:
        """调用API并返回结果"""
        full_url = f"{url}?{urlencode(params)}"

        try:
            with urllib.request.urlopen(full_url, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                self.api_calls += 1

                if data.get("status") != "1":
                    logger.error(f"API错误: {data.get('info', '未知')} - {params}")
                    return [], 0

                pois = data.get("pois", [])
                count = int(data.get("count", 0))
                return pois, count

        except Exception as e:
            logger.error(f"请求错误: {e}")
            return [], 0

    def _search_polygon(self, polygon_coords: str, typecode: str) -> Tuple[List[Dict], int]:
        """
        多边形区域搜索

        Args:
            polygon_coords: 多边形坐标字符串 "lng,lat|lng,lat|..."
            typecode: POI分类码

        Returns:
            (pois列表, 总数)
        """
        params = {
            "key": self.api_key,
            "polygon": polygon_coords,
            "offset": MAX_PAGE_SIZE,
            "page": 1,
            "extensions": "all",
        }
        # 只有指定了typecode才添加types参数
        if typecode:
            params["types"] = typecode

        all_pois = []
        total_count = 0

        # 分页获取
        for page in range(1, 40):  # 最大约1000条
            params["page"] = page
            pois, count = self._fetch_api(self.API_BASE_POLYGON, params)

            logger.debug(f"    page={page}: 返回{len(pois)}条, count={count}")

            if not pois:
                break

            all_pois.extend(pois)
            total_count = max(total_count, count)

            if page * MAX_PAGE_SIZE >= count:
                break

            time.sleep(REQUEST_DELAY)

        return all_pois, total_count

    def _search_rect_recursive(
        self,
        min_lng: float, max_lng: float,
        min_lat: float, max_lat: float,
        typecode: str,
        depth: int = 0
    ) -> List[Dict]:
        """
        动态划分搜索：如果结果达到上限，将矩形4等分递归搜索

        Args:
            min_lng, max_lng, min_lat, max_lat: 矩形边界
            typecode: POI分类码
            depth: 当前递归深度

        Returns:
            POI列表
        """
        if depth > MAX_RECURSION_DEPTH:
            logger.warning(f"达到最大递归深度 {MAX_RECURSION_DEPTH}，停止划分")
            return []

        # 构建矩形坐标字符串（4个角点 + 闭合）
        coords_str = f"{min_lng},{min_lat}|{max_lng},{min_lat}|{max_lng},{max_lat}|{min_lng},{max_lat}|{min_lng},{min_lat}"

        # 搜索该区域
        pois, count = self._search_polygon(coords_str, typecode)

        logger.debug(f"  [depth={depth}] lng[{min_lng:.4f}-{max_lng:.4f}] lat[{min_lat:.4f}-{max_lat:.4f}] -> {len(pois)}/{count} 条")

        # 关键判断：如果API显示的总数远大于实际获取量，说明有遗漏，需要划分
        # 高德polygon API限制约200条/查询，所以当count>200且count>实际获取量时划分
        if count > 200 and count > len(pois):
            logger.info(f"  结果被截断(count={count}>实际={len(pois)})，划分区域...")

            # 计算中心点
            mid_lng = (min_lng + max_lng) / 2
            mid_lat = (min_lat + max_lat) / 2

            # 4等分递归搜索
            all_pois = []

            # 左下
            all_pois.extend(self._search_rect_recursive(
                min_lng, mid_lng, min_lat, mid_lat, typecode, depth + 1
            ))

            # 右下
            all_pois.extend(self._search_rect_recursive(
                mid_lng, max_lng, min_lat, mid_lat, typecode, depth + 1
            ))

            # 左上
            all_pois.extend(self._search_rect_recursive(
                min_lng, mid_lng, mid_lat, max_lat, typecode, depth + 1
            ))

            # 右上
            all_pois.extend(self._search_rect_recursive(
                mid_lng, max_lng, mid_lat, max_lat, typecode, depth + 1
            ))

            return all_pois
        else:
            return pois

    def fetch_pois_by_type(self, typecode: str = None) -> List[Dict]:
        """
        按分类获取POI（使用动态划分搜索）

        Args:
            typecode: POI分类码，None表示不分类获取全部

        Returns:
            POI列表
        """
        type_name = AMAP_POI_TYPES.get(typecode, "全部POI") if typecode else "全部POI"
        logger.info(f"[{type_name}] 开始获取...")

        # 检查是否已完成
        progress_key = typecode or "all"
        if progress_key in self.progress.get("completed_types", {}):
            logger.info(f"[{type_name}] 已完成，跳过")
            return []

        all_pois = []

        # 使用动态划分搜索
        if self.boundary_bounds:
            logger.info(f"  使用动态划分搜索...")
            all_pois = self._search_rect_recursive(
                self.boundary_bounds["min_lng"],
                self.boundary_bounds["max_lng"],
                self.boundary_bounds["min_lat"],
                self.boundary_bounds["max_lat"],
                typecode,  # 传入None表示不分类
                depth=0
            )
        elif self.boundary_polygon_str:
            # 直接使用边界坐标搜索（可能无法处理超500的情况）
            logger.info(f"  使用边界多边形搜索...")
            all_pois, _ = self._search_polygon(self.boundary_polygon_str, typecode)
        else:
            logger.warning("无边界数据，跳过")
            return []

        # 去重（按location坐标）
        unique_pois = {}
        for p in all_pois:
            loc = p.get("location")
            if loc and loc not in unique_pois:
                unique_pois[loc] = p

        unique_list = list(unique_pois.values())
        self.total_fetched += len(unique_list)

        # 标记完成
        self.progress["completed_types"][progress_key] = len(unique_list)
        self._save_progress()

        logger.success(f"[{type_name}] 完成: {len(all_pois)}条(原始) -> {len(unique_list)}条(去重)")
        return unique_list

    def transform_poi_to_entity(self, poi: Dict) -> Dict:
        """将高德POI转换为entity格式"""
        poi_id = poi.get("id", "")
        entity_id = f"amap_{poi_id}"

        name = poi.get("name", "").strip()
        typecode = poi.get("typecode", "")
        # 从typecode获取大类名称，如果没有则使用高德返回的type字段
        if typecode and len(typecode) >= 2:
            type_name = AMAP_POI_TYPES.get(typecode[:2] + "0000", poi.get("type", "未知"))
        else:
            type_name = poi.get("type", "未知")

        # 截断type以适应数据库字段长度(VARCHAR 50)
        type_name = type_name[:50] if type_name else "未知"

        # 解析坐标
        location = poi.get("location", "")
        longitude, latitude = None, None
        if location and "," in location:
            try:
                parts = location.split(",")
                longitude = float(parts[0])
                latitude = float(parts[1])
            except:
                pass

        return {
            "entity_id": entity_id,
            "name": name[:200] if name else "未命名",
            "type": type_name,
            "longitude": longitude,
            "latitude": latitude,
            "address": poi.get("address", ""),
        }

    def save_to_json(self, entities: List[Dict], typecode: str) -> int:
        """保存到JSON文件（备份）"""
        if not entities:
            return 0

        # 备份文件路径
        backup_dir = Path(__file__).parent.parent / "data" / "amap_poi_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # 每个分类一个文件
        type_name = AMAP_POI_TYPES.get(typecode, "all") if typecode else "all"
        backup_file = backup_dir / f"amap_poi_{typecode or 'all'}.json"

        # 读取已有数据并合并
        existing = []
        if backup_file.exists():
            with open(backup_file, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # 合并去重
        all_data = {e["entity_id"]: e for e in existing}
        for e in entities:
            all_data[e["entity_id"]] = e

        # 保存
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(list(all_data.values()), f, ensure_ascii=False, indent=2)

        logger.info(f"  JSON备份: {backup_file} ({len(all_data)} 条)")
        return len(entities)

    def save_to_postgres(self, entities: List[Dict], conn) -> int:
        """批量保存到PostgreSQL (amap_entity_names 表)"""
        if not entities:
            return 0

        data = [
            (e["entity_id"], e["name"], e["type"], e.get("longitude"), e.get("latitude"), e.get("address", ""))
            for e in entities
            if e["name"] and e["name"] != "未命名"
        ]

        if not data:
            return 0

        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO amap_entity_names (entity_id, name, type, longitude, latitude, address)
                VALUES %s
                ON CONFLICT (entity_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    type = EXCLUDED.type,
                    longitude = EXCLUDED.longitude,
                    latitude = EXCLUDED.latitude,
                    address = EXCLUDED.address
                """,
                data,
                template="(%s, %s, %s, %s, %s, %s)"
            )
            conn.commit()

        self.total_saved += len(data)
        return len(data)

    def fetch_all_types(self, save_to_db: bool = True) -> Dict:
        """遍历所有分类获取POI"""
        logger.info("=" * 60)
        logger.info("洪山区POI获取（动态划分搜索）")
        logger.info("=" * 60)

        conn = None
        if save_to_db:
            pg_config = settings.get_postgres_config()
            conn = psycopg2.connect(
                host=pg_config['host'],
                port=pg_config['port'],
                database=pg_config['database'],
                user=pg_config['user'],
                password=pg_config['password']
            )

        try:
            for typecode in self.types:
                pois = self.fetch_pois_by_type(typecode)

                if pois:
                    entities = [self.transform_poi_to_entity(p) for p in pois]
                    # 先保存到JSON备份
                    self.save_to_json(entities, typecode)
                    # 再保存到数据库
                    if save_to_db and conn:
                        saved = self.save_to_postgres(entities, conn)
                        logger.info(f"  保存到数据库: {saved} 条")

        finally:
            if conn:
                conn.close()

        # 清理进度文件
        if self.progress_file.exists():
            self.progress_file.unlink()

        stats = {
            "total_fetched": self.total_fetched,
            "total_saved": self.total_saved,
            "api_calls": self.api_calls,
            "types_processed": len(self.types),
        }

        logger.success("=" * 60)
        logger.success("获取完成!")
        logger.success(f"  POI总数: {stats['total_fetched']}")
        logger.success(f"  数据库保存: {stats['total_saved']}")
        logger.success(f"  API调用次数: {stats['api_calls']}")

        return stats


def main(types: Optional[str] = None, boundary: Optional[str] = None, no_type: bool = False):
    """主函数"""
    amap_config = settings.get_amap_config()
    api_key = amap_config["api_key"]

    # 解析types
    type_list = None
    if no_type:
        # 不分类，获取全部POI
        type_list = [None]  # None表示不指定types参数
        logger.info("模式: 不分类，获取全部POI")
    elif types:
        type_list = [t.strip() for t in types.split(",")]

    # 创建获取器
    fetcher = AmapPOIFetcher(
        api_key=api_key,
        boundary_path=boundary,
        types=type_list
    )

    # 执行获取
    stats = fetcher.fetch_all_types()
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高德POI获取（动态划分搜索）")
    parser.add_argument("--types", default=None, help="POI分类码，如: 050000,060000。不指定则获取全部类型")
    parser.add_argument("--boundary", default=None, help="边界坐标文件路径")
    parser.add_argument("--no-type", action="store_true", help="不分类，直接获取全部POI（推荐）")

    args = parser.parse_args()
    main(types=args.types, boundary=args.boundary, no_type=args.no_type)