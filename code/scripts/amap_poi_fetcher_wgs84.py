"""
高德地图POI数据获取脚本（正确坐标系版本）
使用完整洪山区边界，坐标转换：输入GCJ-02，输出WGS-84

核心功能：
- 边界坐标转换：WGS-84 -> GCJ-02（用于API请求）
- POI坐标转换：GCJ-02 -> WGS-84（用于存储）
- 动态划分搜索：当结果达到上限时自动划分区域
- 保存到新表：amap_poi_wgs84

使用方式：
    python scripts/amap_poi_fetcher_wgs84.py
"""
import os
import sys
import time
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urlencode
import urllib.request
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import settings
import psycopg2
from psycopg2.extras import execute_values


# ============================================================
# 坐标转换函数
# ============================================================

def wgs84_to_gcj02(lng: float, lat: float) -> Tuple[float, float]:
    """WGS-84 转 GCJ-02（火星坐标系）"""
    a = 6378245.0  # 长半轴
    ee = 0.00669342162296594323  # 偏心率平方

    def transform_lat(lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
        return ret

    def transform_lng(lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 150.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
        return ret

    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)

    mglat = lat + dlat
    mglng = lng + dlng

    return mglng, mglat


def gcj02_to_wgs84(lng: float, lat: float) -> Tuple[float, float]:
    """GCJ-02 转 WGS-84"""
    a = 6378245.0
    ee = 0.00669342162296594323

    def transform_lat(lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
        return ret

    def transform_lng(lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 150.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
        return ret

    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)

    wgs_lat = lat - dlat
    wgs_lng = lng - dlng

    return wgs_lng, wgs_lat


# ============================================================
# 高德POI分类码
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

MAX_PAGE_SIZE = 25
REQUEST_DELAY = 0.5
MAX_RECURSION_DEPTH = 6


class AmapPOIFetcherWGS84:
    """高德地图POI获取器（带坐标转换）"""

    API_BASE = "https://restapi.amap.com/v3/place/polygon"

    def __init__(self, api_key: str, boundary_wgs_path: str, types: List[str] = None):
        self.api_key = api_key
        self.types = types or list(AMAP_POI_TYPES.keys())
        self.api_calls = 0
        self.total_saved = 0

        # 加载WGS-84边界并转换为GCJ-02
        self._load_boundary(boundary_wgs_path)

    def _load_boundary(self, path: str):
        """加载边界坐标（WGS-84）并转换为GCJ-02"""
        with open(path, "r", encoding="utf-8") as f:
            coords_str = f.read().strip()

        # 解析WGS-84坐标
        wgs_coords = []
        for pair in coords_str.split("|"):
            if "," in pair:
                lon, lat = pair.split(",")
                wgs_coords.append((float(lon), float(lat)))

        logger.info(f"WGS-84边界: {len(wgs_coords)} 个坐标点")

        wgs_lats = [c[1] for c in wgs_coords]
        logger.info(f"WGS-84纬度范围: {min(wgs_lats):.4f} - {max(wgs_lats):.4f}")

        # 转换为GCJ-02（用于API请求）
        gcj_coords = [wgs84_to_gcj02(lon, lat) for lon, lat in wgs_coords]

        gcj_lats = [c[1] for c in gcj_coords]
        logger.info(f"GCJ-02纬度范围: {min(gcj_lats):.4f} - {max(gcj_lats):.4f}")

        # 保存边界范围（用于动态划分）
        self.bounds_gcj = {
            "min_lng": min(c[0] for c in gcj_coords),
            "max_lng": max(c[0] for c in gcj_coords),
            "min_lat": min(c[1] for c in gcj_coords),
            "max_lat": max(c[1] for c in gcj_coords),
        }

    def _fetch_api(self, polygon: str, typecode: str, page: int) -> Tuple[List[Dict], int]:
        """调用API"""
        params = {
            "key": self.api_key,
            "polygon": polygon,
            "offset": MAX_PAGE_SIZE,
            "page": page,
            "extensions": "all",
        }
        if typecode:
            params["types"] = typecode

        url = f"{self.API_BASE}?{urlencode(params)}"
        self.api_calls += 1

        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("status") != "1":
                    logger.error(f"API错误: {data.get('info', '未知')}")
                    return [], 0
                return data.get("pois", []), int(data.get("count", 0))
        except Exception as e:
            logger.error(f"请求错误: {e}")
            return [], 0

    def _search_rect(self, min_lng, max_lng, min_lat, max_lat, typecode, depth=0) -> List[Dict]:
        """动态划分搜索"""
        if depth > MAX_RECURSION_DEPTH:
            return []

        # 构建矩形多边形
        polygon = f"{min_lng},{min_lat}|{max_lng},{min_lat}|{max_lng},{max_lat}|{min_lng},{max_lat}|{min_lng},{min_lat}"

        all_pois = []
        total_count = 0

        for page in range(1, 40):
            pois, count = self._fetch_api(polygon, typecode, page)
            if not pois:
                break
            all_pois.extend(pois)
            total_count = max(total_count, count)
            if page * MAX_PAGE_SIZE >= count:
                break
            time.sleep(REQUEST_DELAY)

        logger.debug(f"  depth={depth}: {len(all_pois)}/{total_count}")

        # 如果结果被截断，划分区域
        if total_count > 200 and total_count > len(all_pois):
            mid_lng = (min_lng + max_lng) / 2
            mid_lat = (min_lat + max_lat) / 2

            result = []
            result.extend(self._search_rect(min_lng, mid_lng, min_lat, mid_lat, typecode, depth + 1))
            result.extend(self._search_rect(mid_lng, max_lng, min_lat, mid_lat, typecode, depth + 1))
            result.extend(self._search_rect(min_lng, mid_lng, mid_lat, max_lat, typecode, depth + 1))
            result.extend(self._search_rect(mid_lng, max_lng, mid_lat, max_lat, typecode, depth + 1))
            return result

        return all_pois

    def fetch_pois_by_type(self, typecode: str) -> List[Dict]:
        """按分类获取POI"""
        type_name = AMAP_POI_TYPES.get(typecode, "未知")
        logger.info(f"[{type_name}] 开始获取...")

        pois = self._search_rect(
            self.bounds_gcj["min_lng"],
            self.bounds_gcj["max_lng"],
            self.bounds_gcj["min_lat"],
            self.bounds_gcj["max_lat"],
            typecode,
            depth=0
        )

        # 去重（按坐标）
        unique_pois = {}
        for p in pois:
            loc = p.get("location")
            if loc and loc not in unique_pois:
                unique_pois[loc] = p

        logger.success(f"[{type_name}] 完成: {len(unique_pois)}条")
        return list(unique_pois.values())

    def transform_poi(self, poi: Dict) -> Dict:
        """转换POI数据，坐标转回WGS-84"""
        poi_id = poi.get("id", "")
        name = poi.get("name", "").strip()
        typecode = poi.get("typecode", "")
        type_name = AMAP_POI_TYPES.get(typecode[:2] + "0000", poi.get("type", "未知")) if typecode else "未知"

        # 解析GCJ-02坐标并转换为WGS-84
        location = poi.get("location", "")
        longitude, latitude = None, None

        if location and "," in location:
            try:
                gcj_lon, gcj_lat = float(location.split(",")[0]), float(location.split(",")[1])
                longitude, latitude = gcj02_to_wgs84(gcj_lon, gcj_lat)
            except:
                pass

        return {
            "entity_id": f"amap_{poi_id}",
            "name": name[:200] if name else None,
            "type": type_name[:50],
            "longitude": longitude,
            "latitude": latitude,
            "address": poi.get("address", ""),
        }

    def create_table(self, conn):
        """创建新表"""
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS amap_poi_wgs84 (
                    id SERIAL PRIMARY KEY,
                    entity_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    type VARCHAR(50),
                    longitude DOUBLE PRECISION,
                    latitude DOUBLE PRECISION,
                    address TEXT,
                    embedding VECTOR(768),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_amap_poi_wgs84_type ON amap_poi_wgs84(type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_amap_poi_wgs84_location ON amap_poi_wgs84(longitude, latitude)")
            conn.commit()
        logger.info("表 amap_poi_wgs84 已创建")

    def save_to_db(self, entities: List[Dict], conn) -> int:
        """保存到数据库"""
        if not entities:
            return 0

        data = [
            (e["entity_id"], e["name"], e["type"], e["longitude"], e["latitude"], e["address"])
            for e in entities if e["name"]
        ]

        if not data:
            return 0

        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO amap_poi_wgs84 (entity_id, name, type, longitude, latitude, address)
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

    def run(self) -> Dict:
        """运行获取流程"""
        logger.info("=" * 60)
        logger.info("洪山区POI获取（正确坐标系版本）")
        logger.info("WGS-84边界 -> GCJ-02请求 -> WGS-84存储")
        logger.info("=" * 60)

        pg_config = settings.get_postgres_config()
        conn = psycopg2.connect(
            host=pg_config["host"], port=pg_config["port"],
            database=pg_config["database"], user=pg_config["user"], password=pg_config["password"]
        )

        try:
            self.create_table(conn)

            for typecode in self.types:
                pois = self.fetch_pois_by_type(typecode)

                if pois:
                    entities = [self.transform_poi(p) for p in pois]
                    saved = self.save_to_db(entities, conn)
                    logger.info(f"  保存: {saved} 条")

        finally:
            conn.close()

        logger.success("=" * 60)
        logger.success(f"完成! 总保存: {self.total_saved}, API调用: {self.api_calls}")

        return {"total_saved": self.total_saved, "api_calls": self.api_calls}


def main():
    amap_config = settings.get_amap_config()

    # 使用新的正确边界文件
    boundary_path = Path(__file__).parent.parent / "data" / "hsq_boundary_hull_amap.txt"

    fetcher = AmapPOIFetcherWGS84(
        api_key=amap_config["api_key"],
        boundary_wgs_path=str(boundary_path),
        types=list(AMAP_POI_TYPES.keys())
    )

    return fetcher.run()


if __name__ == "__main__":
    main()