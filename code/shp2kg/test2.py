import geopandas as gpd
import numpy as np
from scipy.spatial import Delaunay
from neo4j import GraphDatabase
import math
import pandas as pd
from shapely.geometry import Point

from settings import settings

def shp_to_kg(shp_file, neo4j_uri, neo4j_user, neo4j_password):   
    # 读取Shapefile文件
    gdf = gpd.read_file(shp_file)

    # 创建Neo4j驱动
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    with driver.session() as session:
        for idx, row in gdf.iterrows():
            geometry = row['geometry']
            properties = row.drop('geometry').to_dict()

            if geometry.geom_type == 'Polygon':
                exterior_coords = list(geometry.exterior.coords)
                interior_coords = [list(interior.coords) for interior in geometry.interiors]

                # 创建节点
                session.run("CREATE (n:Polygon {properties: $properties})", properties=properties)

                # 创建关系
                for i in range(len(exterior_coords) - 1):
                    session.run("""
                        MATCH (n:Polygon {properties: $properties})
                        CREATE (n)-[:HAS_EDGE]->(:Edge {start: $start, end: $end})
                    """, properties=properties, start=exterior_coords[i], end=exterior_coords[i + 1])

                for interior in interior_coords:
                    for i in range(len(interior) - 1):
                        session.run("""
                            MATCH (n:Polygon {properties: $properties})
                            CREATE (n)-[:HAS_HOLE]->(:Hole {start: $start, end: $end})
                        """, properties=properties, start=interior[i], end=interior[i + 1])

    driver.close()
    
    

def extract_road_keypoints(road_gdf):
    """提取道路的关键节点（道路端点和交叉点）
    """    

    # 提取道路端点 LineString 的第一个点和最后一个点就是端点。
    endpoints = []
    
    for geom in road_gdf.geometry:

        if geom.geom_type == "LineString":
           coords = list(geom.coords)
           endpoints.append(Point(coords[0]))
           endpoints.append(Point(coords[-1]))
        elif geom.geom_type == "MultiLineString":
             for line in geom:
                coords = list(line.coords)
                endpoints.append(Point(coords[0]))
                endpoints.append(Point(coords[-1]))
    
    # 计算道路交叉点 交叉点就是两条线的 intersection。
    intersections = []
    sindex = roads.sindex
    roads_list = list(roads.iterrows())
    for i, road1 in roads.iterrows():
        possible = list(sindex.intersection(road1.geometry.bounds))
        for j in possible:
            if int(j) <= int(i):
                continue
            road2 = roads.iloc[j]
            inter = road1.geometry.intersection(road2.geometry)
            if not inter.is_empty:
                if inter.geom_type == "Point":
                    intersections.append(inter)
    
    
    key_nodes = endpoints + intersections
    
    return key_nodes


if __name__ == "__main__":
    
    # 将坐标系转为投影坐标系(如 EPSG:3857)，这样计算的距离单位是“米”而不是经纬度

    roads = gpd.read_file(settings.SHPFILES_DIR +
                          '/roads.shp').to_crs(epsg=3857)
    pois = gpd.read_file(settings.SHPFILES_DIR +
                           '/pois.shp').to_crs(epsg=3857)
    blocks = gpd.read_file(settings.SHPFILES_DIR +
                            '/blocks.shp').to_crs(epsg=3857)
    buildings = gpd.read_file(settings.SHPFILES_DIR 
                              +'/buildings.shp').to_crs(epsg=3857)
    
    #提取实体的点
    
    # 道路：只保留关键节点（道路端点 道路交叉点）
    
    # 建筑：中心点
    
    # 街区：边界点（可simplify(10)）


