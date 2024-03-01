import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch
sys.path.append('/home/classlab2/root/OrienterNet')
from maploc.demo import Demo, read_input_image
# demo = Demo(num_rotations=256, device='cpu')
# Query OpenStreetMap for this area
import numpy as np
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap, plot_nodes
from maploc.utils.viz_2d import plot_images
from maploc.utils.geo import BoundaryBox, Projection
from maploc.osm.download import get_osm
from maploc.osm.raster import Canvas, render_raster_map, render_raster_masks
from maploc.osm.reader import OSMData, OSMNode, OSMWay
import os
from maploc.data.boreas.utils import parse_gps_file
tile_manager = TileManager.load(Path("/home/classlab2/16T/datasets/boreas./boreastiles.pkl"))

output_dir = "/home/classlab2/16T/datasets/boreas./raster2"
os.makedirs(output_dir, exist_ok=True)
# 查询 TileManager 并打印每个 canvas
# 创建一个子图
fig, axes = plt.subplots()

# osm = OSMData.from_file(path)#读osm
# osm.add_xy_to_nodes()


import pyproj
import geopandas as gpd
import folium
import osmnx as ox

osm_file = Path("/home/classlab2/root/OrienterNet/datasets/OSM/map8.osm")
data_dir = Path("/home/classlab2/16T/datasets/boreas.")
all_latlon = []
for gps_path in data_dir.glob("*/gps/*.txt"):#boreas的gps数据需要一次性解析多行
    all_latlon.append(parse_gps_file(gps_path)[0])
if not all_latlon:
    raise ValueError(f"Cannot find any GPS file in {data_dir}.")
all_latlon = np.stack(all_latlon)
projection = Projection.from_points(all_latlon)
for i, (ij, tile) in enumerate(tile_manager.tiles.items()):
    # 获取感兴趣区域在当前瓦片中的 bbox_tile
    bbox_tile = tile.bbox
    bbox_osm = projection.unproject(bbox_tile)
    north, south, east, west = 37.9, 37.8, -122.1, -122.3
    # 获取OSM数据
    graph = ox.graph_from_xml(osm_file, simplify=False)
    # 定义经纬度范围
    north, south, east, west = 43.788671542038486, 43.78751965517837, -79.462288258427, -79.46387869540725

    # 获取指定范围内的子图
    subgraph = ox.graph_from_bbox(graph, north, south, east, west)

    # 保存为OSM文件
    ox.save_graph_xml(subgraph, filename='output.osm')

    #bbox_osm已经转成经纬度的
    # 使用pyproj进行投影转换
    osm_data = gpd.read_file(osm_file, bbox=bbox_osm)
    #graph_from_bbox
    utm_projection = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32633', always_xy=True)
    osm_data['utm_geometry'] = osm_data['geometry'].to_crs('EPSG:32633')
    # 根据投影后的点、线段信息，可视化图像
    map_center = [bbox_osm[1], bbox_osm[0]]  # 注意：经纬度的顺序可能需要调整
    my_map = folium.Map(location=map_center, zoom_start=12)
    # 添加投影后的节点
    for idx, row in osm_data.iterrows():
        utm_coords = row['utm_geometry'].coords.xy
        folium.Marker(location=[utm_coords[1][0], utm_coords[0][0]]).add_to(my_map)
    my_map.save(f'map_{i}.html')

plt.close()


