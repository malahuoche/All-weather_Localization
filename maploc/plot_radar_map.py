import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import numpy as np
import sys
import torch
sys.path.append('/home/classlab2/root/OrienterNet')
from maploc.utils.geo import BoundaryBox, Projection
from maploc.data.boreas.utils import parse_gps_file
from projectosm import OSMDataLoader
from radar2UTM import polar_to_cartesian,probabilistic_extract_points_from_scan,visualize_pointcloud,points_local_to_utm,extract_pointcloud,extract_local_coordinates
from PIL import Image
fig, axes = plt.subplots()
data_dir = Path("/home/classlab2/16T/datasets/boreas.")
all_latlon = []
for gps_path in data_dir.glob("*/gps/*.txt"):#boreas的gps数据需要一次性解析多行
    all_latlon.append(parse_gps_file(gps_path)[0])
if not all_latlon:
    raise ValueError(f"Cannot find any GPS file in {data_dir}.")
all_latlon = np.stack(all_latlon)
projection = Projection.from_points(all_latlon)
def bbox_to_boundary_box(bbox):
    # 假设 bbox 是一个表示边界框的对象，比如 [xmin, ymin, xmax, ymax]
    min_coords = np.array([bbox[0], bbox[1]])
    max_coords = np.array([bbox[2], bbox[3]])
    # 创建 BoundaryBox 实例
    boundary_box = BoundaryBox(min_coords, max_coords)
    return boundary_box
def read_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # 跳过空行
                # 按空格分割每行的值
                values = line.strip().split()
                # 前四个数字作为 bbox_tile，后面的是图片地址
                bbox_tile = [int(coord) for coord in values[:4]]
                image_path = values[4]
                # 将数据组合成元组并添加到列表中
                data.append((bbox_tile, image_path))
    return data
def rotate_points(points, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    rotated_points = np.dot(rotation_matrix, points)
    return rotated_points


bbox_coords = [-761.966773833859,-132.68964418849856,-633.966773833859,-4.689644188498562   ]
bbox_tile = bbox_to_boundary_box(bbox_coords)
bbox_osm = projection.unproject(bbox_tile)
osm_file = "/home/classlab2/root/OrienterNet/datasets/OSM/map8.osm"
loader = OSMDataLoader(osm_file)
min_lat, min_lon, max_lat, max_lon = bbox_osm.min_[0], bbox_osm.min_[1], bbox_osm.max_[0], bbox_osm.max_[1]
nodes, ways = loader.get_nodes_and_ways(min_lat, max_lat, min_lon, max_lon)
loader.plot_map_utm(nodes, ways)
image_path = "/home/classlab2/16T/datasets/boreas./boreas-2021-03-30-14-23/radar/cart/1617129015175467.png"
polar_path = image_path.replace("/cart", "")
cartesian_image_path = image_path.replace("/cart", "/cart_orignal")
resolution = 0.2384  # m/pixel

cartesian_image = cv2.imread(cartesian_image_path)
polar_image = cv2.imread(polar_path)
radar_resolution =0.0596
# point_cloud = extract_pointcloud(polar_image,radar_resolution)
resolution = 0.2384  # m/pixel
# 使用提供的函数提取亮点的本地坐标
point_cloud_local = extract_local_coordinates(cartesian_image, resolution)
# visualize_pointcloud(point_cloud_local)
points_local = point_cloud_local.T

visualize_pointcloud(point_cloud_local)
gps_list = [(43.79113757285169,-79.48178680532114), (43.79114580475715,-79.4818193192494)]
# from pyproj import Proj, transform
# wgs84 = Proj(init='epsg:4326')  # EPSG code 4326 corresponds to WGS 84
# utm = Proj(init='epsg:32617')  #
# # 将GPS坐标转换为UTM坐标
gps_utm = [622148.19, 4849795.49]
# gps_utm = [622148.27, 4849795.38]
yaw = -1.2711586863589217

tInd = 1  # 时间步索引
# points_local = point_cloud

# 调用函数进行坐标转换
points_utm = points_local_to_utm(gps_list, tInd, points_local)

# # 输出转换后的 UTM 坐标
# print("转换后的 UTM 坐标：", points_utm)
# print(points_utm.shape)

utm_x = points_utm[0, :]
utm_y = points_utm[1, :]
# 创建散点图
# plt.scatter(utm_x, utm_y, color='blue', label='UTM Coordinates',s=2)
plt.scatter(gps_utm[0],gps_utm[1],s=10,color = "green")
plt.xlim(min(utm_x), max(utm_x))
plt.ylim(min(utm_y), max(utm_y))
# 设置图形标题和标签
plt.title(f'UTM Coordinates at Time Step {tInd}')
plt.xlabel('UTM X')
plt.ylabel('UTM Y')
plt.legend()

plt.show()
# Convert gps_utm to a NumPy array before reshaping
gps_utm_array = np.array(gps_utm)

# Calculate coordinate difference, assuming the rotation center is gps_utm
coord_diff = np.array(points_utm) - gps_utm_array.reshape(-1, 1)

rotated_coord_diff = rotate_points(coord_diff, yaw)
# 将旋转后的坐标差异加回到原始雷达点云的UTM坐标上
points_utm_rotated = gps_utm_array.reshape(-1, 1) + rotated_coord_diff
print(points_utm_rotated)

image_size = 256
# UTM坐标转换到图像坐标的简化函数

plt.scatter(points_utm_rotated[0, :], points_utm_rotated[1, :], color='green', label='rotated_UTM', s=5)
plt.legend()
plt.show()
plt.savefig("utm1215.png")
# plt.close()
# # 遍历调用
# txt_file_path = "/home/classlab2/root/OrienterNet/mapandradar.txt"
# # 读取数据
# data = read_txt_file(txt_file_path)
# for bbox_tile, image_path in data:
#     print(f"bbox_tile: {bbox_tile}, image_path: {image_path}")






