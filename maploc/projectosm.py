# import osmium
# import pyproj
# import folium
# # from maploc.osm.reader import OSMData
# # # 给定的经纬度区域
# # min_lat, min_lon = 40.0, -74.0
# # max_lat, max_lon = 41.0, -73.0
# # import matplotlib.pyplot as plt
# # # 创建投影转换器
# # projection = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32618', always_xy=True)
# # osm_file = Path("/home/classlab2/root/OrienterNet/datasets/OSM/map8.osm")
# # osm = OSMData.from_file(osm_file)

# import osmium
# import pyproj

# # 创建投影转换器
# projection = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32618', always_xy=True)

# import xml.etree.ElementTree as ET
# from xmltodict import parse
# from pyproj import Proj, transform

# # 解析OSM文件
# def parse_osm_file(osm_file):
#     with open(osm_file, 'r', encoding='utf-8') as file:
#         osm_data = parse(file.read())
#         # print(osm_data)
#     return osm_data

# # 从OSM数据中获取指定经纬度范围的节点和线
# def get_nodes_and_ways(osm_data, min_lat, max_lat, min_lon, max_lon):
#     nodes = {}
#     ways = {}

#     for element in osm_data['osm']['node']:
#         lat = float(element['@lat'])
#         lon = float(element['@lon'])
#         if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
#             nodes[element['@id']] = (lat, lon)

#     for element in osm_data['osm']['way']:
#         way_id = element['@id']
#         way_nodes = []
#         if 'nd' in element:
#             for nd in element['nd']:
#                 node_id = nd['@ref']
#                 if node_id in nodes:
#                     way_nodes.append(nodes[node_id])
#             if way_nodes:
#                 ways[way_id] = way_nodes

#     return nodes, ways

# # 将经纬度坐标投影到UTM坐标系
# def project_to_utm(lat, lon):
#     # WGS84坐标系
#     wgs84 = Proj('epsg:4326')

#     # UTM坐标系，这里假设使用UTM Zone 33N
#     utm = Proj('epsg:32633')

#     # 将经纬度坐标投影到UTM坐标系
#     x, y = transform(wgs84, utm, lon, lat)
#     return x, y

# if __name__ == "__main__":
#     osm_file = "/home/classlab2/root/OrienterNet/datasets/OSM/map8.osm"
#     min_lat, max_lat = 30, 50.0
#     min_lon, max_lon = -90.0, -74.0

#     osm_data = parse_osm_file(osm_file)
#     nodes, ways = get_nodes_and_ways(osm_data, min_lat, max_lat, min_lon, max_lon)

#     print("Projected Coordinates (UTM):")
#     for node_id, (lat, lon) in nodes.items():
#         utm_x, utm_y = project_to_utm(lat, lon)
#         print(f"Node {node_id}: {utm_x}, {utm_y}")

#     for way_id, way_nodes in ways.items():
#         print(f"Way {way_id}:")
#         for node in way_nodes:
#             utm_x, utm_y = project_to_utm(node[0], node[1])
#             print(f"  {utm_x}, {utm_y}")

import xml.etree.ElementTree as ET
from xmltodict import parse
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import numpy as np
import utm
class OSMDataLoader:
    def __init__(self, osm_file):
        self.osm_file = osm_file
        self.osm_data = self.parse_osm_file()

    def parse_osm_file(self):
        with open(self.osm_file, 'r', encoding='utf-8') as file:
            osm_data = parse(file.read())
        return osm_data

    def get_nodes_and_ways(self, min_lat, max_lat, min_lon, max_lon):
        nodes = {}
        ways = {}

        for element in self.osm_data['osm']['node']:
            lat = float(element['@lat'])
            lon = float(element['@lon'])
            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                nodes[element['@id']] = (lat, lon)

        for element in self.osm_data['osm']['way']:
            way_id = element['@id']
            way_nodes = []
            if 'nd' in element:
                way_nodes = [nodes[nd['@ref']] for nd in element['nd'] if nd['@ref'] in nodes]
                if way_nodes:
                    ways[way_id] = way_nodes

        return nodes, ways

    def project_to_utm(self, lat, lon):
        x, y, zone_number, zone_letter=utm.from_latlon(lat,lon)
        return x, y
    def plot_map_utm(self, nodes, ways):
        # 提取 Nodes 的 UTM 坐标
        plt.figure()
        node_utm_coords = [self.project_to_utm(lat, lon) for (lat, lon) in nodes.values()]
        utm_x, utm_y = zip(*node_utm_coords)
        print(min(utm_x))
        print(max(utm_x))
        print(min(utm_y))
        print(max(utm_y))
        fig, ax = plt.subplots()
        # 取消科学计数法显示
        ax.ticklabel_format(useOffset=False, style='plain')
        # 提取 Ways 的 UTM 坐标
        way_utm_coords = [[self.project_to_utm(lat, lon) for (lat, lon) in way_nodes] for way_nodes in ways.values()]
        # 创建散点图（Nodes）
        plt.scatter(np.array(utm_x), np.array(utm_y) , color='red', label='Nodes')
        # print(utm_x)
        plt.ticklabel_format(style='plain', axis='both')
        # yticks = np.arange(-8964450, -8964200, 50)  
        # xticks = np.arange(1064705.2784359353, 1064799.748731176, 50)      
        plt.axis('equal')

        # 创建折线图（Ways）
        for way_coords in way_utm_coords:
            way_utm_x, way_utm_y = zip(*way_coords)
            plt.plot(np.array(way_utm_x) , np.array(way_utm_y) , color='blue', linewidth=1)

        # 设置图形标题和标签
        # plt.title('OSM Map (UTM Coordinates)')
        # plt.xlabel('UTM X')
        # plt.ylabel('UTM Y')
        # plt.legend()
        # # 显示图形
        # # plt.xticks(xticks)
        # # plt.yticks(yticks)
        # plt.show()
        # plt.savefig("OSMmap_utm_test1215.png")
    
    def mercator_proj(self, lat, lon, scale):
        
        EARTH_RAD_EQ = 6378.137  # in km
        return np.array([scale * lon * (np.pi / 180.0) * EARTH_RAD_EQ, scale * EARTH_RAD_EQ * np.log(np.tan((90.0 + lat) * (np.pi / 360.0)))])
    def plot_map_mercator(self, nodes, ways, scale=0.731353701619):
        # 提取 Nodes 的 Mercator 坐标
        node_mercator_coords = [self.mercator_proj(lat, lon, scale) for (lat, lon) in nodes.values()]
        mercator_x, mercator_y = zip(*node_mercator_coords)
        print(min(mercator_x))
        print(max(mercator_x))
        print(min(mercator_y))
        print(max(mercator_y))
        # 提取 Ways 的 Mercator 坐标
        way_mercator_coords = [[self.mercator_proj(lat, lon, scale) for (lat, lon) in way_nodes] for way_nodes in ways.values()]

        # 创建散点图（Nodes）
        plt.scatter(mercator_x, mercator_y, color='red', label='Nodes')
        # print(mercator_x)
        # 创建折线图（Ways）
        for way_coords in way_mercator_coords:
            way_mercator_x, way_mercator_y = zip(*way_coords)
            plt.plot(way_mercator_x, way_mercator_y, color='blue', linewidth=2)

        # 设置图形标题和标签
        plt.title('OSM Map (Mercator Projection)')
        plt.xlabel('Mercator X')
        plt.ylabel('Mercator Y')
        plt.legend()

        # 显示图形
        plt.show()
        plt.savefig("OSMmap_mercator11.png")
if __name__ == "__main__":
    osm_file = "/home/classlab2/root/OrienterNet/datasets/OSM/map8.osm"
    loader = OSMDataLoader(osm_file)

    # 第一次查询
    min_lat, max_lat = 43.78751965517837,43.788671542038486
    min_lon, max_lon = -79.46387869540725,-79.462288258427 
    nodes, ways = loader.get_nodes_and_ways(min_lat, max_lat, min_lon, max_lon)
    loader.plot_map_utm(nodes, ways)
    # utm_x = [500000, 510000, -520000]
    # utm_y = [4649776, -4649776, 4649776]

    # # Create a scatter plot
    # plt.scatter(utm_x, utm_y, color='blue', label='UTM Coordinates')

    # # Set plot title and labels
    # plt.title('UTM Coordinates Plot')
    # plt.xlabel('UTM X')
    # plt.ylabel('UTM Y')
    # plt.legend()

    # # Display the plot


    # print("Projected Coordinates (UTM) - First Query:")
    # for node_id, (lat, lon) in nodes.items():
    #     utm_x, utm_y = loader.project_to_utm(lat, lon)
    #     # print(f"Node {node_id}: {utm_x}, {utm_y}")

    # for way_id, way_nodes in ways.items():
    #     print(f"Way {way_id}:")
    #     for node in way_nodes:
    #         utm_x, utm_y = loader.project_to_utm(node[0], node[1])
    #         # print(f"  {utm_x}, {utm_y}")

    # # 提取 Nodes 的 UTM 坐标
    # node_utm_coords = [loader.project_to_utm(lat, lon) for (lat, lon) in nodes.values()]
    # utm_x, utm_y = zip(*node_utm_coords)

    # # 提取 Ways 的 UTM 坐标
    # way_utm_coords = [[loader.project_to_utm(lat, lon) for (lat, lon) in way_nodes] for way_nodes in ways.values()]
    # # 提取 Nodes 的经纬度坐标
    # node_coords = list(nodes.values())
    # lat, lon = zip(*node_coords)

    # # 提取 Ways 的经纬度坐标
    # for way_nodes in ways.values():
    #     way_lat, way_lon = zip(*way_nodes)
    #     plt.plot(way_lon, way_lat, color='blue', linewidth=2)
    # # 创建散点图（Nodes）
    # plt.scatter(lon, lat, color='red', label='Nodes')

    # # 创建折线图（Ways）
    # plt.plot(way_lon, way_lat, color='blue', linewidth=2, label='Ways')

    # # 设置图形标题和标签
    # plt.title('OSM Map')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.legend()

    # # 显示图形
    # plt.show()
    # plt.savefig("OSMmap.png")

    # # 创建散点图（Nodes）
    # plt.scatter(utm_x, utm_y, color='red', label='Nodes')

    # # 创建折线图（Ways）
    # for way_coords in way_utm_coords:
    #     way_x, way_y = zip(*way_coords)
    #     plt.plot(way_x, way_y, color='blue', linewidth=2, label='Ways')

    # # 设置图形标题和标签
    # plt.title('OSM Map')
    # plt.xlabel('UTM X')
    # plt.ylabel('UTM Y')
    # plt.legend()

    # # 显示图形
    # plt.show()
    # plt.savefig("UTMmap.png")






