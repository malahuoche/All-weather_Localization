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
# canvas = tiler.query(bbox)
# canvas = tile_manager.query(bbox_tile)
# 创建保存图像的目录
output_dir = "/home/classlab2/16T/datasets/boreas./raster2"
os.makedirs(output_dir, exist_ok=True)
# 查询 TileManager 并打印每个 canvas
# 创建一个子图
fig, axes = plt.subplots()
#用来叠图的
# def debug_visualize_global_observation(options, radar_global, map_global, local_radar_normals_radian, top_pose, intersect_segments, \
#                             associations, gt_pos, frame_idx, street_id, component_id, b_long_temp, b_lateral_temp,\
#                                        best_long_features_idx, best_lateral_features_idx, mapgraph, street):

#     # To debug normals angle and point angle
#     visualization_offset = 0.2  # km
#     # import matplotlib
#     # matplotlib.use('agg')
#     # import matplotlib.pyplot as plt



#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(111)
#     points_normal_global = np.zeros(radar_global.shape[1])
#     for i in range(radar_global.shape[1]):
#         point_normal_global_angle = wrap_angle(local_radar_normals_radian[i] + top_pose[2])
#         points_normal_global[i] = point_normal_global_angle

#     lines = []
#     for l_idx in range(intersect_segments.shape[0]):
#         line = intersect_segments[l_idx, :]
#         lines.append(np.vstack((line[0:2], line[2:4])))

#     ax.add_collection(
#         LineCollection(lines, transOffset=ax.transData, linewidths=0.1, colors='green'))
#     ax.quiver(radar_global[0, associations[:,0]],  radar_global[1, associations[:,0]],\
#               np.cos(points_normal_global[associations[:,0]]), np.sin(points_normal_global[associations[:,0]]), \
#               width=0.002, color='blue', alpha=0.4)
#     ax.scatter(map_global[0, associations[:,1]], map_global[1, associations[:,1]], c='green', s=0.02)

#     if len(best_long_features_idx) > 0:
#         # To show lateral and long features
#         # ax.scatter(radar_global[0, best_long_features_idx], radar_global[1, best_long_features_idx], alpha=0.4, \
#         #            s=0.2, c='purple', label='long features')
#         ax.quiver(radar_global[0, best_long_features_idx],  radar_global[1, best_long_features_idx],\
#               np.cos(points_normal_global[best_long_features_idx]), np.sin(points_normal_global[best_long_features_idx]), \
#               width=0.002, color='purple', alpha=0.4, label=' long features')

#     if len(best_lateral_features_idx) > 0:
#         ax.quiver(radar_global[0, best_lateral_features_idx],  radar_global[1, best_lateral_features_idx],\
#               np.cos(points_normal_global[best_lateral_features_idx]), np.sin(points_normal_global[best_lateral_features_idx]), \
# #               width=0.002, color='yellow', alpha=0.4, label=' lateral features')


#     # titles_list = ['frame = ' + str(frame_idx) + ', street id = ' + str(street_id) + ' ' + ', comp id = ' + str(component_id) + \
#     #                '\nLongitude correction = ',  str(b_long_temp), ', lateral correction = ', str(b_lateral_temp)]
#     # if b_long_temp:
#     #     long_color = 'green'
#     # else:
#     #     long_color = 'red'
#     # if b_lateral_temp:
#     #     lat_color = 'green'
#     # else:
#     #     lat_color = 'red'
#     # tt_colors = ['black', long_color, 'black', lat_color]
#     # color_title(titles_list, tt_colors)

#     ax.scatter(gt_pos[0], gt_pos[1], s=0.2, c='red')
#     ax.quiver(top_pose[0],  top_pose[1],\
#                np.cos(top_pose[2]), np.sin(top_pose[2]), width=0.002, color='black')
#     ax.set_xlim([gt_pos[0] - visualization_offset, gt_pos[0] + visualization_offset])
#     ax.set_ylim([gt_pos[1] - visualization_offset, gt_pos[1] + visualization_offset])
#     ax.legend(loc='lower left')
#     ax.set_title('Frame = ' + str(frame_idx) + '\nLong correction = ' + str(b_long_temp) + ', lat correction = ' + str(b_lateral_temp))

#     ax.plot([mapgraph.node[street]['origin'][0], mapgraph.node[street]['terminus'][0]], \
#             [mapgraph.node[street]['origin'][1], mapgraph.node[street]['terminus'][1]], \
#             linewidth=0.3, c='black')
#     directory = cwd + '/results/visualization/' + options.dataname + \
#                 '/debug_point_line_normal_' + str(frame_idx).zfill(4) 
#     fig_name = directory +'.png'
#     fig.savefig(fig_name, dpi=displayDPI)
#     plt.close()

# def build_osm_map(binFile,doSimplify = True,prunePolylineNodes = True, lat_range = None, lon_range = None):
#     doSimplify = False

#     tObj = osm_objects.Objects(True)
#     tObj.loadFromFile(binFile)
#     tRange = tObj.getMapRange()
#     # print 'map range: lat {0} - {1}, lon {2} - {3}'.format(tRange.lat_min,tRange.lat_max,tRange.lon_min,tRange.lon_max)
#     if lat_range != None:
#         tRange.geo_min.lat,tRange.geo_max.lat = lat_range[0],lat_range[1] 
#     if lon_range != None:
#         tRange.geo_min.lon,tRange.geo_max.lon = lon_range[0],lon_range[1] 
#     tObjs = tObj.getObjectsInRange(tRange)

#     Npt = 0
#     Npoly = 0
#     NpolySegs = 0
#     ptCoordSum = np.array([0,0]).T

#     intersections = {}
#     for t in tObjs:
#         # todo ziyang added
#         # if 'service' in t.tag:
#         #     continue
#         # todo ziyang added
#         if t.isPoint():
#             tSha = t.sha
#             tPt = t.getPoint()
#             ptLat = tPt.geo.lat
#             ptLon = tPt.geo.lon
#             intersections[tSha] = dict()
#             ptCoord = np.array([ptLat,ptLon]).T
#             intersections[tSha]['latlon'] = ptCoord
#             Npt += 1
#             # ptCoordSum += ptCoord
#             np.add(ptCoordSum, ptCoord, out=ptCoordSum, casting="unsafe")

#     ptCoordMean = ptCoordSum/Npt
#     projScale = np.cos(ptCoordMean[0]*(np.pi/180.0)) # default

#     ptCoordOrigin = mercatorProj(ptCoordMean,projScale)

def mercatorProj(latlon,scale):
    EARTH_RAD_EQ = 6378.137 # in km
    return np.array([scale*latlon[1]*(np.pi/180.0)*EARTH_RAD_EQ, scale*EARTH_RAD_EQ*np.log(np.tan((90.0 + latlon[0]) * (np.pi/360.0)))])
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
    # print(bbox_tile)
    bbox_osm = projection.unproject(bbox_tile)
    # print(bbox_osm)
    # 查询瓦片中感兴趣区域的栅格数据
    canvas = tile_manager.query(bbox_tile)
    map_viz = Colormap.apply_lines(canvas.raster)
    # plot_images([map_viz], titles=["OpenStreetMap raster"])
    # plot_nodes(1, canvas.raster[2], fontsize=6, size=10)
    # 保存图像
    output_path = os.path.join(output_dir, f"tile_{ij}.png")
    axes.clear()
    image_tensor=plot_images([map_viz], titles=["OpenStreetMap raster"])
    # print("Image Tensor:")
    # print(image_tensor[0])
    # print("Image Tensor Shape:", image_tensor[0].size())
    
    # plot_nodes(0, canvas.raster[2], fontsize=6, size=10)
    # plt.savefig(output_path)
    
    # print(f"Saved image to {output_path}")
plt.close()


