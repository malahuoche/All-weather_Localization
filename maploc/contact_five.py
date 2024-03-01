import cv2
import torch
import numpy as np
from pyproj import Proj, transform
gps_list = [(43.78214137924524,-79.46470487049496), (43.782127610992724,-79.4646919465159),(43.78211354534507,-79.46467897822718),(43.78209932213006,-79.46466585177697),(43.782084932005986,-79.46465252623862),(43.78207040992329,-79.4646388768435)]  # 代表两个时间步的GPS坐标
wgs84 = Proj(init='epsg:4326')  # EPSG code 4326 corresponds to WGS 84
utm = Proj(init='epsg:32617')  #
# 将GPS坐标转换为UTM坐标
utm_coords = [transform(wgs84, utm, lon, lat) for lat, lon in gps_list]

# 打印转换后的UTM坐标
for utm_coord in utm_coords:
    print(utm_coord)