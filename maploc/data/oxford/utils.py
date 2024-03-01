# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...utils.geo import Projection

split_files = ["test1_files.txt", "test2_files.txt", "train_files.txt"]
from scipy.spatial.transform import Rotation
import os.path as osp

import numpy as np


class Calib:
    """
    Class for loading and storing calibration matrices.
    """

    def __init__(self, calib_root):
        self.P0 = np.loadtxt(osp.join(calib_root, "P_camera.txt"))
        self.T_applanix_aeva = np.eye(4)
        if osp.exists(osp.join(calib_root, "T_applanix_aeva.txt")):
            self.T_applanix_aeva = np.loadtxt(osp.join(calib_root, "T_applanix_aeva.txt"))
        self.T_applanix_lidar = np.loadtxt(osp.join(calib_root, "T_applanix_lidar.txt"))
        self.T_camera_lidar = np.loadtxt(osp.join(calib_root, "T_camera_lidar.txt"))
        self.T_radar_lidar = np.loadtxt(osp.join(calib_root, "T_radar_lidar.txt"))

    def print_calibration(self):
        print("P0:")
        print(self.P0)
        print("T_applanix_aeva:")
        print(self.T_applanix_aeva)
        print("T_applanix_lidar:")
        print(self.T_applanix_lidar)
        print("T_camera_lidar:")
        print(self.T_camera_lidar)
        print("T_radar_lidar:")
        print(self.T_radar_lidar)
def parse_gps_file(path, projection: Projection = None):
    # print(path)
    with open(path, "r") as fid:
        lat, lon,roll, pitch, yaw = map(float, fid.read().split())
    latlon = np.array([lat, lon])
    R_world_gps = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    t_world_gps = None if projection is None else np.r_[projection.project(latlon), 0]
    return latlon, R_world_gps, t_world_gps #先对齐
def parse_combined_file(root, date, index):
    combined_path = root / date / "gps" / Path(index).with_suffix(".txt")
    gps_list = []
    with open(combined_path, 'r') as file:
        # 逐行读取文件
        for line in file:
            if str(Path(index).stem) in line:
                data = line.split()
                # 提取第5列、第6列、第10列和第11列数据，并转换为浮点数
                lat1, lon1, lat2, lon2 = map(float, (data[4], data[5], data[9], data[10]))
                # 构成坐标对，添加到 gps_list
                gps_list.append((lat1, lon1))
                gps_list.append((lat2, lon2))
    return gps_list

def parse_split_file(path: Path):
    with open(path, "r") as fid:
        info = fid.read()
    names = []
    shifts = []
    for line in info.split("\n"):
        if not line:
            continue
        name, *shift = line.split()
        names.append(tuple(name.split("/")))
        if len(shift) > 0:
            assert len(shift) == 3
            shifts.append(np.array(shift, float))
    shifts = None if len(shifts) == 0 else np.stack(shifts)
    return names, shifts


def parse_calibration_file(path):
    calib = {}
    with open(path, "r") as fid:
        for line in fid.read().split("\n"):
            if not line:
                continue
            key, *data = line.split(" ")
            key = key.rstrip(":")
            if key.startswith("R"):
                data = np.array(data, float).reshape(3, 3)
            elif key.startswith("T"):
                data = np.array(data, float).reshape(3)
            elif key.startswith("P"):
                data = np.array(data, float).reshape(3, 4)
            calib[key] = data
    # print(calib)
    return calib


def get_camera_calibration(calib_dir):
    calib_path = calib_dir  #目标路径：/home/classlab2/radiate/calib_cam_to_gps.txt
    # 相机的外参
    R_camera_gps = np.eye(3)  # 旋转矩阵为单位矩阵
    t_camera_gps = np.zeros(3)  # 平移向量为零向量

    # GPS的外参
    R_gps = np.array([[0.999882, -0.015508, 0.001437],
                    [0.015509, 0.999875, -0.004458],
                    [-0.001428, 0.004459, 0.999989]])  # 旋转矩阵
    t_gps = np.array([-1.7132, 0.1181, 1.1948])  # 平移向量
    # 计算相机到GPS的变换矩阵
    R_cam_gps = np.dot(R_gps, R_camera_gps)
    t_cam_gps = np.dot(R_gps, t_camera_gps) + t_gps
    #直接读取参数
    left_cam_calib_params = {
        "fx": 983.044006,
        "fy": 983.044006,
        "cx": 643.646973,
        "cy": 493.378998,
        "k1": 0.0144210402954516,
        "k2": -0.0148287624062537,
        "k3": 0,                      # 畸变系数 k3
        "p1": 0,                      # 畸变系数 p1
        "p2": 0,                      # 畸变系数 p2
    }
    #构建相同格式的相机字典
    #格式：
    #{'model': 'PINHOLE', 'width': 672, 'height': 376, 'params': array([337.91914489, 338.69570685, 341.73660109, 200.73597353])}
    camera = {
        "model": "PINHOLE",
        "width": 1280,                 # 图像宽度
        "height": 960,                # 图像高度
        "params": np.array([
            left_cam_calib_params["fx"],left_cam_calib_params["fy"], left_cam_calib_params["cx"],left_cam_calib_params["cy"]])  # 内参矩阵 K
    }
    #radiate直接提供了6自由度的参数
    # # 相机的旋转参数
    # camera_rotation_params = [1.278946, -0.530201, 0.000132]

    # # 相机的平移参数
    # camera_translation_params = [0.34001, -0.06988923, 0.287893]
    # # 将旋转参数转换成旋转矩阵
    # camera_rotation = Rotation.from_euler('xyz', camera_rotation_params, degrees=False)
    # R_camera = camera_rotation.as_matrix()

    # # 平移向量直接作为 t_cam_gps
    # t_cam_gps = np.array(camera_translation_params)

    # # 组合旋转矩阵和平移向量
    # R_cam_gps = R_camera
    # R_cam_gps=R_radar_gps
    # t_cam_gps=t_radar_gps
    return camera, R_cam_gps, t_cam_gps
