# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...utils.geo import Projection

split_files = ["test1_files.txt", "test2_files.txt", "train_files.txt"]
from scipy.spatial.transform import Rotation
import math

def quaternion_to_yaw(orientation_quaternion):
    x, y, z, w = orientation_quaternion
    t0 = +2.0 * (w * z + x * y)
    t1 = +1.0 - 2.0 * (z * z + x * x)
    yaw_rad = math.atan2(t0, t1)

    # Adjust the yaw angle to be within the range of -π to +π
    if yaw_rad > math.pi:
        yaw_rad -= 2.0 * math.pi
    elif yaw_rad < -math.pi:
        yaw_rad += 2.0 * math.pi

    # Convert radians to degrees
    yaw_deg = math.degrees(yaw_rad)

    # Adjust the yaw angle based on your custom definition
    yaw_custom = -yaw_deg

    return yaw_custom
def parse_gps_file(path, projection: Projection = None):
    # print(path)
    #raidate数据为：经纬度+四元数
    with open(path, "r") as fid:
        lines = fid.readlines()  # 读取所有行
        # 获取第一行和第五行数据
        first_line = lines[0].strip().replace(' ', ',')
        fifth_line = lines[4].strip().replace(' ', ',')

        # 解析第一行数据
        lat, lon, *_ = map(float, first_line.split(","))  # 拆分并转换字段为浮点数
        quat_x, quat_y, quat_z, quat_w = map(float, fifth_line.split(","))
        quaternion = [quat_x,quat_y,quat_z,quat_w]
        yaw_angle = quaternion_to_yaw(quaternion)
        print(f"Yaw Angle: {yaw_angle} degrees")
        # 计算姿态矩阵 R
    R_world_gps = Rotation.from_quat([quat_x, quat_y, quat_z, quat_w]).as_matrix()
    # with open(path, "r") as fid:
    #     lat, lon, _, roll, pitch, yaw, *_ = map(float, fid.read().split())

    latlon = np.array([lat, lon])
    # R_world_gps = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()#计算R
    t_world_gps = None if projection is None else np.r_[projection.project(latlon), 0]#计算T
    return latlon, R_world_gps, t_world_gps,yaw_angle


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
    calib_path = calib_dir / "calib_cam_to_gps.txt" #目标路径：/home/classlab2/radiate/calib_cam_to_gps.txt
    #直接读取参数
    left_cam_calib_params = {
        "fx": 3.379191448899105e+02,  # 焦距X
        "fy": 3.386957068549526e+02,  # 焦距Y
        "cx": 3.417366010946575e+02,  # 光学中心X
        "cy": 2.007359735313929e+02,  # 光学中心Y
        "k1": -0.183879883467351,     # 畸变系数 k1
        "k2": 0.0308609205858947,     # 畸变系数 k2
        "k3": 0,                      # 畸变系数 k3
        "p1": 0,                      # 畸变系数 p1
        "p2": 0,                      # 畸变系数 p2
    }
    #构建相同格式的相机字典
    #格式：
    #{'model': 'PINHOLE', 'width': 672, 'height': 376, 'params': array([337.91914489, 338.69570685, 341.73660109, 200.73597353])}
    camera = {
        "model": "PINHOLE",
        "width": 672,                 # 图像宽度
        "height": 376,                # 图像高度
        "params": np.array([
            left_cam_calib_params["fx"],left_cam_calib_params["fy"], left_cam_calib_params["cx"],left_cam_calib_params["cy"]])  # 内参矩阵 K
    }
    #radiate直接提供了6自由度的参数
    # 相机的旋转参数
    camera_rotation_params = [1.278946, -0.530201, 0.000132]

    # 相机的平移参数
    camera_translation_params = [0.34001, -0.06988923, 0.287893]
    # 将旋转参数转换成旋转矩阵
    camera_rotation = Rotation.from_euler('xyz', camera_rotation_params, degrees=False)
    R_camera = camera_rotation.as_matrix()

    # 平移向量直接作为 t_cam_gps
    t_cam_gps = np.array(camera_translation_params)

    # 组合旋转矩阵和平移向量
    R_cam_gps = R_camera
    return camera, R_cam_gps, t_cam_gps
