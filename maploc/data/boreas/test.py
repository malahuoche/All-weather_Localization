import numpy as np
from scipy.spatial.transform import Rotation
# def parse_calibration_file(path):
#     calib = {}
#     with open(path, "r") as fid:
#         for line in fid.read().split("\n"):
#             if not line:
#                 continue
#             key, *data = line.split(" ")
#             key = key.rstrip(":")
#             if key.startswith("R"):
#                 data = np.array(data, float).reshape(3, 3)
#             elif key.startswith("T"):
#                 data = np.array(data, float).reshape(3)
#             elif key.startswith("P"):
#                 data = np.array(data, float).reshape(3, 4)
#             calib[key] = data
#     return calib

# calib=parse_calibration_file("/home/classlab2/root/OrienterNet/datasets/kitti/2011_10_03/calib_cam_to_cam.txt")
# # print(calib[f"P_rect_02"])

# def get_camera_calibration(calib_dir):
#     calib_path = "/home/classlab2/root/OrienterNet/datasets/kitti/2011_10_03/calib_cam_to_cam.txt"
#     calib_cam = parse_calibration_file(calib_path)
#     P = calib_cam[f"P_rect_02"]
#     K = P[:3, :3]
#     print(K)
#     size = np.array(calib_cam[f"S_rect_02"], float).astype(int)
#     camera = {
#         "model": "PINHOLE",
#         "width": size[0],
#         "height": size[1],
#         "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
#     }

#     t_cam_cam0 = P[:3, 3] / K[[0, 1, 2], [0, 1, 2]]
#     R_rect_cam0 = calib_cam["R_rect_00"]

#     calib_gps_velo = parse_calibration_file(path="/home/classlab2/root/OrienterNet/datasets/kitti/2011_10_03/calib_imu_to_velo.txt")
#     calib_velo_cam0 = parse_calibration_file(path="/home/classlab2/root/OrienterNet/datasets/kitti/2011_10_03/calib_velo_to_cam.txt")
#     R_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["R"]
#     t_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["T"] + calib_velo_cam0["T"]
#     R_cam_gps = R_rect_cam0 @ R_cam0_gps
#     t_cam_gps = t_cam_cam0 + R_rect_cam0 @ t_cam0_gps
#     print(camera)
#     print(R_cam_gps)
#     print(t_cam_gps)
#     return camera, R_cam_gps, t_cam_gps

# get_camera_calibration(calib_dir=None)

# def get_camera_calibration_radiate(calib_dir):
#     calib_path = "/home/classlab2/radiate/calib_cam_to_gps.txt" #目标路径：/home/classlab2/radiate/calib_cam_to_gps.txt
#     # calib_cam = parse_calibration_file(calib_path)

#     # P = calib_cam[f"P_rect_{cam_index:02}"]
#     # K = P[:3, :3]
#     # size = np.array(calib_cam[f"S_rect_{cam_index:02}"], float).astype(int)
#     # camera = {
#     #     "model": "PINHOLE",
#     #     "width": size[0],
#     #     "height": size[1],
#     #     "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
#     # }

#     #直接读取参数
#     left_cam_calib_params = {
#         "fx": 3.379191448899105e+02,  # 焦距X
#         "fy": 3.386957068549526e+02,  # 焦距Y
#         "cx": 3.417366010946575e+02,  # 光学中心X
#         "cy": 2.007359735313929e+02,  # 光学中心Y
#         "k1": -0.183879883467351,     # 畸变系数 k1
#         "k2": 0.0308609205858947,     # 畸变系数 k2
#         "k3": 0,                      # 畸变系数 k3
#         "p1": 0,                      # 畸变系数 p1
#         "p2": 0,                      # 畸变系数 p2
#     }

#     # 构建相机参数字典
#     camera = {
#         "model": "PINHOLE",
#         "width": 672,                 # 图像宽度
#         "height": 376,                # 图像高度
#         "params": np.array([
#             left_cam_calib_params["fx"],left_cam_calib_params["fy"], left_cam_calib_params["cx"],left_cam_calib_params["cy"]])  # 内参矩阵 K
#     }
#     #PS:由【fx，fy，cx，cy的到内参矩阵：
#     K=[[left_cam_calib_params["fx"], 0, left_cam_calib_params["cx"]],
#     [0, left_cam_calib_params["fy"], left_cam_calib_params["cy"]],
#     [0, 0, 1]]

#     # t_cam_cam0 = P[:3, 3] / K[[0, 1, 2], [0, 1, 2]]
#     # R_rect_cam0 = calib_cam["R_rect_00"]
#     # calib_gps_velo = parse_calibration_file(calib_dir / "calib_imu_to_velo.txt")
#     # calib_velo_cam0 = parse_calibration_file(calib_dir / "calib_velo_to_cam.txt")
#     # R_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["R"]
#     # t_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["T"] + calib_velo_cam0["T"]
#     # R_cam_gps = R_rect_cam0 @ R_cam0_gps
#     # t_cam_gps = t_cam_cam0 + R_rect_cam0 @ t_cam0_gps

#     #radiate直接提供了6自由度的参数
#     # 相机的旋转参数
#     camera_rotation_params = [1.278946, -0.530201, 0.000132]

#     # 相机的平移参数
#     camera_translation_params = [0.34001, -0.06988923, 0.287893]
#     # 将旋转参数转换成旋转矩阵
#     camera_rotation = Rotation.from_euler('xyz', camera_rotation_params, degrees=False)
#     R_camera = camera_rotation.as_matrix()

#     # 平移向量直接作为 t_cam_gps
#     t_cam_gps = np.array(camera_translation_params)

#     # 组合旋转矩阵和平移向量
#     R_cam_gps = R_camera
#     print(camera)
#     print(R_cam_gps)
#     print(t_cam_gps)
#     return camera, R_cam_gps, t_cam_gps
# get_camera_calibration_radiate(calib_dir=None)
def align_gps_data(gps_timestamps_file, cam_timestamps_file, target_timestamp_index):
    """
    将GPS数据对齐到目标时间戳索引。

    Args:
        gps_timestamps (path): GPS数据的时间戳文件txt。
        cam_timestamps (path): cam数据的时间戳文件txt.
        target_timestamp_index (int): 目标时间戳索引，用于对齐。

    Returns:
        aligned_gps_index: 对齐后的GPS索引。
    """
    gps_timestamps = []
    cam_timestamps = []
    with open(gps_timestamps_file, 'r') as file:
        for line in file:
            if line.startswith('Frame') and 'Time' in line:
                parts = line.strip().split()
                frame_number = int(parts[1])
                timestamp = float(parts[-1])
                gps_timestamps.append((frame_number, timestamp))
    with open(cam_timestamps_file, 'r') as file:
        for line in file:
            if line.startswith('Frame') and 'Time' in line:
                parts = line.strip().split()
                frame_number = int(parts[1])
                timestamp = float(parts[-1])
                cam_timestamps.append((frame_number, timestamp))
    print(cam_timestamps)
    print(gps_timestamps)
    # 找到距离目标时间戳索引最近的GPS数据
    nearest_gps_data = min(gps_timestamps, key=lambda x: abs(x[0] - target_timestamp_index))

    # 返回对齐后的GPS数据对应的帧索引
    aligned_gps_index = nearest_gps_data[0]

    # 返回对齐后的GPS数据
    return aligned_gps_index

data=align_gps_data(gps_timestamps_file="/home/classlab2/radiate/city_2_0/GPS_IMU_Twist.txt",cam_timestamps_file="/home/classlab2/radiate/city_2_0/zed_left.txt",target_timestamp_index= 1)
print(data)