import numpy as np
import cv2
import matplotlib.pyplot as plt

# 假设 polar_image 是极坐标图像
polar_image = cv2.imread("/home/classlab2/16T/datasets/boreas./boreas-2021-01-15-12-17/radar/1610731064202405.png")
import utm
import numpy as np
def polar_to_cartesian(azimuth, range, radar_resolution, num_rows, T0t):
    if T0t is None:
        T0t = np.eye(3)
    theta = azimuth  * 2 * np.pi / num_rows
    x = range * radar_resolution * np.cos(theta)
    y = range * radar_resolution * np.sin(theta)
    corrected_measurement = T0t.dot(np.array([[x],[y],[1]]))
    x = corrected_measurement[0,0]
    y = corrected_measurement[1,0]
    return x,y
def probabilistic_extract_points_from_scan(scan, radar_resolution):
    # 高斯分布的标准差，可以根据实际情况调整
    sigma = 1.0
    # 使用高斯滤波平滑雷达扫描
    smoothed_scan = np.convolve(scan, np.exp(-np.arange(-2, 3)**2 / (2 * sigma**2)), mode='same')

    # 寻找局部最大值的索引
    peaks_idx = np.where((smoothed_scan[1:-1] > smoothed_scan[:-2]) & (smoothed_scan[1:-1] > smoothed_scan[2:]))[0] + 1

    return peaks_idx

def visualize_pointcloud(pointcloud):
    # plt.scatter(pointcloud[0, :], pointcloud[1, :], s=1)  # 使用散点图绘制点云
    plt.scatter(pointcloud[ :,0], pointcloud[ :,1], s=1)
    plt.xlim(min(pointcloud[:,0]),max(pointcloud[:,0]))
    plt.ylim(min(pointcloud[:,1]),max(pointcloud[:,1]))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Cloud Visualization')
    plt.show()
    plt.savefig("cart_point_cloud11.png")
def extract_pointcloud( polar_img, radar_resolution):
    all_x_y = []
    threshold = 40
    polar_img  = np.mean(polar_img, axis=-1)
    num_rows, num_cols = polar_img.shape
    for j in range(polar_img.shape[0]):
        scan = polar_img[j,0:num_cols].copy()
        # peaks_idx = topK_intensity_extract_points_from_scan(scan, topK)
        peaks_idx = probabilistic_extract_points_from_scan(scan, radar_resolution)

        for p in peaks_idx:
            if scan[p] < threshold:
                continue
            azimuth = j
            distance = p
            x, y = polar_to_cartesian(azimuth, distance, radar_resolution, num_rows, T0t=None)
            all_x_y.append([x,y])

    x_y_np = np.array(all_x_y)
    pointcloud_np = np.zeros((2, len(all_x_y)))
    pointcloud_np[0, :] = -np.array(x_y_np[:,1])
    pointcloud_np[1, :] = np.array(x_y_np[:,0])
    return pointcloud_np
radar_resolution =0.0596
point_cloud = extract_pointcloud(polar_image,radar_resolution)
# # print(point_cloud[0, :])
# visualize_pointcloud(point_cloud)

def points_local_to_utm(gps_list, tInd, points):
    gt_curr_utm = utm.from_latlon(gps_list[tInd][0], gps_list[tInd][1])  # in meters
    gt_prev_utm = utm.from_latlon(gps_list[tInd - 1][0], gps_list[tInd - 1][1])
    x = gt_curr_utm[0] - gt_prev_utm[0]
    y = gt_curr_utm[1] - gt_prev_utm[1]

    heading_mercator = np.arctan2(y, x)
    # 创建变换矩阵
    transformation = np.asarray([[np.cos(heading_mercator), -np.sin(heading_mercator)], \
                                 [np.sin(heading_mercator), np.cos(heading_mercator)]])

    # 转换局部坐标到 UTM 坐标
    transformed_points = np.dot(transformation, points)

    # 加上当前 UTM 坐标作为平移
    points_utm = transformed_points + np.array([[gt_curr_utm[0]], [gt_curr_utm[1]]])
    return points_utm

# points_local_to_utm(gps_list=None,tInd=None,indices)
# gps_list = [(43.7821313687205,-79.46607471991867), (43.7821313687205,-79.46607471991867)]  # 代表两个时间步的GPS坐标
# tInd = 1  # 时间步索引
# points_local = point_cloud

# # 调用函数进行坐标转换
# points_utm = points_local_to_utm(gps_list, tInd, points_local)

# 输出转换后的 UTM 坐标
# print("转换后的 UTM 坐标：", points_utm)
# print(points_utm.shape)

# utm_x = points_utm[0, :]
# utm_y = points_utm[1, :]
# # 创建散点图
# plt.scatter(utm_x, utm_y, color='blue', label='UTM Coordinates',s=5)
# plt.xlim(min(utm_x), max(utm_x))
# print(min(utm_x))
# print(max(utm_x))
# print(min(utm_y))
# print(max(utm_y))
# plt.ylim(min(utm_y), max(utm_y))
# # 设置图形标题和标签
# plt.title(f'UTM Coordinates at Time Step {tInd}')
# plt.xlabel('UTM X')
# plt.ylabel('UTM Y')
# plt.legend()
# plt.savefig("utm.png")
# # 显示图形
# plt.show()
def extract_local_coordinates(cartesian_image, resolution):
    # 找到笛卡尔图像中非零值的索引
    non_zero_indices = np.where(cartesian_image[:,:,0] > 50)

    # 提取亮点的坐标
    x_indices, y_indices = non_zero_indices[0], non_zero_indices[1]

    # 计算中心点的坐标
    center_x = cartesian_image.shape[0] // 2
    center_y = cartesian_image.shape[1] // 2

    # 根据分辨率和中心点的坐标计算在本地坐标系中的坐标
    x_local = -(x_indices - center_x) * resolution
    y_local = -(y_indices - center_y) * resolution

    # 如果需要，可以将z坐标设置为某个值，例如0


    # 创建一个3D点云
    point_cloud_local = np.column_stack((x_local, y_local))

    return point_cloud_local


# 示例用法：
# 假设您有笛卡尔图像数据和分辨率
cartesian_image = cv2.imread("/home/classlab2/16T/datasets/boreas./boreas-2021-03-30-14-23/radar/cart_orignal/1617128651922197.png")  # 用实际的笛卡尔图像数据替换
# print(cartesian_image)
resolution = 0.2384  # m/pixel

# 使用提供的函数提取亮点的本地坐标
point_cloud_local = extract_local_coordinates(cartesian_image, resolution)
# visualize_pointcloud(point_cloud_local)
# print(point_cloud_local[:, 0])
# print(point_cloud_local[1])
gps_list = [(43.78214137924524,-79.46470487049496), (43.782127610992724,-79.4646919465159)]  # 代表两个时间步的GPS坐标
utm_coords = [utm.from_latlon(lat, lon) for lat, lon in gps_list]
print("utm",utm_coords)
gps_utm = [623541.08, 4848821.63]
yaw = 2.5319817326863063 + 1/2 * np.pi 

tInd = 1  # 时间步索引
points_local = point_cloud_local.T

# 调用函数进行坐标转换
points_utm = points_local_to_utm(gps_list, tInd, points_local)
print("转换后的 UTM 坐标：", points_utm)
print(points_utm.shape)

utm_x = points_utm[0, :]
utm_y = points_utm[1, :]
plt.scatter(utm_x, utm_y, color='blue',s=1)
plt.xlim(min(utm_x), max(utm_x))
plt.ylim(min(utm_y), max(utm_y))
plt.show()
plt.savefig("cart_point_cloud111.png")