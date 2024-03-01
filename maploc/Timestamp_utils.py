import os
import re
directory_path = "/home/classlab2/16T/datasets/boreas./boreas-2021-01-26-11-22/radar/cart"


# 获取目录中所有文件的文件名
file_names = [filename for filename in os.listdir(directory_path)
              if os.path.isfile(os.path.join(directory_path, filename))]

# 提取文件名中的时间戳部分
files_with_timestamp = [(filename, re.search(r'\d+', filename).group())
                        for filename in file_names if re.search(r'\d+', filename)]

# 按照时间戳排序文件
files_with_timestamp.sort(key=lambda x: x[1])
base_directory = "/home/classlab2/16T/datasets/boreas."
output_file_path = os.path.join(base_directory,'boreas-2021-01-26-11-22', 'matched_timestamp.txt')

camera_timestamp_file = os.path.join(base_directory, 'boreas-2021-01-26-11-22', 'camera_timestamp.txt')
camera_timestamps = []
with open(camera_timestamp_file, 'r') as file:
    for line in file:
        timestamp = int(line.strip())
        camera_timestamps.append(timestamp)
# 打印按照时间戳排序后的文件名
for filename, timestamp in files_with_timestamp:
    name, extension = os.path.splitext(filename)
    # print(name)
    closest_camera_timestamp = min(camera_timestamps, key=lambda x: abs(x -int(name)))
    # closest_camera_timestamp = min(camera_timestamps, key=lambda x: abs(x -int(name)))
    print(f"File: {filename}, Closest Camera Timestamp: {closest_camera_timestamp}")
    # txt_path = os.path.join(base_directory,'boreas-2021-03-30-14-23', 'gps', f'{timestamp}.txt')
    # if os.path.isfile(txt_path):
    #     with open(txt_path, 'r') as txt_file:
    #         # 读取txt文件的内容
    #         gps_info = txt_file.read()
    #         # 创建新的txt文件并写入时间戳和gps信息
    #         with open(output_file_path, 'a') as output_file:
    #             output_file.write(f'{timestamp} {gps_info}\n')
    #         print(f'Combined file for {timestamp} created at {output_file_path}')
    # else:
    #     print(f'Txt file not found for timestamp: {timestamp}')
    # 记录camera的时间戳
    with open(output_file_path, 'a') as output_file:
        output_file.write(f'{timestamp} {closest_camera_timestamp}\n')
        print(f'Combined file for {timestamp} created at {output_file_path}')

