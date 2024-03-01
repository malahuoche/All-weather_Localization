import os
import re
directory_path = "/home/classlab2/16T/datasets/boreas./boreas-2021-01-26-11-22/camera"

# 获取目录中所有文件的文件名
file_names = [filename for filename in os.listdir(directory_path)
              if os.path.isfile(os.path.join(directory_path, filename))]

# 提取文件名中的时间戳部分
files_with_timestamp = [(filename, re.search(r'\d+', filename).group())
                        for filename in file_names if re.search(r'\d+', filename)]

# 按照时间戳排序文件
files_with_timestamp.sort(key=lambda x: x[1])
base_directory = "/home/classlab2/16T/datasets/boreas."
output_file_path = os.path.join(base_directory,'boreas-2021-01-26-11-22', 'camera_timestamp.txt')
# 打印按照时间戳排序后的文件名
for filename, timestamp in files_with_timestamp:
    name, extension = os.path.splitext(filename)
    with open(output_file_path, 'a') as output_file:
        output_file.write(f'{timestamp}\n')
        print(f'Combined file for {timestamp} created at {output_file_path}')
