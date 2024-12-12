import os, sys, shutil

input_path = sys.argv[1]
output_path = sys.argv[2]
os.makedirs(output_path, exist_ok=True)

for root, dirs, files in os.walk(input_path):
    for file in files:
        if file.endswith('.png'):
            path = os.path.join(root, file)
            shutil.copy(path, os.path.join(output_path, file))
            print(f'copy {file}')

"""
import os
import random
import shutil
 
def copy_random_files(source_folder, destination_folder, num_files=10):
    # 获取源文件夹内的所有文件路径
    files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    # 随机挑选num_files个文件
    random.shuffle(files)  # 先打乱文件顺序
    files_to_copy = files[:num_files]  # 取前num_files个文件
    
    # 复制文件到目标文件夹
    for file in files_to_copy:
        shutil.copy(file, destination_folder)
 
# 使用示例
source_folder = 'F:\\ly\\total_background'
destination_folder = 'F:\\ly\\total_background_random'
copy_random_files(source_folder, destination_folder,2000)
"""
"""
import shutil
import glob
import os
for each_jason in glob.glob("F:\\SJJR\\all_image_sjjr\\*.tif"):
	if os.path.exists(each_jason.replace('tif','json')):
		print('json')
	else:
		print('tif')
		#print(each_jason.replace('/ ',' '))
		#break.replace('')
		filename = each_jason
		print(filename)
		os.remove(filename)
		#os.system('rm -rf '+each_jason)
	
"""





