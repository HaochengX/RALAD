import os
import sys
import csv
import string
import random
from datetime import datetime
from shutil import copyfile, rmtree
# from hls_timing_parser import extract_hls_timing
# from rtl_timing_parser import extract_rtl_timing
# from adb_parser_modified import parse
import time
hls_cmd = "vitis_hls"

def create_tcl_files(src_directory):
    # 检查并创建输出目录
    output_dir = os.path.join(src_directory, 'output-4')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历源目录中所有的文件和文件夹
    for folder_name in os.listdir(src_directory):
        # 确保是以 'kernel_' 开始的文件夹
        if folder_name.startswith('kernel_'):
            folder_path = os.path.join(src_directory, folder_name)
            if os.path.isdir(folder_path):
                # 为每个符合条件的文件夹创建对应的输出子文件夹
                output_subfolder = os.path.join(output_dir, folder_name)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                # 遍历子文件夹中所有文件
                for file_name in os.listdir(folder_path):
                    # 检查是否是.c文件
                    if file_name.endswith('.c'):
                        # 获取文件的绝对路径
                        file_path = os.path.join(folder_path, file_name)
                        # 获取文件名，去除扩展名，为其创建.tcl文件
                        base_name = os.path.splitext(file_name)[0]
                        tcl_file_path = os.path.join(output_subfolder, base_name + '.tcl')
                        # 创建.tcl文件，并写入内容
                        with open(tcl_file_path, 'w') as tcl_file:
                            tcl_content = f"""open_project -reset {folder_path}/{folder_name}
set_top {folder_name}
add_files {file_path} -cflags "-std=c++11"  ;# Use appropriate C++ standard

open_solution -reset "solution1"
set_part {{xcu250-figd2104-2L-e}}
create_clock -period 5.0ns  ;# Adjust clock period as needed based on your design

# Run synthesis and export design
csynth_design
export_design -format ip_catalog -version "1.0"

exit
"""
                            tcl_file.write(tcl_content)
                            tcl_file.flush()
                            os.fsync(tcl_file.fileno())
                            print('Running TCL file (HLS syn):' + hls_cmd + " " + tcl_file_path)
                            start = time.time()
                            log_path = output_subfolder + "/vitis_hls.log"
                            os.chdir(output_subfolder)
                            if os.system(hls_cmd + " ../" + tcl_file_path + " -l ../" + log_path) != 0:
                                print("Error during HLS syn of " + folder_name)  
                                tcl_file.close()  # an error occurred
                                os.chdir("..")
                                # keep only the last 250 lines of the log file and store them in 'errors' folder
                                os.system("tail -n 250 " + log_path + " > " + output_subfolder + "/" + folder_name + ".log")
                                # rmtree(tmp_prj)
                                continue


    print("TCL files have been created successfully.")

# 调用函数
create_tcl_files('/home/haotian/RALAD/Retrieve-code-gpt4')
