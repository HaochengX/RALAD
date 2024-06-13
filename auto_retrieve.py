import sentence_embedding as st
from typing import List, Optional
import fire
from llama import Dialog, Llama
 

# def rename_files(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith('_SMALL.c'):
#             name_match = re.match(r'(.*)_SMALL\.c', filename)
#             if name_match:
#                 name = name_match.group(1)
#                 new_filename = f'kernel_{name}.c'
#                 os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
#                 print(f"Renamed '{filename}' to '{new_filename}'")
# directory_path = '/home/haotian/RALAD/cfile-small'
# rename_files(directory_path)

import os

def read_c_files(directory):
    c_files_content = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.c'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                c_files_content[filename] = file.read()
    
    return c_files_content

# 指定cfile-small文件夹路径
directory_path = '/home/haotian/RALAD/cfile-small'

# 读取所有.c文件内容
c_files_content = read_c_files(directory_path)

# Assuming you already have the c_files_content dictionary and the select_sentenceembedding function
for filename, content in c_files_content.items():
    helper = st.select_sentenceembedding(query=content)
    helper = ', '.join([f'assistant {I + 1}: {c}' for I, c in enumerate(helper)])
    helper = f'You are now given some HLS pragma insertion assistants\n{helper}\n Now try your best to Optimize the following code through inserting pragma hls lines\n{content}'
    # 创建一个新的文件夹，文件夹名称为“output_files”
    output_dir = 'retrieve_query'
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个文件创建一个新的文件，文件名为filename.txt，同时文件内容为helper
    with open(os.path.join(output_dir, f'{filename}.txt'), 'w', encoding='utf-8') as f:
        f.write(helper)