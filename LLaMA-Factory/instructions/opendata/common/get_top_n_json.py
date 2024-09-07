import os
import json
from second_handle import *
# 设置数据目录的路径
ori_path = "/data/datasets/XinHai_temp"
new_path = "/data/datasets/XinHai_V1.1"

top_n = 10000
cut_file_name = '.json'
# 获取所有.json文件的文件名
files = [f for f in os.listdir(ori_path) if f.endswith(cut_file_name)]
# 遍历每个文件，并对文件名进行排序
files.sort(key=lambda x: x[:-5])  # 排序时去掉.json后缀
# 遍历每个文件
for file_name in files:
    # 打开并读取JSON文件
    with open(os.path.join(ori_path, file_name), 'r') as file:
        # 解析JSON文件中的数据（假设是一个JSON数组）
        data = json.load(file)
        # 检查data确实是一个列表
        if not isinstance(data, list):
            continue

        if file_name == 'cmb.json' or file_name == 'headqa.json' or file_name == 'medqajin.json' or file_name == 'mlecqa.json' or file_name == 'nlpec.json':
            sorted_data = data
        else:
            json_entries = []
            for entry in data:
                # 将json条目转换为字符串
                entry_str = json.dumps(entry)
                # 记录长度和内容
                json_entries.append((len(entry_str), entry))
            # 根据长度从长到短排序，并取前top_n个（如果有的话）
            json_entries_sorted = sorted(json_entries, key=lambda x: x[0], reverse=True)[:top_n]
            # 从排序后的元组中提取字符串部分
            sorted_data = [item[1] for item in json_entries_sorted]

        # 将处理后的数据写回到新的JSON文件中
        with open(os.path.join(new_path, file_name), 'w', encoding='utf-8') as outfile:
            json.dump(sorted_data, outfile, ensure_ascii=False)
        print(f"{file_name}已保存到{new_path}中！")
print("所有文件处理完成！")
