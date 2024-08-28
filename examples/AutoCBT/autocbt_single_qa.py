import json
import os
import uuid
import subprocess
import yaml
import chromadb
import datetime
import threading
import pytz

# project_path = "/data/xuancheng/koenshen/XinHaiLLM_240821"
# data_path = "/data/xuancheng/koenshen/XinHaiLLM_240821/data"
# db_path = "/data/xuancheng/koenshen/AutoCBT-DB"

project_path = "/data/yangmin/autocbt/XinHaiLLM"
data_path = "/data/yangmin/autocbt/data"
db_path = "/data/yangmin/autocbt/AutoCBT-DB"

client = chromadb.PersistentClient(path=db_path)

def hello(config:dict, role_description:str, data_dict: dict) -> dict:
    res = []
    environment_id = str(uuid.uuid4())
    config['arena']['environment']['environment_id'] = environment_id
    config['arena']['agents'][0]['role_description'] = role_description
    with open(f'{data_path}/configs/{environment_id}.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    try:
        subprocess.run(['python', '-m', 'xinhai.arena.simulation', '--config_path', f'{data_path}/configs/{environment_id}.yaml'], cwd=work_dir)
    except Exception as e:
        print(f"An error occurred: {e}")
        pass

    target_whole = client.get_collection(f"{environment_id}-1")
    unsorted_metadatas_whole = target_whole.get(include=['metadatas'])['metadatas']
    metadatas_whole = sort_message_of_history(unsorted_metadatas_whole)

    final_answer = metadatas_whole[-1]['message']['content']
    data_dict["cbt_answer"] = final_answer
    data_dict["cbt_history"] = metadatas_whole
    beijing_tz = pytz.timezone('Asia/Shanghai')  # 创建北京时间（亚洲/上海）时区对象
    beijing_time = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')  # 获取当前的北京时间
    data_dict["cbt_generate_time"] = beijing_time
    data_dict["environment_id"] = environment_id

    return data_dict

def process_data(data_list: list, start_index: int, end_index: int):
    temp_list = data_list[start_index: end_index]
    result_list = []
    for index, data_dict in enumerate(temp_list):
        print(f"XXX {index} : {start_index}->{end_index} XXXXXXXXXXXXXXXXXXXXXXXXX\n")
        role_description = data_dict['question'] + data_dict['description']
        with open(f'{project_path}/examples/AutoCBT/configs/xinhai_cbt_{language}.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        result_dict = hello(config, role_description, data_dict)
        result_list.append(result_dict)

        with open(f"{data_path}/result/therapistqa_balanced_cbtagency_{start_index}_{end_index}.json", 'w', encoding='utf-8') as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)

def sort_message_of_history(data_list: list):
    parsed_list = []
    # 假设 data_list 是你的原始列表
    for item in data_list:
        # 将 content 字符串解析为字典
        message = json.loads(item["message"])
        # 将 indexId 添加为 item 的键值对，以便排序使用
        item["indexId"] = int(message["indexId"])
        item["message"] = message
        parsed_list.append(item)

    # 根据 indexId 从大到小排序
    sorted_list = sorted(parsed_list, key=lambda x: x["indexId"], reverse=False)
    return sorted_list

if __name__ == "__main__":
    for language in ["en"]:
        dataset_name = "psyqa_balanced" if language == "zh" else "therapistqa_balanced"

        with open(f"{data_path}/{dataset_name}.json", 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        work_dir = f'{project_path}/backend/src'

        num_threads = 10
        num_dicts_per_thread = len(data_list) // num_threads
        threads = []
        for i in range(num_threads):
            print(f"=================== {i} start==========================")
            start_index = i * num_dicts_per_thread
            end_index = (i + 1) * num_dicts_per_thread if i < num_threads - 1 else len(data_list)
            thread = threading.Thread(target=process_data, args=(data_list, start_index, end_index))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()
