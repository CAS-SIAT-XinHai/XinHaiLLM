import json
import os
import uuid
import subprocess
import yaml
import chromadb
import datetime
import threading
import pytz
import time
import random

# project_path = "/data/xuancheng/koenshen/XinHaiLLM_240821"
# data_path = "/data/xuancheng/koenshen/XinHaiLLM_240821/data"
# db_path = "/data/xuancheng/koenshen/AutoCBT-DB"

# project_path = "/data/yangmin/autocbt/XinHaiLLM"
# data_path = "/data/yangmin/autocbt/data"
# db_path = "/data/yangmin/autocbt/AutoCBT-DB"

project_path = "/mnt/c/koenshen/SVN/XinHaiLLM_240921/XinHaiLLM"
data_path = "/mnt/c/koenshen/SVN/XinHaiLLM_data_and_db/data"
db_path = "/mnt/c/koenshen/SVN/XinHaiLLM_data_and_db/AutoCBT-DB"
work_dir = f'{project_path}/backend/src'
language = 'zh'
# 创建北京时间（亚洲/上海）时区对象
beijing_tz = pytz.timezone('Asia/Shanghai')
client = chromadb.PersistentClient(path=db_path)

def read_files_by_path(path: str, suffix=".json"):
    # 遍历指定目录下的所有文件
    result_list = []
    for filename in os.listdir(path):
        if filename.endswith(suffix):
            result_list.append(filename)
    return result_list

def hello(config:dict, role_description:str, data_dict: dict):
    environment_id = str(uuid.uuid4())
    config['arena']['environment']['environment_id'] = environment_id
    config['arena']['agents'][0]['role_description'] = role_description
    config_file = f"{data_path}/configs/{config['arena']['agents'][0]['locale']}_{data_dict['questionID']}_{environment_id}.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    # 避开api的限速
    time.sleep(30 + random.randint(0, 60))
    try:
        subprocess.run(['python', '-m', 'xinhai.arena.simulation', '--config_path', config_file], cwd=work_dir)
    except Exception as e:
        print(f"An error occurred: {e}")

def process_data(data_list: list, start_index: int, end_index: int):
    temp_list = data_list[start_index: end_index]
    for index, data_dict in enumerate(temp_list):
        print(f"XXX {index} : {start_index}->{end_index} XXXXXXXXXXXXXXXXXXXXXXXXX\n")
        role_description = data_dict['question'] + data_dict['description']
        with open(f'{project_path}/examples/AutoCBT/configs/xinhai_cbt_{language}.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        hello(config, role_description, data_dict)

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

def chat_completion(start_index, end_index, num_threads, language):
    #先对话，保存对话记录到storage中
    dataset_name = "psyqa_balanced" if language == "zh" else "therapistqa_balanced"
    with open(f"{data_path}/{dataset_name}.json", 'r', encoding='utf-8') as f:
        whole_list = json.load(f)
        data_list = whole_list[start_index: end_index]

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

def generate_data_result_file(start_index):
    # 从storage中提取对话记录作为结果
    dataset_name = "psyqa_balanced" if language == "zh" else "therapistqa_balanced"
    yaml_file_list = read_files_by_path(path=f"{data_path}/configs", suffix=".yaml")

    with open(f"{data_path}/{dataset_name}.json", 'r', encoding='utf-8') as f:
        whole_list = json.load(f)
    # 创建一个字典，键为 questionID，值为对应的元素
    id_index = {element['questionID']: element for element in whole_list}

    result_list = []
    for yaml_file in yaml_file_list:
        name_str_list = yaml_file.replace(".yaml", "").split('_')
        question_id = name_str_list[1]
        environment_id = name_str_list[2]

        target_whole = client.get_collection(f"{environment_id}-1")
        unsorted_metadatas_whole = target_whole.get(include=['metadatas'])['metadatas']
        metadatas_whole = sort_message_of_history(unsorted_metadatas_whole)
        final_answer_message = metadatas_whole[-1]['message']
        if final_answer_message["username"] == '心理咨询师' or final_answer_message["username"] == 'counselor':
            final_answer = final_answer_message['content']
        else:
            raise Exception("Empty content of final answer to user")

        data_dict = id_index.get(question_id)
        data_dict["cbt_answer"] = final_answer
        data_dict["cbt_history"] = metadatas_whole
        beijing_time = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')  # 获取当前的北京时间
        data_dict["cbt_generate_time"] = beijing_time
        data_dict["environment_id"] = environment_id
        result_list.append(data_dict)
    with open(f"{data_path}/result/{dataset_name}_{start_index}_autocbt.json", 'w', encoding='utf-8') as f:
        json.dump(result_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    num_threads = 10
    each_time_add_num = 10
    start_index = 0
    language = "en"
    while start_index < 100:
        print(f"new start threads {start_index}")
        time.sleep(120)
        chat_completion(start_index, start_index+each_time_add_num, num_threads, language)
        generate_data_result_file(start_index)
        start_index += each_time_add_num