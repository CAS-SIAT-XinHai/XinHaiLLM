import json
import os
import uuid
import subprocess
import yaml
import chromadb
import datetime


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
    # start1()
    for language in ["en", "zh"]:
        dataset_name = "psyqa_balanced" if language == "zh" else "therapistqa_balanced"
        environment_id = str(uuid.uuid4())
        with open(f'/data/xuancheng/koenshen/XinHaiLLM_240821/examples/AutoCBT/configs/xinhai_cbt_{language}.yaml', 'r',
                  encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(f"/data/xuancheng/{dataset_name}.json", 'r', encoding='utf-8') as f:
            ques_descs = json.load(f)
        client = chromadb.PersistentClient(path='/data/pretrained_models/AutoCBT-DB')
        texts = []
        for ques_desc in ques_descs:
            texts.append(ques_desc['question'] + ques_desc['description'])
        work_dir = '/data/xuancheng/koenshen/XinHaiLLM_240821/backend/src'
        result = []
        for i in range(0, 100):
            res = []
            config['arena']['environment']['environment_id'] = environment_id
            config['arena']['agents'][0]['role_description'] = texts[i]
            with open(f'/data/xuancheng/final_cbtagency/configs/xinhai_cbt_{language}_single_turn_{i}.yaml', 'w',
                      encoding='utf-8') as f:
                yaml.safe_dump(config, f, allow_unicode=True)

            always_repeate = True
            while always_repeate:
                try:
                    subprocess.run(['python', '-m', 'xinhai.arena.simulation', '--config_path',
                                    f'/data/xuancheng/final_cbtagency/configs/xinhai_cbt_{language}_single_turn_{i}.yaml'],
                                   cwd=work_dir)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    pass
                target = client.get_collection(f"{environment_id}-0")
                unsorted_metadatas = target.get(include=['metadatas'])['metadatas']
                metadatas = sort_message_of_history(unsorted_metadatas)

                second_meet_user = False
                history = []
                final_answer = ""
                for index, metadata in enumerate(metadatas):
                    aa = metadata['message']
                    if "咨询者" in aa['username'] or "user" in aa['username']:
                        if second_meet_user:
                            final_answer = metadatas[index - 1]['message']["content"]
                            always_repeate = False
                            ques_descs[i]["cbt_answer"] = final_answer
                            ques_descs[i]["cbt_history"] = history
                            ques_descs[i]["cbt_generate_time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            result.append(ques_descs[i])
                            break
                        else:
                            second_meet_user = True
                    dialogue = aa['username'] + "：" + aa['content']
                    history.append(dialogue)
                # 如果always_repeate依旧是True，说明咨询师在10轮对话中还没回复咨询者，开始新一轮的对话

        with open(f'/data/xuancheng/final_cbtagency/single_turn/{dataset_name}_cbtagency_{language}.json', 'w',
                  encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
