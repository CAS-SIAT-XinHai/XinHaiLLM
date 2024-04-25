import argparse
import json
import logging
from collections import OrderedDict

from more_itertools import chunked
from tqdm.auto import tqdm
from second_handle import *

"""
label_4class is the primary type that contains:
病症 药物 治疗方案 其他

label_36class is the secondary type that contains:
病症：定义，病因，临床表现，相关病症，治疗方法，推荐医院，预防，所属科室，禁忌，传染性，治愈率，严重性
药物：作用，适用症，价钱，药物禁忌，用法，副作用，成分
治疗方案：方法，费用，有效时间，临床意义/检查目的，治疗时间，疗效，恢复时间，正常指标，化验/体检方案，恢复
其他：设备用法，多问，养生，整容，两性，对比，无法确定

{
 "originalText": "间质性肺炎的症状?", 
 "entities": [{"label_type": "疾病和诊断", "start_pos": 0, "end_pos": 5}], 
 "seg_result": ["间质性肺炎", "的", "症状", "?"], 
 "label_4class": ["病症"], 
 "label_36class": ["临床表现"]
}
"""

label_types = {'疾病和诊断', '解剖部位', '手术', '影像检查', '药物', '实验室检验'}


def convert(opts):
    with open(f"{opts.data_dir}/CMID.json", 'r', encoding='utf-8') as fd:
        data = json.load(fd)

    sft_entries = []
    for i, item in enumerate(tqdm(data)):
        if i == 0:
            print(item)

        desc = item['originalText']

        entities = []
        for e in item['entities']:
            if e['label_type'] == '疾病和诊断':
                entities.append(desc[e['start_pos']:e['end_pos']])
            elif e['label_type'] == '解剖部位':
                entities.append(desc[e['start_pos']:e['end_pos']])
            elif e['label_type'] == '手术':
                entities.append(e["label_type"] + desc[e['start_pos']:e['end_pos']])
            elif e['label_type'] == '影像检查':
                entities.append(e["label_type"] + desc[e['start_pos']:e['end_pos']])
            elif e['label_type'] == '药物':
                entities.append(e["label_type"] + desc[e['start_pos']:e['end_pos']])
            elif e['label_type'] == '实验室检验':
                entities.append(e["label_type"] + desc[e['start_pos']:e['end_pos']])

        label_4class = '，'.join([label.strip("'") for label in item['label_4class'] if label not in ["'其他'"]])
        label_36class = '，'.join([label.strip("'") for label in item['label_36class'] if label not in ["'无法确定'", "'多问'"]])

        output = '该描述'

        if label_4class:
            if entities:
                output += f"提到了{'，'.join(entities)}"
                output += f", 并期望知道该{label_4class}"
            else:
                output += f"期望知道该{label_4class}"

            if label_36class:
                output += f"及其{label_36class}。"
            else:
                output += "。"
        else:
            if entities:
                output += f"提到了{'，'.join(entities)}"
            if label_36class:
                output += f"期望知道{label_36class}。"
            else:
                output += "。"

        if output == "该描述。":
            continue

        sft_entry = {
            "instruction": "分析下列描述中提到的医学专有名词并分析其意图：\n\n",
            "input": desc,
            "output": output
        }
        sft_entries.append(sft_entry)

    result_data = remove_repeate(sft_entries)
    # sft_entries = handle_conflict_answers(sft_entries)
    with open(opts.output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='CMID SFT', description='')
    parser.add_argument("--data_dir", type=str, default="/data/xuancheng/cmid/")
    parser.add_argument("--output_file", type=str, default="../../data/cmid.json")
    # 初始化消息
    args = parser.parse_args()
    convert(args)
