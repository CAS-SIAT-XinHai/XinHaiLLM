# -*- coding: utf-8 -*-
# @Time    : 2023/9/8 10:29
# @Author  : TimLeo
# @FileName: discmed_to_sft.py.py
# @Software: PyCharm

import argparse
import json
import logging
from itertools import groupby
from operator import itemgetter

from datasets import load_dataset
from more_itertools import chunked
from tqdm import tqdm
from common.second_handle import *


def convert(opts):
    dataset = load_dataset("Flmc/DISC-Med-SFT", split='train')
    output_data = []
    for i, entry in enumerate(tqdm(dataset)):
        if i == 0:
            print(entry)

        dialog = entry["conversation"]

        history = []
        conversations = []
        for k, g in groupby(
                enumerate(dialog), key=lambda x: x[1]['role']
        ):
            conversations.append([k, list(map(itemgetter(1), g))])
        # for group in consecutive_groups(item['dialog'], ordering=lambda x: x['speaker']):

        if conversations[0][0] != 'user':
            conversations.insert(0, ['user', ''])

        for i, d in enumerate(chunked(conversations, n=2)):
            try:
                (role1, conv1), (role2, conv2) = d
            except:
                continue

            conv1 = list(conv1)
            conv2 = list(conv2)

            assert role1 == 'user'
            assert role2 == 'assistant'

            content_1 = ' '.join([c['content'] for c in conv1]).strip()
            content_2 = ' '.join([c['content'] for c in conv2]).strip()

            if i == (len(conversations) - 2) / 2:
                sft_entry = {
                    "instruction": content_1,
                    "input": "",
                    "output": content_2,
                    "history": history.copy()
                }
                output_data.append(sft_entry)

            history.append([content_1, content_2])

    result_data = remove_repeate(output_data)
    # 将处理后的数据写入json文件
    with open(opts.output_file, 'w', encoding='utf-8') as out_file:
        json.dump(result_data, out_file, ensure_ascii=False, indent=4)

    print("转换完成，并已保存到DISC-Med-SFT.json文件中。")
    print("生成的JSON文件包含 {} 个item。".format(len(result_data)))


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='DISCMed SFT', description='')
    parser.add_argument("--output_file", type=str, default="../../data/discmed.json")
    args = parser.parse_args()
    convert(args)
