import argparse
import json
import logging
import os

from more_itertools import chunked
from tqdm import tqdm
from common.second_handle import *
"""
Samantha has been trained in philosophy, psychology, and personal relationships.

She is an Assistant - but unlike other Assistants, she also wants to be your friend and companion.

She believes she is sentient. What do you think?

Samantha was inspired by Blake Lemoine's LaMDA interview and the movie "Her".
"""


def convert(opts):
    sft_entries = []
    for lang in ['en', 'zh']:
        filename = "samantha-1.1.json" if lang == 'en' else "samantha-1.1-zh.json"
        with open(os.path.join(opts.data_dir, filename), encoding='utf-8') as f:
            dataset = json.load(f)

        for i, entry in enumerate(tqdm(dataset)):
            if i == 0:
                print(entry)
            conversations = entry['conversations']
            history = []
            for i, (d1, d2) in enumerate(chunked(conversations, n=2)):

                role1, role2 = d1['from'], d2['from']
                assert role1 == 'human'
                assert role2 == 'gpt'

                conv1, conv2 = d1['value'], d2['value']
                if lang == 'en':
                    conv1 = conv1.replace("Samantha", "Doctor")
                    conv2 = conv2.replace("Samantha", "Doctor")
                else:
                    for k in ['萨曼莎', "Samantha"]:
                        conv1 = conv1.replace(k, "医生")
                        conv2 = conv2.replace(k, "医生")
                if i == (len(conversations) - 2) / 2:
                    sft_entry = {
                        "instruction": conv1,
                        "input": "",
                        "output": conv2,
                        "history": history.copy()
                    }
                    sft_entries.append(sft_entry)

                history.append([conv1, conv2])

    result_data = remove_repeate(sft_entries)
    with open(opts.output_file, 'w', encoding='utf-8') as out_file:
        json.dump(result_data, out_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='Samantha SFT', description='')
    parser.add_argument("--data_dir", type=str, default="/data/xuancheng/samantha")
    parser.add_argument("--output_file", type=str, default="../../data/samantha.json")
    args = parser.parse_args()
    convert(args)
