# -*- coding: utf-8 -*-
# @Time    : 2023/8/30 9:50
# @Author  : TimLeo
# @FileName: efqaq_to_sft.py
# @Software: PyCharm

import argparse
import gzip
import json
import logging
from itertools import groupby
from operator import itemgetter

import jsonlines
from more_itertools import chunked
from tqdm import tqdm
from common.second_handle import *
'''
{
    "chats": [
        {
            "label": {"knowledge": false, "negative": false, "question": true},
            "sender": "audience",
            "time": "11:02:45",
            "type": "textMessage", "value": "这样的议论是针对谁呢？"
        },
        {
            "label": {"knowledge": false, "negative": false, "question": false},
            "sender": "audience",
            "time": "11:08:38",
            "type": "textMessage",
            "value": "我也是一个从小被这样训到大的女生哦，总会被指责缺心少肺、没心眼儿、没眼力见儿、看不出来眉眼高低等等。不过在我成长一段时间之后，发现这件事情其实很简单，也没有什么大的问题。如果你愿意的话，可以找我聊聊，倾诉一下你遇到的事情，希望能够帮到你。我是树洞小太阳，欢迎你来找我玩❤"
        },
        {
            "label": {"knowledge": false, "negative": false, "question": false},
            "sender": "audience",
            "time": "11:15:17",
            "type": "textMessage", "value": "好惨"
        },
        {
            "label": {"knowledge": false, "negative": false, "question": false},
            "sender": "audience",
            "time": "11:15:35",
            "type": "textMessage", "value": "原生家庭也这么对你吗"
        }
    ],
    "date": "2020-03-02 11:01:08",
    "label": {"s1": "1.13", "s2": "2.7", "s3": "3.4"},
    "owner": "匿名",
    "title": "女 听过别人最多的议论就是干啥啥不行不长心眼没有脑子"
}
'''


def is_empathetic(scores):
    try:
        scores['knowledge']
        return (not scores['negative']) or scores['question']
    except:
        print(scores)
        return False


def is_ad(text):
    for ad in [
        '点击我头像',
        '加关注'
    ]:
        if ad in text:
            return True


def is_null(text):
    for s in [
        '私聊',
        '下一位',
        '下一位专家'
    ]:
        if s == text.strip():
            return True


def convert(opts):
    sft_entries = []
    with gzip.open(f"{opts.data_dir}/data/efaqa-corpus-zh.utf8.gz", 'rb') as f:
        with jsonlines.Reader(f) as reader:
            for entry in tqdm(reader):
                conversations = []
                for k, g in groupby(
                        enumerate(entry['chats']), key=lambda x: x[1]['sender']
                ):
                    conversations.append([k, list(map(itemgetter(1), g))])

                # for group in consecutive_groups(item['dialog'], ordering=lambda x: x['speaker']):
                if 'owner' != conversations[0][0]:
                    conversations.insert(0, ['owner', [{"value": entry['title'], "sender": "owner"}]])
                else:
                    conversations[0][1][0]["value"] = entry['title'] + "\n" + conversations[0][1][0]["value"]

                if 'audience' != conversations[-1][0]:
                    # conversations.append(['audience', [{"value": "", "sender": "audience"}]])
                    conversations.pop(-1)

                history = []
                for i, d in enumerate(chunked(conversations, n=2, strict=True)):
                    (role1, conv1), (role2, conv2) = d

                    assert role1 == 'owner'
                    assert role2 == 'audience'

                    content_1 = '\n\n'.join(
                        [c['value'] for c in conv1 if not is_null(c['value']) and not is_ad(c['value'])]).strip()
                    content_2 = '\n\n'.join(
                        [c['value'] for c in conv2 if not is_null(c['value']) and not is_ad(c['value'])]).strip()
                    if i == (len(conversations) - 2) / 2:
                        sft_entry = {
                            "instruction": content_1,
                            "input": "",
                            "output": content_2,
                            "history": history.copy()
                        }
                        sft_entries.append(sft_entry)
                    history.append([content_1, content_2])

                # instruction = entry['title']
                # for chat in entry['chats']:
                #     if is_empathetic(chat['label']) and not is_ad(chat['value']) and len(chat['value']) > 10:
                #         sft_entry = {
                #             "instruction": instruction,
                #             "input": "",
                #             "output": chat['value']
                #         }
                #
                #         sft_entries.append(sft_entry)

    result_data = remove_repeate(sft_entries)
    with open(opts.output_file, 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='efqaq-to-sft', description='')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/efaqa-corpus-zh")
    parser.add_argument("--output_file", type=str, default="../../data/efaqa.json")
    # Initialize arguments
    args = parser.parse_args()
    convert(args)
