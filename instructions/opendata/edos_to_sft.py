import argparse
import json
import os

import pandas as pd
from tqdm import tqdm
from second_handle import *

emotions = [
    'angry', 'furious', 'prepared', 'trusting', 'confident', 'hopeful', 'caring',
    'sentimental', 'anticipating', 'surprised', 'ashamed', 'sad', 'nostalgic',
    'devastated', 'terrified', 'embarrassed', 'lonely', 'content', 'afraid',
    'impressed', 'apprehensive', 'proud', 'annoyed', 'anxious', 'grateful', 'excited', 'neutral',
    'faithful', 'guilty', 'disgusted', 'disappointed', 'jealous', 'joyful'
]

empathetic_response_intents = [
    'acknowledging',
    'agreeing',
    'consoling',
    'encouraging',
    'questioning',
    'sympathizing',
    'suggesting',
    'wishing'
]


def is_empathetic(emotion):
    return emotion in empathetic_response_intents


def sft_convert(opts):
    csv_file = os.path.join(opts.data_dir, "Data/EDOS/EDOS 1M.csv")
    json_file = opts.output_file
    # 读取CSV文件中的数据
    df = pd.read_csv(csv_file)
    # 统计条数
    count = 0

    # 初始化SFT数据列表
    sft_data = []

    # 对每条记录进行处理
    for dialogue_id, group in tqdm(df.groupby('dialogue_id')):
        # 获取对话内容
        turn = group['turn'].tolist()
        dialog = group['uttr'].tolist()
        emotion_list = group['eb+_emot'].tolist()

        if any([is_empathetic(e) for e in emotion_list]):
            # 初始化历史记录列表
            history = []

            # 对每个句子进行处理
            for i in range(0, len(dialog), 2):
                # 获取当前句子和下一个句子
                # instruction = emotion_list[i] + f": {turn[i]} " + dialog[i]
                # output = emotion_list[i + 1] + f": {turn[i + 1]}" + dialog[i + 1] if i + 1 < len(dialog) else ''
                instruction = dialog[i]
                output = dialog[i + 1] if i + 1 < len(dialog) else ''

                # 检查instruction和output是否都有效
                if instruction and output:
                    if is_empathetic(emotion_list[i + 1]):
                        # 生成SFT格式的数据
                        sft_entry = {
                            'instruction': instruction,
                            'input': '',
                            'output': output,
                            'history': history.copy()
                        }

                        # 将SFT数据添加到列表中
                        sft_data.append(sft_entry)
                        # 增加计数
                        count += 1

                    # 更新历史记录列表
                    history.append([instruction, output])

    result_data = remove_repeate(sft_data)
    # 将SFT数据保存到JSON文件中
    with open(json_file, 'w') as f:
        json.dump(result_data, f, indent=4)

    # 输出统计数据
    print(count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='edos-to-sft', description='')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/EDOS")
    parser.add_argument("--output_file", type=str, default="../../data/edos.json")
    # Initialize arguments
    args = parser.parse_args()
    # 调用sft_convert函数，将CSV文件中的数据转换为SFT格式并保存到JSON文件中
    sft_convert(args)
