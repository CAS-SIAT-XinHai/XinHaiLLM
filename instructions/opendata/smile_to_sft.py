import argparse
import json
import logging
from collections import OrderedDict

from more_itertools import chunked
from tqdm.auto import tqdm
from second_handle import *
"""
数据集说明：数据集通过 ChatGPT 改写真实的心理互助 QA 为多轮的心理健康支持多轮对话（single-turn to multi-turn inclusive language expansion via ChatGPT），
该数据集含有 56k 个多轮对话，其对话主题、词汇和篇章语义更加丰富多样，更加符合在长程多轮对话的应用场景。数据目录说明如下：

|-- data
  |-- plain (5k对话)
  |-- smile (56k对话，使用PsyQA生成了一轮)
  |-- smile_cot (5k对话)

"""


def convert(opts):
    # "prompt": "instruction",
    # "query": "input",
    # "response": "output",
    # "history": "history"
    with open(f"{opts.smile_dir}/src/train_data/train.json") as fd:
        data = json.load(fd)

    # data = [
    #     {
    #         "instruction": "",
    #         "input": "求助者：我最近感到很迷茫。高三后的我，不仅感到高考前的恐惧，还感到从自己出生以来就是多余的，毫无意义。我没有朋友，只能一个人默默承受。您能给我一些建议吗？支持者：看到你的话，我感到你很焦虑，这是常见的高中生活状态。首先想说的是，我看到了你的决心。这点是很多人没有的。高考最重要的是心态，你需要具备必胜的心态！求助者：谢谢你的鼓励！我真的很想放松一下，但是也觉得自己不能失去目标和动力。支持者：有目标就好办，计划！制定一个时间合理配置的复习计划非常重要，这可以帮助你缓解焦虑。如果你不知道怎样制定计划，我可以提供一些建议。求助者：那我应该如何制定计划呢？支持者：从高一开始，一步一步来，起点要低，试试题海战术。先做一些基础练习，逐渐加强难度，这样你会慢慢有成就感。同时，要注意每种类型的题目都要做，多做一些经典题目。这样你就能够更快地理解并解答考题了。求助者：如果我对某些题目无法理解怎么办呢？支持者：",
    #         "output": "别害羞，去问老师。大胆地告诉他们你对这问题的不解，老师会帮助你理解，并解答你的疑问，这是老师应该做的。不用等毕业再叙旧，马上去问吧！"
    #     },
    # ]

    output = []
    for index, item in enumerate(tqdm(data)):
        conversation = item['input']

        i = 0
        d = []
        while i < len(conversation):
            if conversation[i: i + 4] in ['求助者：', '支持者：']:
                key = conversation[i: i + 4]
                d.append([key, ''])
                i += 4
            else:
                d[-1][1] += conversation[i]
                i += 1

        if len(d) % 2 != 0:
            # print(item)
            # print(d)
            # print("Conversation is not in turn!")
            d.pop(0)

        role, instruction = d.pop(-1)
        assert role == '支持者：' and not instruction
        role, instruction = d.pop(-1)
        assert role == '求助者：'
        history = [[u[1], v[1]] for u, v in chunked(d, n=2, strict=True)]

        if len(d) == 0 and index != 0:
            output.append(sft_entry)

        sft_entry = {
            "instruction": instruction,
            "input": "",
            "output": item['output'],
            "history": history
        }
        # output.append(sft_entry)
        # pre_instruction = instruction
        # pre_output = item['output']
        # pre_history = history
    result_data = remove_repeate(output)
    with open(opts.output_file, 'w') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM', description='')
    parser.add_argument("--smile_dir", type=str, default="/data/datasets/AI4Psychology/smile")
    parser.add_argument("--output_file", type=str, default="../../data/smile.json")
    # 初始化消息
    args = parser.parse_args()
    convert(args)
