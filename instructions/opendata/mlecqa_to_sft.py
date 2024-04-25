import argparse
import json
import logging
import os.path

from datasets import load_dataset
from tqdm import tqdm
from second_handle import *
"""
MLEC-QA is composed of 5 subsets with 136,236 Chinese multi-choice biomedical questions with extra materials (images or tables) annotated by human experts, and covers the following biomedical sub-fields:

Clinic (临床): Clinic_{train,dev,test}.json
Stomatology (口腔): Stomatology_{train,dev,test}.json
Public Health (公共卫生): PublicHealth_{train,dev,test}.json
Traditional Chinese Medicine (中医): TCM_{train,dev,test}.json
Traditional Chinese Medicine Combined with Western Medicine (中西医结合): CWM_{train,dev,test}.json

（一）A1型题（单句型最佳选择题）
（二）A2型题（病例摘要型最佳选择题）
（三）A3型题（病例组型最佳选择题）
（四）A4型题（病例串型最佳选择题）
（五）B1型题（标准配伍题）

{
    "qid": "13413fc4-61a8-4e0c-b24f-01634e6262c6",
    "qtype": "A1型题",
    "qtext": "治愈后，遗留有萎缩性瘢痕的皮肤病是（　　）。",
    "options": {"A": "白秃疮", "B": "肥疮", "C": "鹅掌风", "D": "疥疮", "E": "白疕"},
    "answer": "B"
}
"""

question_types = {
    "A1型题": "单句型最佳选择题",
    "A2型题": "病例摘要型最佳选题",
    "A3型题": "病例组型最佳选择",
    "A4型题": "病例串型最佳选择",
    "B1型题": "标准配伍题"
}


def convert(opts):
    formatted_data = []
    for name in ['Clinic', 'Stomatology', 'PublicHealth', 'TCM', 'CWM']:
        with open(os.path.join(opts.data_dir, f"{name}_train.json"), encoding='utf-8') as f:
            dataset = json.load(f)

        for i, entry in enumerate(tqdm(dataset)):
            if i == 0:
                print(json.dumps(entry, indent=2))

            prompt = f"这是国家医师资格考试的题目。\n\n"
            output = f"答案是{entry['answer']}。"

            options = entry['options']

            question = f"{entry['qtext']}\n"
            for option in sorted(options.keys()):
                question += f"{option}: {options[option]}\n"

            formatted_entry = dict(instruction=prompt,
                                   input=question,
                                   output=output)

            formatted_data.append(formatted_entry)

    result_data = remove_repeate(formatted_data)
    with open(opts.output_file, 'w', encoding='utf-8') as out_file:
        json.dump(result_data, out_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='MLECQA', description='')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/mlec-qa")
    parser.add_argument("--output_file", type=str, default="../../data/mlecqa.json")
    args = parser.parse_args()
    convert(args)
