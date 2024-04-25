import argparse
import json
import logging
import re
import ast

from datasets import load_dataset
from tqdm import tqdm
from second_handle import *

"""
{
    "exam_type": "医师考试",
    "exam_class": "执业医师",
    "exam_subject": "口腔执业医师",
    "question": "患者，男性，11岁。近2个月来时有低热（37～38℃），全身无明显症状。查体无明显阳性体征。X线检查发现右肺中部有一直径约0.8cm类圆形病灶，边缘稍模糊，肺门淋巴结肿大。此男孩可能患",
    "answer": "D",
    "question_type": "单项选择题",
    "option": {
        "A": "小叶型肺炎",
        "B": "浸润性肺结核",
        "C": "继发性肺结核",
        "D": "原发性肺结核",
        "E": "粟粒型肺结核"
    }
}
"""


def convert(opts):
    dataset = load_dataset("FreedomIntelligence/CMB", 'exam', split='train')
    formatted_data = []
    for entry in tqdm(dataset):
        # prompt = f"该问题源自{entry['exam_type']}，面向{entry['exam_class']}中的{entry['exam_subject']}。\n\n"
        prompt = f"以下是关于{entry['exam_type']}中，{entry['exam_class']}里{entry['exam_subject']}的题目。请根据题目内容，从多个选项中选出最恰当的一个答案。"
        options = entry['option']
        if isinstance(options, str):
            print(f"{options}是字符串，要转为dict再处理")
            options = ast.literal_eval(options)
        # output = f"答案是{entry['answer']}。"
        answer = entry['answer']
        matches = re.findall(r'[A-Z]', answer)
        values_from_dict = [options[match] for match in matches if match in options]
        output_str = "分别是" + "、".join(filter(None, values_from_dict))
        output = f"答案选{answer}，{output_str}。"
        question = f"{entry['question']}\n"
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
    parser = argparse.ArgumentParser(prog='CMB SFT', description='')
    parser.add_argument("--output_file", type=str, default="../../data/cmb.json")
    args = parser.parse_args()
    convert(args)
