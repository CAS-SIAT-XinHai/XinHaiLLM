import argparse
import json
import logging
import os

import jsonlines
from tqdm import tqdm
from second_handle import *
"""
{
    "question": "A 3900-g (8.6-lb) male infant is delivered at 39 weeks' gestation via spontaneous vaginal delivery. Pregnancy and delivery were uncomplicated but a prenatal ultrasound at 20 weeks showed a defect in the pleuroperitoneal membrane. Further evaluation of this patient is most likely to show which of the following findings?",
    "answer": "Gastric fundus in the thorax",
    "options": {
        "A": "Gastric fundus in the thorax",
        "B": "Pancreatic ring around the duodenum",
        "C": "Small and cystic kidneys",
        "D": "Hypertrophy of the gastric pylorus",
        "E": "Large bowel in the inguinal canal"
    },
    "meta_info": "step1",
    "answer_idx": "A"
}
"""


def convert(opts):
    formatted_data = []
    for t in ['Mainland', 'Taiwan', 'US']:
        with jsonlines.open(os.path.join(opts.data_dir, "questions", t, "train.jsonl")) as f:
            dataset = [item for item in f]
            for entry in tqdm(dataset):
                answer = entry['answer_idx']
                if t == 'Mainland':
                    prompt = f"这是国家医学委员会考试的题目。\n\n"
                    output = f"答案是{answer}。"
                elif t == 'Taiwan':
                    prompt = f"這是台灣醫師資格考試的題目。\n\n"
                    output = f"答案是{answer}。"
                else:
                    prompt = f"This is a question about medical board exams in US.\n\n"
                    output = f"The answer is {answer} ."

                question = f"{entry['question']}\n"
                options = entry['options']
                for i in range(len(options)):
                    o = chr(ord('A') + i)
                    question += f"{o}: {options[o]}\n"

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
    parser = argparse.ArgumentParser(prog='MedQAJin', description='')
    parser.add_argument("--data_dir", type=str, default="/data/xuancheng/MEDQA-data")
    parser.add_argument("--output_file", type=str, default="../../data/medqajin.json")
    args = parser.parse_args()
    convert(args)
