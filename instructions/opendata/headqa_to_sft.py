import argparse
import json
import logging

from datasets import load_dataset
from tqdm import tqdm
from second_handle import *

"""
{
'qid': '1', 
'category': 'biology', 
'qtext': 'Los potenciales postsinápticos excitadores:',
'answers': [
    {
        'aid': 1, 
        'atext': 'Son de tipo todo o nada.'
    }, 
    {
        'aid': 2, 
        'atext': 'Son hiperpolarizantes.'
    },
    {
        'aid': 3, 
        'atext': 'Se pueden sumar.'
    },
    {
        'aid': 4, 
        'atext': 'Se propagan a largas distancias.'
    },
    {
        'aid': 5, 
        'atext': 'Presentan un periodo refractario.'
    }],
'ra': '3',
'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=675x538 at 0x1B42B6A1668>,
'name': 'Cuaderno_2013_1_B',
'year': '2013'
}
"""


def convert(opts):
    dataset = load_dataset('head_qa', 'en', split='train')

    formatted_data = []
    for entry in tqdm(dataset):
        prompt = f"This is a question about {entry['category']} from exams to access a specialized position in healthcare system.\n\n"
        answer = chr(ord('A') + int(entry['ra']) - 1)
        output = f"The answer is {answer} ."

        question = f"{entry['qtext']}\n"
        options = entry['answers']
        for option in options:
            o = chr(ord('A') + int(option['aid']) - 1)
            question += f"{o}: {option['atext']}\n"

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
    parser = argparse.ArgumentParser(prog='HeadQA', description='')
    parser.add_argument("--output_file", type=str, default="../../data/headqa.json")
    args = parser.parse_args()
    convert(args)
