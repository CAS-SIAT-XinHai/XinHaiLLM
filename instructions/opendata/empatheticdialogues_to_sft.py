import argparse
import json
import logging

from datasets import load_dataset
from second_handle import *

'''
{
        "conv_id": "hit:0_conv:0",
        "utterance_idx": 1,
        "context": "guilty",
        "prompt": "I felt guilty when I was driving home one night and a person tried to fly into my lane_comma_ and didn't see me. I honked and they swerved back into their lane_comma_ slammed on their brakes_comma_ and hit the water cones.",
        "speaker_idx": 0,
        "utterance": "Yeah about 10 years ago I had a horrifying experience. It was 100% their fault but they hit the water barrels and survived. They had no injuries but they almost ran me off the road.",
        "selfeval": "2|2|5_5|5|5",
        "tags": ""
}
'''


def is_empathetic(scores):
    try:
        a, b = scores.split("_")
        empathy_a, relevance_a, fluency_a = list(map(int, a.split("|")))
        empathy_b, relevance_b, fluency_b = list(map(int, b.split("|")))
        return empathy_a + empathy_b > 6
    except:
        print(scores)
        return False


def convert(opts):
    # with open(f"{opts.data_dir}/empathetic_dialogues-test.json", 'r', encoding='utf-8') as file:
    #     dataset = json.load(file)

    dataset = load_dataset("empathetic_dialogues", split='train')

    formatted_data = []

    idx = 0
    conversation_pairs = []

    while idx < len(dataset):
        entry = dataset[idx]

        # instruction = entry['prompt'].replace('_comma_', ',')
        input_text = entry['utterance'].replace('_comma_', ',')

        # 如果下一个 entry 存在，且 conv_id 相同，则获取 output_text
        if idx + 1 < len(dataset) and dataset[idx + 1]['conv_id'] == entry['conv_id']:
            next_entry = dataset[idx + 1]
            output_text = next_entry['utterance'].replace('_comma_', ',')
            conversation_pairs.append([input_text, output_text])
            idx += 2
        else:
            output_text = ""
            idx += 1

        if output_text and is_empathetic(entry['selfeval']):
            history = list(conversation_pairs[:-1]) if output_text else list(conversation_pairs)
            formatted_entry = {
                "instruction": input_text,
                "input": '',
                "output": output_text,
                "history": history
            }
            # 如果下一个 entry 的 conv_id 不同，重置 conversation_pairs
            if idx < len(dataset) and dataset[idx]['conv_id'] != entry['conv_id']:
                formatted_data.append(formatted_entry)
                conversation_pairs = []
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
    parser = argparse.ArgumentParser(prog='Empathetic Dialogues SFT', description='')
    parser.add_argument("--output_file", type=str, default="../../data/empatheticdialogues.json")
    opts = parser.parse_args()
    convert(opts)
