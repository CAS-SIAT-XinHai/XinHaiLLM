import argparse
import json
import logging

import pandas as pda
from tqdm.auto import tqdm
from common.second_handle import *


# data = [
#    {
#        "uuid": 1,
#        "seeker_post": "Help. Help me. I dunno what I'm doing anymore",
#        "response_post": "That's pretty vague, do you not know what you're doing in regards to a specific section of your life? Like school or work?"
#    },
# ]
def convert(opts):
    # "prompt": "instruction",
    # "query": "input",
    # "response": "output",
    # "history": "history"
    # df_reactions = pda.read_csv(f"{opts.emh_dir}/dataset/emotional-reactions-reddit.csv")
    # df_explorations = pda.read_csv(f"{opts.emh_dir}/dataset/explorations-reddit.csv")
    df_interpretations = pda.read_csv(f"{opts.data_dir}/dataset/interpretations-reddit.csv", encoding='utf-8')
    # print(df_reactions.head())
    # print(df_explorations.head())
    # print(df_interpretations.head())

    df = df_interpretations[df_interpretations.level > 1]

    output = []
    for item in tqdm(df.itertuples(), total=df.shape[0]):
        # history_id = f"session_{item['uuid']}"

        sft_entry = {
            "instruction": item.seeker_post,
            "input": "",
            "output": item.response_post,
            # "history": history_id
        }

        output.append(sft_entry)

    result_data = remove_repeate(output)
    # Save JSON data to output file
    with open(f"{opts.output_file}.json", 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)

    print("==================Start generate v2")
    sft_entries_v2 = process_data('llama', result_data)
    sft_entries_v2 = remove_llm_other(sft_entries_v2)
    with open(f"{opts.output_file}_v2.json", 'w', encoding='utf-8') as json_file_v2:
        json.dump(sft_entries_v2, json_file_v2, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM', description='')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/Empathy-Mental-Health")
    parser.add_argument("--output_file", type=str, default="../../data/epitome")
    # 初始化消息
    args = parser.parse_args()
    convert(args)
