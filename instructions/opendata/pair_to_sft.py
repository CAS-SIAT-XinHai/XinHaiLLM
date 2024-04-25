import argparse
import json
import logging

import pandas as pd
from tqdm import tqdm
from common.second_handle import *
"""
A dataset consisting of brief interactions between counselors and clients portraying different levels of reflective listening skills. 
Each interaction is in English and includes a client prompt, 
i.e., 
a client's statement that is usually given to the counseling trainee, 
paired with counseling responses portraying different levels of reflections skill, 
i.e., low quality, medium quality, and high quality. 
We build the dataset using both expert and crowd-sourced annotators and also leverage conversational data from an MI dataset to obtain additional prompt-response pairs from conversations snippets containing reflections. 


{
  "prompt": "I know I am too big, and I probably should exercise more and eat better, but I am so busy. I\u2019ve got school, homework, and my job at the mall, so I don\u2019t see anywhere to fit it in. Plus, I can\u2019t afford any of those gyms. And none of my friends want to exercise with me. They\u2019re lazier than I am.  ",
  "hq1": "You are starting to think it\u2019s time to do something about your weight, and you know exercise and eating a little better would help. But fitting it in, between school and work, seems almost impossible. The gym isn\u2019t an option, and you can\u2019t think of any friends who would work out with you. But it is something you are starting to think about.",
  "hq2": "You have put a lot of effort into losing weight but it has not paid off. You are starting to feel a bit desperate. Diets don\u2019t seem to work for you. You are looking for something different that might last.\r\n",
  "mq1": "You don't know how you'd fit exercise into your schedule.",
  "lq1": "It's free to exercise at home. Maybe ride your bike or walk. Ask one of your parents to help. Try to eat some salads and fruit. Bring snacks to work and school.",
  "lq2": "Do you have a cheap gym near you, like a Planet Fitness?",
  "lq3": "You always have time for something, and about the lack of money, just work harder or even if you lack money, start training in public places.",
  "lq4": "Your feelings are valid, but I would advise that you take some time in your day for self care. It is important that you are happy with yourself. Where there is a will there is a way.",
  "lq5": "Start with small steps, start exercising at home and then progress. And really consider if you really don't have the time or just need to adjust your priorities."
}
"""


def convert(opts):
    # Read question data into a DataFrame
    df = pd.read_csv(f"{opts.data_dir}/pair_data.csv")

    # Convert grouped DataFrame to a list of JSON entries
    sft_entries = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if index == 0:
            print(json.dumps(row.to_dict(), indent=2))

        for q in ['hq1', 'hq2']:
            sft_entry = {
                "instruction": row['prompt'],
                "input": "",
                "output": row[q],  # list of content from cMedQA_A.csv
            }
            sft_entries.append(sft_entry)

    result_data = remove_repeate(sft_entries)
    # Save JSON data to output file
    with open(f"{opts.output_file}.json", 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)

    print("==================Start generate v2")
    sft_entries_v2 = process_data('llama', result_data)
    with open(f"{opts.output_file}_v2.json", 'w', encoding='utf-8') as json_file_v2:
        json.dump(sft_entries_v2, json_file_v2, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='PAIR SFT', description='')
    parser.add_argument("--data_dir", type=str,  default="/data/datasets/AI4Psychology/PAIR")
    parser.add_argument("--output_file", type=str, default="../../data/pair")
    # Initialize arguments
    args = parser.parse_args()
    convert(args)
