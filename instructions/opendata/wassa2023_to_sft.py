import argparse
import json
import logging

import pandas as pd
from more_itertools import chunked
from second_handle import *
"""

"""


def convert(opts):
    df_articles = pd.read_csv(f"{opts.data_dir}/articles_adobe_AMT.csv",
                              encoding='utf-8')
    # df_train_essay = pd.read_csv(f"{opts.data_dir}/WASSA23_essay_level_with_labels_train.tsv",
    #                              sep='\t', encoding='utf-8')
    df_train_conv = pd.read_csv(f"{opts.data_dir}/WASSA23_conv_level_with_labels_train.tsv",
                                sep='\t', encoding='utf-8')

    merged_df = pd.merge(df_train_conv, df_articles, on='article_id', how='inner')
    merged_df.sort_values(['conversation_id', 'turn_id'], inplace=True)
    sft_entries = []
    for item in merged_df.groupby('conversation_id').apply(lambda x: (x.text_y, x.turn_id, x.Empathy, x.text_x)):
        article, turns, empathy, conversations = item
        article = article.tolist()[0]
        empathy = empathy.tolist()
        history = []
        for i, d in enumerate(chunked(zip(empathy, conversations), n=2)):
            try:
                ((emp1, conv1), (emp2, conv2)) = d
            except:
                # print(conversations)
                continue

            if i == 0:
                conv1 = article + conv1
            if i == (len(conversations) - 2) / 2:
                sft_entry = {
                    "instruction": conv1,  # content from cMedQA_Q.csv
                    "input": "",
                    "output": conv2,  # list of content from cMedQA_A.csv
                    "history": history.copy()
                }
                sft_entries.append(sft_entry)

            history.append([conv1, conv2])

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
    parser = argparse.ArgumentParser(prog='WASSA2023 SFT', description='')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/wassa2023")
    parser.add_argument("--output_file", type=str, default="../../data/wassa2023.json")
    # Initialize arguments
    args = parser.parse_args()
    convert(args)
