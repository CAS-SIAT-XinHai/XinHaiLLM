import argparse
import json
import logging

import pandas as pd
from common.second_handle import *
"""

"""


def convert(opts):
    df_articles = pd.read_csv(f"{opts.data_dir}/articles_adobe_AMT.csv", encoding='utf-8')
    df_train_tags = pd.read_csv(f"{opts.data_dir}/messages_train_sentencized_automatic_emotion_tags.tsv",
                                sep='\t',
                                encoding='utf-8')
    df_train = pd.read_csv(f"{opts.data_dir}/messages_train_ready_for_WS.tsv",
                           sep='\t', encoding='utf-8')

    print(df_articles.head())
    print(df_train_tags.head())
    print(df_train.columns)
    print(df_train.head())
    print(df_train.emotion.unique())

    merged_df = pd.merge(df_train, df_articles, on='article_id', how='inner')
    print(merged_df.head())

    # Convert grouped DataFrame to a list of JSON entries
    sft_entries = []
    for index, row in merged_df.iterrows():
        if row['emotion'] in ['sadness', 'neutral', 'fear', 'anger', 'disgust', 'surprise', 'joy'] and row[
            'empathy_bin'] == 1:
            sft_entry = {
                "instruction": f"{row['text']} After reading the news, how do you feel?",  # content from cMedQA_Q.csv
                "input": "",
                "output": row['essay'],  # list of content from cMedQA_A.csv
            }
            sft_entries.append(sft_entry)

    result_data = remove_repeate(sft_entries)
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
    parser = argparse.ArgumentParser(prog='WASSA2022 SFT', description='')
    parser.add_argument("--data_dir", type=str, default="/data/xuancheng/wassa2022")
    parser.add_argument("--output_file", type=str, default="../../data/wassa2022")
    # Initialize arguments
    args = parser.parse_args()
    convert(args)
