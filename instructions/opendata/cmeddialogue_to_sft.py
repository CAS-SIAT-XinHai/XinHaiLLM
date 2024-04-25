# -*- coding: utf-8 -*-
# @Time    : 2023/9/8 8:56
# @Author  : TimLeo
# @FileName: cmeddialogue_to_sft.py
# @Software: PyCharm
import argparse
import glob
import json
import logging
import os

import pandas as pd
from tqdm import tqdm
from second_handle import *


def convert_csv_to_json(csv_file):
    # Read the CSV file into a DataFrame
    data_df = pd.read_csv(csv_file, encoding='gb18030')

    # Prepare the JSON data
    json_data = []
    for i, row in tqdm(data_df.iterrows(), desc=csv_file, total=data_df.shape[0]):
        if i == 0:
            print(row)
        sft_entry = {
            "instruction": row['title'],  # Use 'title' column as instruction
            "input": row['ask'],  # Use 'ask' column as input
            "output": row['answer']  # Use 'answer' column as output
        }
        json_data.append(sft_entry)

    return json_data


def convert_csvs_to_jsons(input_dir, output_file):
    # Initialize a list to store JSON data
    json_data_list = []

    # Process each CSV file and convert to JSON
    for csv_file in glob.glob(os.path.join(input_dir, "Data_数据/**/*.csv")):
        csv_path = os.path.join(input_dir, csv_file)
        json_data_list.extend(convert_csv_to_json(csv_path))

    result_data = remove_repeate(json_data_list)
    # Save JSON data to the output file
    with open(f"{output_file}.json", 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)

    print("==================Start generate v2")
    sft_entries_v2 = process_data('qwen', result_data)
    with open(f"{output_file}_v2.json", 'w', encoding='utf-8') as json_file_v2:
        json.dump(sft_entries_v2, json_file_v2, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(description='Convert CSV files to JSON')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/Chinese-medical-dialogue-data")
    parser.add_argument("--output_file", type=str, default="../../data/cmeddialogue", help="Output JSON file")
    args = parser.parse_args()

    convert_csvs_to_jsons(args.data_dir, args.output_file)
