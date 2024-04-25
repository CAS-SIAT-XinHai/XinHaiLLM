import argparse
import json
import logging

import pandas as pd
from second_handle import *
"""

PMID                                                   31196764
Sentence      Flibanserin, a multifunctional serotonin recep...
ID_1                                                    C098107
E1                                                  Flibanserin
Type_E1                                                CHEMICAL
Full_E1                                             Flibanserin
MeSH_E1                     flibanserin [Supplementary Concept]
Synonyms_1                                       Benzimidazoles
Relation      be approved in united states and canada for tr...
ID_2                                                   NEW00000
E2            acquired generalized hypoactive sexual desire ...
Type_E2                                           MENTAL_HEALTH
Full_E2       acquired generalized hypoactive sexual desire ...
MeSH_E2       acquired generalized hypoactive sexual desire ...
Synonyms_2                                          No synonyms

"""


def convert(opts):
    # Read question data into a DataFrame
    df = pd.read_csv(f"{opts.data_dir}/gena_data/gena_data_final.zip", compression='zip', encoding='utf-8')

    # Convert grouped DataFrame to a list of JSON entries
    sft_entries = []
    for index, row in df.iterrows():
        if any([pd.isna(row[k]) for k in ['E1', 'E2', 'Sentence']]):
            continue
        sft_entry = {
            "instruction": f"Can you tell me something about {row['E1']} and {row['E2']}",  # content from cMedQA_Q.csv
            "input": "",
            "output": row['Sentence'],  # list of content from cMedQA_A.csv
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
    parser = argparse.ArgumentParser(prog='GENA SFT', description='')
    parser.add_argument("--data_dir", type=str, default="/data/xuancheng/gena-db-master")
    parser.add_argument("--output_file", type=str, default="../../data/gena")
    # Initialize arguments
    args = parser.parse_args()
    convert(args)
