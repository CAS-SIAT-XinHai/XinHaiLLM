import argparse
import json
import os.path

import jsonlines
from tqdm import tqdm
from lxml import etree
from common.second_handle import *


def elem2dict(node, attributes=True):
    """
    Convert an lxml.etree node tree into a dict.
    """
    result = {}
    if attributes:
        for item in node.attrib.items():
            key, result[key] = item

    for element in node.iterchildren():
        # Remove namespace prefix
        key = etree.QName(element).localname

        # Process element as tree element if the inner XML contains non-whitespace content
        if element.text and element.text.strip():
            value = element.text
        else:
            value = elem2dict(element)
        if key in result:
            if type(result[key]) is list:
                result[key].append(value)
            else:
                result[key] = [result[key], value]
        else:
            result[key] = value
    return result


def convert_yahoo_xml_to_jsonl(opts):
    with open(os.path.join(opts.yahoo_dir, "FullOct2007.xml")) as f:
        tree = etree.parse(f)

    #with open("small_sample.xml") as f:
    #   tree = etree.parse(f)
    with jsonlines.open(os.path.join(opts.yahoo_dir, 'output.jsonl'), mode='w') as writer:
        for item in tree.getroot().findall('vespaadd'):
            doc = elem2dict(item)['document']
            #print(json.dumps(doc, indent=2))
            writer.write(doc)

def convert(opts):
    with open(os.path.join(opts.data_dir, 'train.json')) as f:
        data = json.load(f)

    sft_entries = []
    with jsonlines.open(os.path.join(opts.yahoo_dir, 'output.jsonl')) as reader:
        for entry in tqdm(reader, total=len(data)):
            if entry['uri'] in data:
                annotation = data[entry['uri']]
                if "Yes" in annotation['support_in_answer']:
                    instruction = entry['content']
                    try:
                        sft_entry = {
                            "instruction": instruction,
                            "input": "",
                            "output": entry['bestanswer']
                        }

                        sft_entries.append(sft_entry)
                    except:
                        print(json.dumps(data[entry['uri']], indent=2))
                        print(entry)

    result_data = remove_repeate(sft_entries)
    # Save JSON data to output file
    with open(f"{opts.output_file}.json", 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)

    print("==================Start generate v2")
    sft_entries_v2 = process_data('llama', result_data)
    with open(f"{opts.output_file}_v2.json", 'w', encoding='utf-8') as json_file_v2:
        json.dump(sft_entries_v2, json_file_v2, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='ESConv to SFT', description='Convert ESConv to SFT based on filtering conditions')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/CHQ-SocioEmo/")
    parser.add_argument("--yahoo_dir", type=str, default="/data/datasets/AI4Psychology/L6-Yahoo/")
    parser.add_argument("--output_file", type=str, default="../../data/chqsocialemo")
    # 初始化消息
    args = parser.parse_args()
    convert(args)

