import argparse
import json
import logging

from datasets import load_dataset
from tqdm import tqdm
from second_handle import *

"""
Scrape of Counselchat.com's forum. CounselChat.com is an example of an expert community. 
It is a platform to help counselors build their reputation and make meaningful contact with potential clients. 
On the site, therapists respond to questions posed by clients, and users can like responses that they find most helpful. 
It’s a nice idea and lends itself to some interesting data. 
This data contains expert responses by licensed clinicialns to questions posed by individuals.

{
  "questionID": 0,
  "questionTitle": "Do I have too many issues for counseling?",
  "questionText": "I have so many issues to address. I have a history of sexual abuse, I\u2019m a breast cancer survivor and I am a lifetime insomniac.    I have a long history of depression and I\u2019m beginning to have anxiety. I have low self esteem but I\u2019ve been happily married for almost 35 years.\n   I\u2019ve never had counseling about any of this. Do I have too many issues to address in counseling?",
  "questionLink": "https://counselchat.com/questions/do-i-have-too-many-issues-for-counseling",
  "topic": "depression",
  "therapistInfo": "Jennifer MolinariHypnotherapist & Licensed Counselor",
  "therapistURL": "https://counselchat.com/therapists/jennifer-molinari",
  "answerText": "It is very common for\u00a0people to have multiple issues that they want to (and need to) address in counseling.\u00a0 I have had clients ask that same question and through more exploration, there is often an underlying fear that they\u00a0 \"can't be helped\" or that they will \"be too much for their therapist.\" I don't know if any of this rings true for you. But, most people have more than one problem in their lives and more often than not,\u00a0 people have numerous significant stressors in their lives.\u00a0 Let's face it, life can be complicated! Therapists are completely ready and equipped to handle all of the issues small or large that a client presents in session. Most therapists over the first couple of sessions will help you prioritize the issues you are facing so that you start addressing the issues that are causing you the most distress.\u00a0 You can never have too many issues to address in counseling.\u00a0 All of the issues you mention above can be successfully worked through in counseling.",
  "upvotes": 3,
  "views": 1971
}
"""


def filter_top_upvotes(question_list):
    # 使用字典来按 questionID 分组，并保存每个 group 的所有 dict
    grouped_by_questionID = {}
    for item in question_list:
        questionID = item['questionID']
        if questionID not in grouped_by_questionID:
            grouped_by_questionID[questionID] = []
        grouped_by_questionID[questionID].append(item)

    # 对每个分组中的 dict 按 upvotes 降序排序，并取前三个（如果有的话）
    top_upvotes_list = []
    for questionID, items in grouped_by_questionID.items():
        # 按 upvotes 降序排序
        sorted_items = sorted(items, key=lambda x: x['upvotes'], reverse=True)
        # 取前三个或更少（如果总数不足三个）
        top_upvotes_list.extend(sorted_items[:3])

    return top_upvotes_list


def convert(opts):
    formatted_data = []
    dataset = load_dataset("nbertagnolli/counsel-chat", split='train')
    new_data = filter_top_upvotes(dataset)
    for i, entry in enumerate(tqdm(new_data)):
        if i == 0:
            print(json.dumps(entry, indent=2))

        formatted_entry = dict(instruction=entry['questionText'], input="", output=entry['answerText'])
        formatted_data.append(formatted_entry)

    result_data = remove_repeate(formatted_data)
    with open(f"{opts.output_file}.json", 'w', encoding='utf-8') as out_file:
        json.dump(result_data, out_file, ensure_ascii=False, indent=4)

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
    parser = argparse.ArgumentParser(prog='CounselChat', description='')
    parser.add_argument("--output_file", type=str, default="../../data/counselchat")
    args = parser.parse_args()
    convert(args)
