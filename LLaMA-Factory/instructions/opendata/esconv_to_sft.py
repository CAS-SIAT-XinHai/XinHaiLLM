import argparse
import json
import logging
from enum import Enum
from itertools import groupby
from operator import itemgetter

from more_itertools import chunked
from tqdm.auto import tqdm

from edos_to_sft import empathetic_response_intents
from common.second_handle import *


class SupportStrategy(Enum):
    question = "Question"  # 3,109 20.9%
    paraphrase = "Restatement or Paraphrasing"  # 883 5.9%
    reflection = "Reflection of feelings"  # 1,156 7.8%
    self_disclosure = "Self-disclosure"  # 1,396 9.4%
    affirmation = "Affirmation and Reassurance"  # 2,388 16.1%
    suggestion = "Providing Suggestions"  # 2,323 15.6%
    information = "Information"  # 904 6.1%
    others = "Others"  # 2,696 18.1%

    @classmethod
    def to_dict(cls):
        return {e.name: e.value for e in cls}

    @classmethod
    def items(cls):
        return [(e.name, e.value) for e in cls]

    @classmethod
    def keys(cls):
        return [e.name for e in cls]

    @classmethod
    def values(cls):
        return [e.value for e in cls]


def is_empathetic(emotion):
    return emotion in empathetic_response_intents


def convert(opts):
    # example:{'experience_type': 'Previous Experience', 'emotion_type': 'anxiety', 'problem_type': 'job crisis', 'situation': 'I hate my job but I am scared to quit and seek a new career.', 'survey_score': {'seeker': {'initial_emotion_intensity': '5', 'empathy': '5', 'relevance': '5', 'final_emotion_intensity': '1'}, 'supporter': {'relevance': '5'}}, 'dialog': [{'speaker': 'seeker', 'annotation': {}, 'content': 'Hello\n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Question'}, 'content': 'Hello, what would you like to talk about?'}, {'speaker': 'seeker', 'annotation': {}, 'content': 'I am having a lot of anxiety about quitting my current job. It is too stressful but pays well\n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Question'}, 'content': 'What makes your job stressful for you?'}, {'speaker': 'seeker', 'annotation': {'feedback': '5'}, 'content': 'I have to deal with many people in hard financial situations and it is upsetting \n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Question'}, 'content': 'Do you help your clients to make it to a better financial situation?'}, {'speaker': 'seeker', 'annotation': {}, 'content': 'I do, but often they are not going to get back to what they want. Many people are going to lose their home when safeguards are lifted \n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Affirmation and Reassurance'}, 'content': 'But you offer them a better future than what they have currently. It may not be what they wanted, but it helps them in the long run.'}, {'speaker': 'seeker', 'annotation': {'feedback': '5'}, 'content': 'That is true but sometimes I feel like I should put my feelings and health first \n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Affirmation and Reassurance'}, 'content': 'I can understand that. '}, {'speaker': 'supporter', 'annotation': {'strategy': 'Question'}, 'content': 'Is there another job that would pay you close to what you currently make?'}, {'speaker': 'seeker', 'annotation': {'feedback': '5'}, 'content': 'Probably not. I was with the same company for a long time and I consistently get a bonus every year '}, {'speaker': 'supporter', 'annotation': {'strategy': 'Others'}, 'content': "Is it possible to reframe how you look at your clients' dire financial situations?"}, {'speaker': 'seeker', 'annotation': {}, 'content': 'I could try. It mostly gets to me at the end of the day \n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Information'}, 'content': "Some people can't do what you do because they don't have the heart to give someone else bad news. The reality is though, someone needs to fill that role and you do help people"}, {'speaker': 'seeker', 'annotation': {'feedback': '4'}, 'content': 'That is also true. Sometimes I wonder if it really is for me though  \n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Self-disclosure'}, 'content': "I've had to deal with collections before when I was in  bad financial condition. The person on the other line was really helpful though. She was understanding,"}, {'speaker': 'supporter', 'annotation': {'strategy': 'Providing Suggestions'}, 'content': 'It may not be for you. I think you should think about the pros and cons of keeping your position. It might make things clearer for you. '}, {'speaker': 'seeker', 'annotation': {'feedback': '5'}, 'content': 'That is true. Maybe I just need to sit down and really think about it \n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Restatement or Paraphrasing'}, 'content': "I wouldn't stay if it really impacts your mental health in a negative way. Still, you may need to zoom out and see the bigger picture: that you provide a needed service and you do it compassionately"}, {'speaker': 'seeker', 'annotation': {}, 'content': 'It really is a big decision \n'}, {'speaker': 'seeker', 'annotation': {}, 'content': 'Thank you for the different perspective \n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Affirmation and Reassurance'}, 'content': 'No doubt, but you know in your heart what is right for you. '}, {'speaker': 'seeker', 'annotation': {'feedback': '4'}, 'content': 'That is true. Thanks again \n'}, {'speaker': 'seeker', 'annotation': {}, 'content': 'Bye\n'}, {'speaker': 'supporter', 'annotation': {'strategy': 'Affirmation and Reassurance'}, 'content': "It's no problem. I hope you can make a decision about this situation and then be at peace with it"}, {'speaker': 'supporter', 'annotation': {'strategy': 'Others'}, 'content': 'Ok, take care'}], 'seeker_question1': 'Partner was very supportive', 'seeker_question2': 'More guidance in conversation or examples', 'supporter_question1': '', 'supporter_question2': ''}

    with open(f"{opts.data_dir}/ESConv.json", encoding='gbk', errors='ignore') as fd:
        data = json.load(fd)

    # output = []
    sft_entries = []
    for item in tqdm(data):
        intro = '\"experience_type\":' + item["experience_type"] + ', \"emotion_type\":' + item[
            "emotion_type"] + ', \"problem_type\":' + item["problem_type"] + ', \"situation\":' + item["situation"]
        # print(intro)
        # dialog = item["dialog"]
        # print(dialog)
        # print(len(dialog))
        # d = []
        # i = 0
        # print(json.dumps(item['dialog'], indent=2))
        history = []

        conversations = []
        for k, g in groupby(
                enumerate(item['dialog']), key=lambda x: x[1]['speaker']
        ):
            conversations.append([k, list(map(itemgetter(1), g))])
        # for group in consecutive_groups(item['dialog'], ordering=lambda x: x['speaker']):

        if conversations[0][0] != 'seeker':
            conversations.insert(0, ['seeker', [{'speaker': 'seeker', 'annotation': {}, 'content': ''}]])

        for i, d in enumerate(chunked(conversations, n=2)):
            try:
                (role1, conv1), (role2, conv2) = d
            except:
                continue

            conv1 = list(conv1)
            conv2 = list(conv2)

            assert role1 == 'seeker'
            assert role2 == 'supporter'

            content_1 = ' '.join([c['content'] for c in conv1]).strip()
            content_2 = ' '.join([c['content'] for c in conv2]).strip()
            # if i == 0:
            #     content_1 = intro + content_1
            if i == (len(conversations) - 2) / 2:
                sft_entry = {
                    "instruction": content_1,  # content from cMedQA_Q.csv
                    "input": "",
                    "output": content_2,  # list of content from cMedQA_A.csv
                    "history": history.copy()
                }
                sft_entries.append(sft_entry)

            history.append([content_1, content_2])

    result_data = remove_repeate(sft_entries)
    with open(opts.output_file, 'w') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='ESConv to SFT',
                                     description='Convert ESConv to SFT based on filtering conditions')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/Emotional-Support-Conversation")
    parser.add_argument("--output_file", type=str, default="../../data/esconv.json")
    # 初始化消息
    args = parser.parse_args()
    convert(args)
