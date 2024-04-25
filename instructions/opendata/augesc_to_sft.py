import argparse
import json
import logging
from itertools import groupby
from operator import itemgetter

from datasets import load_dataset
from more_itertools import chunked
from tqdm import tqdm
from second_handle import *

"""
{
    'text': '[["usr", "My upstairs neighbors make a ton of noise at all hours of the night. It makes it difficult for me to sleep."], ["sys", "Ah, I understand, there\'s more noise in your area than in other parts of the world?"], ["usr", "I\'m not sure if they\'re directly in my room, but it feels like it."], ["sys", "What sort of noise do you hear? Do you feel that its necessary to live in close proximity to them?"], ["usr", "They have what sounds like parties, loud music playing, someone flushing a toilet at 3am, slamming doors, and shouting loudly."], ["sys", "I can see how it may be difficult to sleep at night with that type of activity going on. What is the relationship between your neighbors and you?"], ["usr", "The only thing I\'ve found that helps a little is having the door closed, but that only works for a short time before they walk in."], ["sys", "Have you tried asking them to turn their music down, or complaining to your landlord?"], ["usr", "I\'ve complained to my landlord but haven\'t had any real success. I think it\'s more of a nuisance to them than it is to me."], ["sys", "I would imagine that they don\'t care much about what you like or do, they\'re just being noisy to get away from their own problems. What is your relationship with your landlord like?"], ["usr", "He\'s never spoken to them in any way, he thinks it\'s their problem not ours."], ["sys", "So you haven\'t had any relationship with him. Is there any way you can get your own space, perhaps some sort of partition?"], ["usr", "I think it\'s unlikely that I will be able to get my own space anytime soon, but I don\'t mind having a good neighbor for now."], ["sys", "I\'m sorry to hear that. Maybe you could tell him that the noise is keeping you awake and ask him to talk to your neighbors to avoid you living in such discomfort."], ["usr", "I don\'t want to ask him to do anything. I feel like I\'m being forced into this situation and I don\'t have any alternatives."], ["sys", "That\'s understandable. Have you tried speaking with your neighbors? They might like to make amends."], ["usr", "I called their door and left a message but got no response. The same goes for my neighbors."], ["sys", "I\'m sorry to hear that. Have you tried speaking with your landlord again? It may seem like he\'s not paying attention, but he may just be busy."], ["usr", "My landlord says he will talk to them, but he\'s been a silent enforcer for as long as I\'ve lived here."], ["sys", "I can see why it is hard for you. Is there anything else that I could say to help you feel better?"], ["usr", "It really is, and I appreciate you taking the time to listen to me."], ["sys", "I hope you have a good rest of the evening. :)"]]'
}
"""


def convert(opts):
    dataset = load_dataset("thu-coai/augesc", split='train')
    formatted_data = []
    for i, entry in enumerate(tqdm(dataset)):
        if i == 0:
            print(entry)

        dialog = eval(entry["text"])

        history = []
        conversations = []
        for k, g in groupby(
                enumerate(dialog), key=lambda x: x[1][0]
        ):
            conversations.append([k, list(map(itemgetter(1), g))])
        # for group in consecutive_groups(item['dialog'], ordering=lambda x: x['speaker']):

        if conversations[0][0] != 'usr':
            conversations.insert(0, ['usr', ''])

        for i, d in enumerate(chunked(conversations, n=2)):
            try:
                (role1, conv1), (role2, conv2) = d
            except:
                continue

            conv1 = list(conv1)
            conv2 = list(conv2)

            assert role1 == 'usr'
            assert role2 == 'sys'

            content_1 = ' '.join([c[1] for c in conv1]).strip()
            content_2 = ' '.join([c[1] for c in conv2]).strip()

            if i == (len(conversations) - 2) / 2:
                sft_entry = {
                    "instruction": content_1,  # content from cMedQA_Q.csv
                    "input": "",
                    "output": content_2,  # list of content from cMedQA_A.csv
                    "history": history.copy()
                }
                formatted_data.append(sft_entry)

            history.append([content_1, content_2])


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
    parser = argparse.ArgumentParser(prog='AugESC SFT', description='')
    parser.add_argument("--output_file", type=str, default="../../data/augesc.json")
    args = parser.parse_args()
    convert(args)
