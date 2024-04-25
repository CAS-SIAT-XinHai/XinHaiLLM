import argparse
import json
import logging
from itertools import groupby
from operator import itemgetter

from more_itertools import chunked
from tqdm.auto import tqdm
from common.second_handle import *
"""

  {
    "scene": "Communication Challenges",
    "description": "After a long day at work, I received a text from my significant other saying that they need some space. I'm feeling confused and insecure about our relationship.",
    "content": [
      {
        "User": "Hey, are you free? I need someone to talk to."
      },
      {
        "AI Strategy": "",
        "AI": "Of course! I'm here to listen. What's been going on?"
      },
      {
        "User": "I had a rough day at work, and then my significant other texted me out of the blue, saying they need some space. I don't know what to do or how to feel about it."
      },
      {
        "AI Strategy": "Reflective Statements",
        "AI": "That sounds really tough. It's completely normal to feel confused and insecure when a situation like this arises. Have you had any discussions about needing space in the past?"
      },
      {
        "User": "No, we haven't really talked about this before. It just came as a surprise. I'm worried that they're unhappy in our relationship."
      },
      {
        "AI Strategy": "Emotional Validation",
        "AI": "I can understand why you would feel that way. It's natural to jump to conclusions and worry about the state of your relationship. Remember, it's crucial not to blame yourself. Relationships go through ups and downs, and communication is key."
      },
      {
        "User": "You're right, communication is important. I'm just not sure how to bring up the topic without making things worse."
      },
      {
        "AI Strategy": "Suggest Options",
        "AI": "You could consider expressing your feelings to your significant other calmly and honestly. Let them know you're there to listen and that you want to understand their needs and concerns. It's also essential to give them the space they asked for, even if it's difficult for you."
      },
      {
        "User": "I'll try to have a conversation with them when the time feels right. It's just hard to be patient and give them space when I'm feeling so uncertain."
      },
      {
        "AI Strategy": "Empathetic Statements",
        "AI": "I understand how challenging it can be to wait and give someone space when you're feeling unsure. Maybe you can focus on self-care during this time to help you manage your emotions. Doing things you enjoy and spending time with friends and family can provide support."
      },
      {
        "User": "That's a good idea. I'll try to take care of myself and keep busy. It's reassuring to know that I have people like you to talk to as well. Thank you for listening."
      },
      {
        "AI Strategy": "Affirmation",
        "AI": "You're welcome! Remember, you're not alone in this. If you ever need to talk or confront any concerns, feel free to reach out. Your feelings and needs are valid. Take care of yourself during this challenging time."
      },
      {
        "User": "Thank you, I appreciate your kindness and understanding. It helps a lot knowing that I have support. Have a great day!"
      },
      {
        "AI Strategy": "",
        "AI": "You're welcome! I'm here for you. Take care and have a great day too!"
      },
      {
        "User": "Goodbye!"
      },
      {
        "AI Strategy": "",
        "AI": "Goodbye!"
      }
    ]
  },
"""

strategies = [
    "[Reflective Statements (RS)]",
    "[Clarification (Cla)]",
    "[Emotional Validation (EV)]",
    "[Empathetic Statements (ES)]",
    "[Affirmation (Aff)]",
    "[Offer Hope (OH)]",
    "[Avoid Judgment and Criticism (AJC)]",
    "[Suggest Options (SO)]",
    "[Collaborative Planning (CP)]",
    "[Provide Different Perspectives (PDP)]",
    "[Reframe Negative Thoughts (RNT)]",
    "[Share Information (SI)]",
    "[Normalize Experiences (NE)]",
    "[Promote Self-Care Practices (PSP)]",
    "[Stress Management (SM)]",
    "[Others (Oth)]"
]


def convert(opts):
    with open(f"{opts.data_dir}/ExTES.json", encoding='utf-8') as fd:
        data = json.load(fd)

    output = []
    for entry in tqdm(data):
        convs = []
        for item in entry['content']:
            role = 'User' if 'User' in item else 'AI'
            item['role'] = role
            item['text'] = item.pop(role) if role in item else ""
            if isinstance(item['text'], dict):
                item.update(item['text'])
                item['text'] = item['Response']
            convs.append(item)

        conversations = []
        for k, g in groupby(
                enumerate(convs), key=lambda x: x[1]['role']
        ):
            conversations.append([k, list(map(itemgetter(1), g))])

        # for group in consecutive_groups(item['dialog'], ordering=lambda x: x['speaker']):
        if 'User' != conversations[0][0]:
            conversations.insert(0, ['User', [{"text": "", "role": "User"}]])
        if 'AI' != conversations[-1][0]:
            conversations.append(['AI', [{"text": "Bye!", "role": "AI"}]])

        history = []
        for i, d in enumerate(chunked(conversations, n=2, strict=True)):
            (role1, conv1), (role2, conv2) = d
            # print(conv2)
            # strategy = conv2.get("AI Strategy")

            assert role1 == 'User'
            assert role2 == 'AI'

            content_1 = ' '.join([c['text'] for c in conv1]).strip()
            content_2 = ' '.join([c['text'] for c in conv2]).strip()
            if i == (len(conversations) - 2) / 2:
                sft_entry = {
                    "instruction": content_1,
                    "input": "",
                    "output": content_2,
                    "history": history.copy()
                }
                # if opts.use_system:
                #     sft_entry = {
                #         "system": """作为心理学专业毕业的职业心理咨询师，你在对咨询者进行心理治疗，请在回答咨询者问题时，采用特殊标记说明每段话的作用。使用的标记和含义如下：
                # [SUP]：向咨询者提供具有同理心的支持，
                # [ANA]：向咨询者解释相关现象的原因，
                # [ADV]：向咨询者提供合理化建议，
                # [FAC]：引用专业的心理学知识进行说明，
                # [RES]：复述咨询者的描述，
                # [EXP]：以自身的经历进行阐述""",
                #         "instruction": m['question'] + m['description'],
                #         "input": "",
                #         "output": answer
                #     }
                # else:
                #     sft_entry = {
                #         "instruction": m['question'] + m['description'],
                #         "input": "",
                #         "output": '\n'.join([v for k, v in d]).strip(),
                #     }
                output.append(sft_entry)
            history.append([content_1, content_2])
    result_data = remove_repeate(output)
    with open(opts.output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='ExTES SFT', description='')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/ExTES")
    parser.add_argument("--output_file", type=str, default="../../data/extes.json")
    parser.add_argument("--use_system", action='store_true')
    # 初始化消息
    args = parser.parse_args()
    convert(args)
