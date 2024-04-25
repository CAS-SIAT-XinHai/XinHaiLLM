import argparse
import json
import jsonlines
import logging
import os.path

from datasets import load_dataset
from tqdm import tqdm
from second_handle import *
"""
{
    "questionType": "配伍选择题",
    "questionId": 1005674,
    "questionText": "可能导致牙龈出血的药物是",
    "optionImg": [],
    "questionImg": [],
    "option": ["拉西地平", "依诺肝素", "普伐他汀", "普萘洛尔", "呋塞米"], 
    "answer": ["2"],
    "audiourl": "",
    "subject": "药学专业知识（二）", 
    "backgroundText": "", 
    "time": "20200227",
    "q_s": "", 
    "context_s": [],
    "option_s": [],
    "context": [
        "CCB具有很强的血管选择性，CCB中的硝苯地平、氨氯地平、非洛地平和拉西地平用于冠心病和高血压的治疗。左氨氯地平是氨氯地平的左旋体，阻滞钙通道的活性是消旋体的2倍。尼莫地平多用于缺血性脑血管病、偏头痛、脑血管痉挛。氨氯地平、非洛地平、尼卡地平和尼索地平等二代CCB或硝苯地平长效制剂，具有较长的作用时间，每天可以给药1次或2次，能减少心绞痛的发作，其有效性和安全性已在多项研究中得到了证实。",
        "2.抗凝血药与可能引起出血倾向的药物合用，增加出血风险。如肝素与华法林合用，可导致严重的因子X缺乏而出血。",
        "普伐他汀（Pravastatin）是在洛伐他汀的基础上将内酯环开环成3，5 - 二羟基戊酸，通常与钠成盐，以及将十氢萘环3位的甲基用羟基取代而得的药物。普伐他汀比洛伐他汀具有更大的亲水性，这种亲水性增加的优点是减少了药物进入亲脂性细胞，对肝组织有更好的选择性，从而减少了洛伐他汀偶尔出现的副作用。临床上用于治疗高脂血症、家族性高胆固醇血症。",
        "阿托品与吗啡合用，可减轻后者所引起的平滑肌痉挛而加强镇痛作用。普萘洛尔与硝酸酯类产生抗心绞痛的协同作用，并抵消或减少各自的不良反应。普萘洛尔与硝苯地平联用，可提高抗高血压疗效，并对劳力型和不稳定型心绞痛有较好疗效；普萘洛尔与阿托品合用，阿托品可消除普萘洛尔所致心动过缓，普萘洛尔也可消除阿托品所致心动过速。",
        "【注意事项】对磺胺过敏者，可能对布美他尼或呋塞米过敏；严重的磺胺过敏者可以选择依他尼酸作为袢利尿剂的替代药物。其他见呋塞米。"
    ]
}
"""


def convert(opts):
    formatted_data = []
    with jsonlines.open(os.path.join(opts.data_dir, f"NLPEC_public_data/train.json")) as f:
        dataset = [item for item in f]

    for i, entry in enumerate(tqdm(dataset)):
        if i == 0:
            print(json.dumps(entry, indent=2, ensure_ascii=False))

        if 'subject' in entry:
            prompt = f"这是国家药品监督管理局执业药师资格认证考试的题目。\n源自{entry['subject']}\n\n"
        else:
            prompt = f"这是国家药品监督管理局执业药师资格认证考试的题目。\n\n"

        assert len(entry['option']) == len(entry['context'])

        if len(set(entry['context'])) == 1:
            explain = entry['context'][0]
        else:
            explain = '\n'.join([f"{o}：{c}" for o, c in zip(entry['option'], entry['context'])])
        a = ''.join([chr(ord('A') + int(c) - 1) for c in entry['answer']])
        output = f"{explain}\n答案是{a}。"
        options = entry['option']
        question = f"{entry['questionText']}\n"
        for j, option in enumerate(options):
            o = chr(ord('A') + j)
            question += f"{o}: {option}\n"

        formatted_entry = dict(instruction=prompt,
                               input=question,
                               output=output)

        formatted_data.append(formatted_entry)

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
    parser = argparse.ArgumentParser(prog='NLPEC', description='')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/NLPEC")
    parser.add_argument("--output_file", type=str, default="../../data/nlpec.json")
    args = parser.parse_args()
    convert(args)
