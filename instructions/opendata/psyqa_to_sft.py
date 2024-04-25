import argparse
import json
import logging
import re
from tqdm.auto import tqdm
from second_handle import *

def convert(opts):
    # example = """[QUESTION]相亲介绍的对象，我无法直视他，怎么办？[DESCRIPTION]因为长得不是很好看，站在一起我会觉得非常别扭但其他的暂时还没有反感的地方[LABEL]婚姻,相亲[ANSWER][SUP]您好，给您一个温暖的抱抱。[ANA]虽然是家里安排的相亲，您也没有反对，但是看得出来您对伴侣还是有一定的要求的，只是也许自己不太清楚的意识到，或者是处于一种朦胧的状态，也有可能是自己还没遇到符合自己要求的男性就先被相亲了。您对于相亲对象的态度一方面是因为“长得不是很好看”所以觉得非常别扭，但是同时又没有其他反感的地方，证明您还是对这个男生不是很排斥的，虽然您可能不止于是颜控，但是对长得好看的人有好感或者喜欢这样的人是没有问题的，毕竟追求美好的事物和美感也是我们人的一项本能嘛。[ADV]如果您想和这位男性相处一段时间却又因为觉得别扭而苦恼的话，可以尝试转移一下注意力，不要把重点放在让你觉得别扭的地方，毕竟无论是恋爱还是婚姻，重要的都是两个人相处之间的过程，是否能够好好磨合和对待彼此，而不是因为某一个原因就一时脑热。"""
    # example = """[QUESTION]以前恶搞别人，现在想恶搞自己，死是解脱？[DESCRIPTION]十几年前初出茅庐刚踏入社会的我，因为受委屈看不惯顶撞了上司，被人撂下了一句“看老板整不死你”，现在经历了这种种安排，被放大的生活，成为众人嘲笑恶搞的对象，我觉得对，甚至也想参与恶搞我自己的队伍中去，因为我也很不喜欢以前的我自己。希望看到一些特别令我捧腹的，或者特别可怕的，笑死吓死都行。死了对我来说是个解脱[LABEL]情绪,表达情绪,内疚羞耻[ANSWER]您好！[SUP]看到题目有些好奇，是怎样的恶搞呢？看你的描述，有些心疼，抱抱你！[ANA]到底发生了什么，经历了什么，让你都想恶搞自己呢？！[EXP]十几年前初出茅庐刚踏入社会的我，因为受委屈看不惯顶撞了上司，被人撂下了一句“看老板整不死你”当初具体发生了什么？[ANA]具体的时间、地点、事情还想得起来吗？你因何感受到委屈？除了委屈还有其他感受吗？当初还想到些什么呢？你顶撞上司后，TA的反应怎样？是谁撂下的那句话？TA和你是什么关系？你听到这句话后的反应又是怎样呢？结果怎么啦？发生这件事后，你有没有和你的父母说过？如果有，他们的反应如何？如果没有，你是怎么想的？怎么没有把自己的委屈告诉他们，获得他们的支持？“现在经历了这种种安排，被放大的生活，成为众人嘲笑恶搞的对象”这期间又发生了什么？你记忆中最深刻的事还记得吗？回忆下细节，能想起什么？感受到什么？众人是如何嘲笑恶搞你的？你能回忆起具体的细节吗？你确定他们都是在嘲笑恶搞你吗？你真的能百分百确定他们都是在嘲笑恶搞你吗？有没有对你友好的人？或者你能回忆起没被人嘲笑恶搞的事吗？十多年来一直是这样吗？有没有不一样的情况？[RES]“我觉得对，甚至也想参与恶搞我自己的队伍中去，因为我也很不喜欢以前的我自己。”什么导致你这么想的？[ANA]什么时候开始想参与到恶搞自己的？当初又发生了什么？不喜欢以前的自己，那对现在的自己你是怎么看的呢？️我们可以不喜欢以前的自己，但我们可以珍惜现在的自己，更可以为未来更好的自己做些什么！[RES]“希望看到一些特别令我捧腹的，或者特别可怕的，笑死吓死都行。[ANA]死了对我来说是个解脱”死是一种解脱，但也可以是一种逃避，你希望的死法有点两极化，你生活中是否也认为只有好坏之分呢？我问了这么多，不知道能否让你想到些什么？也许我无法感同身受你的心情和感受，但能在这里看到你问题，和你聊一聊，是希望能让你知道这世界上并不是所有的人都会嘲笑你、恶搞你！[ADV]如果可以，建议你找专业咨询师或打公益心理热线聊一聊，帮助你缓解下想恶搞自己的心情！️请记住，任何时候我们要学会爱护自己！珍惜自己！[SUP]再次抱抱你！祝好。"""
    with open(f"{opts.data_dir}/PsyQA_generation_split/large_span_enstra_PsyQA_train.json") as fd:
        data = json.load(fd)
    # data = [example]

    # "prompt": "instruction",
    # "query": "input",
    # "response": "output",
    # "history": "history"
    special_tokens = [
        "[QUESTION]",  # question
        "[DESCRIPTION]",  # description
        "[LABEL]",  # label
        "[ANSWER]",  # answer
    ]

    pat = re.compile(
        r"\[QUESTION\](?P<question>.*)\[DESCRIPTION\](?P<description>.*)\[LABEL\](?P<label>.*)\[ANSWER\](?P<answer>.*)")

    strategy_token = {
        "[SUP]",  # support
        "[ANA]",  # analysis
        "[ADV]",  # advice
        "[FAC]",
        "[RES]",
        "[EXP]"
    }

    output = []
    for item in tqdm(data):
        # print(item)
        m = pat.match(item)
        # print(m.groupdict())
        answer = m['answer']
        i = 0
        d = [['_', '']]
        while i < len(answer):
            if answer[i: i + 5] in strategy_token:
                key = answer[i: i + 5]
                d.append([key, ''])
                i += 5
            else:
                d[-1][1] += answer[i]
                i += 1

        if opts.use_system:
            sft_entry = {
                "system": """作为心理学专业毕业的职业心理咨询师，你在对咨询者进行心理治疗，请在回答咨询者问题时，采用特殊标记说明每段话的作用。使用的标记和含义如下：
[SUP]：向咨询者提供具有同理心的支持，
[ANA]：向咨询者解释相关现象的原因，
[ADV]：向咨询者提供合理化建议，
[FAC]：引用专业的心理学知识进行说明，
[RES]：复述咨询者的描述，
[EXP]：以自身的经历进行阐述""",
                "instruction": m['question'] + m['description'],
                "input": "",
                "output": answer
            }
        else:
            sft_entry = {
                "instruction": m['question'] + m['description'],
                "input": "",
                "output": '\n'.join([v for k, v in d]).strip(),
            }
        output.append(sft_entry)

    result_data = remove_repeate(output)
    # Save JSON data to output file
    with open(f"{opts.output_file}.json", 'w', encoding='utf-8') as json_file:
        json.dump(result_data, json_file, ensure_ascii=False, indent=4)

    print("==================Start generate v2")
    sft_entries_v2 = process_data('qwen', result_data)
    with open(f"{opts.output_file}_v2.json", 'w', encoding='utf-8') as json_file_v2:
        json.dump(sft_entries_v2, json_file_v2, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='PsyQA SFT', description='')
    parser.add_argument("--data_dir", type=str, default="/data/datasets/AI4Psychology/PsyQA")
    parser.add_argument("--output_file", type=str, default="../../data/psyqa")
    parser.add_argument("--use_system", action='store_true')
    # 初始化消息
    args = parser.parse_args()
    convert(args)
