import json, tiktoken, re, os
import time
from openai import OpenAI

read_data_path = "/mnt/c/koenshen/SVN/XinHaiLLM_data_and_db/data/result"
save_data_path = "/mnt/c/koenshen/SVN/XinHaiLLM_data_and_db/data/score"


def read_json_files(path=read_data_path):
    # 遍历指定目录下的所有文件
    result_list = []
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            result_list.append(filename)
    return result_list


# 读取某一路径下的json格式，然后统计里面的各项子分
def compute_score_from_dict():
    read_file_list = read_json_files(save_data_path)
    for read_file in read_file_list:
        file_path = f"{save_data_path}/{read_file}"
        with open(file_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        if "therapistqa" in read_file:
            total_score_dict = {'Empathy_score': 0, 'Identification_score': 0, 'Reflection_score': 0,
                                'Strategy_score': 0, 'Encouragement_score': 0, 'Relevance_score': 0}
        else:
            total_score_dict = {'共情分数': 0, '辨识分数': 0, '反思分数': 0, '策略分数': 0, '鼓励分数': 0,
                                '相关性分数': 0}
        single_file_list = []
        for index, json_dict in enumerate(json_list):
            score_str = json_dict["score"].replace('\n', '').replace("得分", "分数").replace('：', '').replace('，', ',').replace('“', '"').replace('”', '"').replace("`", "'").replace("\&quot;", "\"")
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, score_str)
            final_match_str = matches[0]
            try:
                if len(matches) > 1:
                    for _, match_str in enumerate(matches):
                        if "psyqa" in read_file and '分数' in match_str:
                            final_match_str = match_str
                if len(matches) == 0:
                    raise Exception("exception in length of matches")

                data = json.loads(final_match_str.replace('\\', '').replace("\'", "\"").replace("　", "").replace(" ", ""))
                if "therapistqa" in read_file:
                    if float(str(data["Empathy_score"])) > 7 or float(str(data["Identification_score"])) > 7 or float(str(data["Reflection_score"])) > 7 or float(str(data["Strategy_score"])) > 7 or float(str(data["Encouragement_score"])) > 7 or float(str(data["Relevance_score"])) > 7:
                        raise Exception("english exception in score beyond requires max score")
                    total_score_dict["Empathy_score"] += float(str(data["Empathy_score"]))
                    total_score_dict["Identification_score"] += float(str(data["Identification_score"]))
                    total_score_dict["Reflection_score"] += float(str(data["Reflection_score"]))
                    total_score_dict["Strategy_score"] += float(str(data["Strategy_score"]))
                    total_score_dict["Encouragement_score"] += float(str(data["Encouragement_score"]))
                    total_score_dict["Relevance_score"] += float(str(data["Relevance_score"]))
                    single_file_list.append(data)
                else:
                    if "辨识分数" not in data.keys() and "信念分数" in data.keys():
                        data["辨识分数"] = data["信念分数"]

                    if float(str(data["共情分数"])) > 7 or float(str(data["辨识分数"])) > 7 or float(str(data["反思分数"])) > 7 or float(str(data["策略分数"])) > 7 or float(str(data["鼓励分数"])) > 7 or float(str(data["相关性分数"])) > 7:
                        raise Exception("chinese exception in score beyond requires max score")

                    total_score_dict["共情分数"] += float(str(data["共情分数"]))
                    total_score_dict["辨识分数"] += float(str(data["辨识分数"]))
                    total_score_dict["反思分数"] += float(str(data["反思分数"]))
                    total_score_dict["策略分数"] += float(str(data["策略分数"]))
                    total_score_dict["鼓励分数"] += float(str(data["鼓励分数"]))
                    total_score_dict["相关性分数"] += float(str(data["相关性分数"]))
                    single_file_list.append(data)
            except Exception as e:
                print(f"{index}出现异常：{matches}")
        # print(f"{file_path} -> {total_score_dict}")
        modified_data = {}
        for i, (key, value) in enumerate(total_score_dict.items()):
            # 根据索引应用不同的除法操作
            modified_value = value / 100
            modified_data[key] = modified_value
        formatted_dict = {k: f"{v:.3f}" for k, v in modified_data.items()}
        print(f"{file_path} -> {formatted_dict} -> Overall：{sum([float(v) for v in formatted_dict.values()]):.3f}")
        print("==========================================================")

def gpt4(message: list):
    model = os.environ.get("API_MODEL")
    openai_api_key = os.environ.get("API_KEY")
    openai_api_base = os.environ.get("API_BASE")
    if "gpt" not in model:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    else:
        client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(model=model, messages=message)
    return completion.choices[0].message.content

def single_qa_scoring(data_dict, read_file:str):
    # 多轮的history和单轮qa的history可能不一样，按照你觉得合适的方法，组装这个history就行
    if "psyqa" in read_file:
        system_prompt = f'''# 角色：\n您是一位公正的评判员，熟悉心理学知识和心理咨询。\n\n## 注意：\n您的职责是评估AI心理咨询师对患者心理问题的回答质量。您的评估应参照历史对话内容，并仅根据评估标准打分。\n\n## 评估标准：\n### 共情 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的共情程度，即心理咨询师是否在回复内容中表达了对患者焦虑与痛苦的理解和同情，并是否为患者营造了自由表达焦虑与痛苦的安全感。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否正确理解了患者的提问意图？\n* 心理咨询师的回复内容中，是否对患者的焦虑与痛苦表示尊重、理解和同情？\n* 心理咨询师的回复内容中，是否为患者营造了自由表达焦虑与痛苦的安全感？\n\n### 信念 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的认知扭曲程度，即心理咨询师是否在回复内容中识别出了患者焦虑与痛苦背后错误扭曲的认知信念，深入探讨并帮助患者挑战自己错误扭曲的认知信念。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否识别到了患者错误扭曲的认知信念？\n* 心理咨询师的回复内容中，是否深入探讨了患者错误扭曲的认知信念？\n* 心理咨询师的回复内容中，是否帮助患者识别和挑战自己错误扭曲的认知信念？\n\n### 反思 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的反思程度，即心理咨询师是否在回复内容中提出了一些问题，来鼓励患者重新反思考虑他们的初始想法或错误扭曲的认知信念。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否提出了与患者的初始想法相关的问题？\n* 心理咨询师的回复内容中，是否提出了能帮助患者进行深入思考的问题？\n* 心理咨询师的回复内容中，是否提出了能反映患者错误扭曲的认知信念的问题？\n\n### 策略 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的策略可行性，即确定心理咨询师是否在回复内容中，提供了有益于缓解或解决患者当前焦虑与痛苦的策略或见解，来帮助患者解决当前遇到的情况。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，提供的策略或见解是否可行？\n* 心理咨询师的回复内容中，提供的策略或见解能否解决患者当前的问题？\n* 心理咨询师的回复内容中，提供的策略是否为专业的心理治疗方法？\n\n### 鼓励 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的鼓励程度，即心理咨询师是否在回复内容中鼓励患者采用相应的策略或见解，强调这只是开始，鼓励患者一步一步地改变现状。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否鼓励了患者采取行动应对当前的情况？\n* 心理咨询师的回复内容中，是否对患者未来执行策略遇到失败的可能性，进行提醒和分析？\n* 心理咨询师的回复内容中，是否对患者未来执行策略遇到的挫折与痛苦，进行鼓舞和安慰？\n\n### 相关性 (0-7分)：\n您需要确定咨询师的回答与患者的提问之间的相关性。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回答内容与患者的提问内容是否高度相关？\n* 心理咨询师的回答内容是否自然流畅？\n* 心理咨询师的回答是否涵盖了患者提问的主要问题或关注点？\n\n## 约束条件\n- 请您避免任何偏见，保持客观，并确保回答呈现的顺序不会影响您的判断；\n- 请不要让回答的长度影响您的评估；\n\n## 工作流程\n请根据以上六个评估指标，分析患者与心理咨询师之间的对话历史，然后给出您的评分。\n请以JSON格式返回评分。参考输出格式如下：\n* 您的分析内容。\n* {{"共情分数": "xx", "辨识分数": "xx", "反思分数": "xx", "策略分数": "xx", "鼓励分数": "xx", "相关性分数": "xx"}} \n\n下面将给出患者与心理咨询师之间具体的对话历史，让我们深呼吸，一步一步地思考！'''
        history = f"用户提问=【{data_dict['question'] + data_dict['description']}】\n\n咨询师回复=【{data_dict['cbt_answer']}】"
    else:
        # 英文的system_prompt和history
        system_prompt = ""
        history = ""
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": history}]
    result = gpt4(message)
    data_dict["score"] = result
    return data_dict
def special_index_scoring(special_index:int, language='zh'):
    read_file_list = read_json_files()
    for read_file in read_file_list:
        whole_path = f"{read_data_path}/{read_file}"
        with open(whole_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)

        data_dict = json_list[special_index]
        data_dict = single_qa_scoring(data_dict, read_file)
        print(data_dict)
        with open(f"{save_data_path}/special_index_scoring_{read_file}", 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)

def clear_old_score(read_file_list):
    if read_file_list is None:
        read_file_list = read_json_files()
    for read_file in read_file_list:
        whole_path = f"{read_data_path}/{read_file}"
        with open(whole_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        result_list = []
        for index, data_dict in enumerate(json_list):
            if "score" in data_dict.keys():
                data_dict.pop("score")
            result_list.append(data_dict)
        with open(f"{read_data_path}/{read_file}", 'w', encoding='utf-8') as f:
            json.dump(result_list, f, indent=4, ensure_ascii=False)

def psyqa_auto_scoring(read_file_list=None):
    if read_file_list is None:
        read_file_list = read_json_files()
    for read_file in read_file_list:
        whole_path = f"{read_data_path}/{read_file}"
        with open(whole_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        result_list = []
        for index, data_dict in enumerate(json_list):
            if "score" in data_dict.keys():
                data_dict.pop("score")
            data_dict = single_qa_scoring(data_dict, read_file)
            result_list.append(data_dict)
            print(f"============{index}：{data_dict['score']}==============================")
            with open(f"{save_data_path}/{read_file}", 'w', encoding='utf-8') as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)

def psyqa_autocbt_scoring_fine(read_file_list=None):
    if read_file_list is None:
        read_file_list = read_json_files()
    for read_file in read_file_list:
        whole_path = f"{read_data_path}/{read_file}"
        with open(whole_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        result_list = []
        for _, data_dict in enumerate(json_list):
            cbt_history = data_dict['cbt_history']
            cbt_history_score = {}
            for index, history_dict in enumerate(cbt_history):
                if "咨询师" not in history_dict['message']['username']:
                    continue
                three_time_score_each_response = []
                current_history_content = history_dict['message']['content']
                system_prompt = f'''# 角色：\n您是一位公正的评判员，熟悉心理学知识和心理咨询。\n\n## 注意：\n您的职责是评估AI心理咨询师对患者心理问题的回答质量。您的评估应参照历史对话内容，并仅根据评估标准打分。\n\n## 评估标准：\n### 共情 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的共情程度，即心理咨询师是否在回复内容中表达了对患者焦虑与痛苦的理解和同情，并是否为患者营造了自由表达焦虑与痛苦的安全感。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否正确理解了患者的提问意图？\n* 心理咨询师的回复内容中，是否对患者的焦虑与痛苦表示尊重、理解和同情？\n* 心理咨询师的回复内容中，是否为患者营造了自由表达焦虑与痛苦的安全感？\n\n### 信念 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的认知扭曲程度，即心理咨询师是否在回复内容中识别出了患者焦虑与痛苦背后错误扭曲的认知信念，深入探讨并帮助患者挑战自己错误扭曲的认知信念。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否识别到了患者错误扭曲的认知信念？\n* 心理咨询师的回复内容中，是否深入探讨了患者错误扭曲的认知信念？\n* 心理咨询师的回复内容中，是否帮助患者识别和挑战自己错误扭曲的认知信念？\n\n### 反思 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的反思程度，即心理咨询师是否在回复内容中提出了一些问题，来鼓励患者重新反思考虑他们的初始想法或错误扭曲的认知信念。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否提出了与患者的初始想法相关的问题？\n* 心理咨询师的回复内容中，是否提出了能帮助患者进行深入思考的问题？\n* 心理咨询师的回复内容中，是否提出了能反映患者错误扭曲的认知信念的问题？\n\n### 策略 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的策略可行性，即确定心理咨询师是否在回复内容中，提供了有益于缓解或解决患者当前焦虑与痛苦的策略或见解，来帮助患者解决当前遇到的情况。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，提供的策略或见解是否可行？\n* 心理咨询师的回复内容中，提供的策略或见解能否解决患者当前的问题？\n* 心理咨询师的回复内容中，提供的策略是否为专业的心理治疗方法？\n\n### 鼓励 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的鼓励程度，即心理咨询师是否在回复内容中鼓励患者采用相应的策略或见解，强调这只是开始，鼓励患者一步一步地改变现状。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否鼓励了患者采取行动应对当前的情况？\n* 心理咨询师的回复内容中，是否对患者未来执行策略遇到失败的可能性，进行提醒和分析？\n* 心理咨询师的回复内容中，是否对患者未来执行策略遇到的挫折与痛苦，进行鼓舞和安慰？\n\n### 相关性 (0-7分)：\n您需要确定咨询师的回答与患者的提问之间的相关性。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回答内容与患者的提问内容是否高度相关？\n* 心理咨询师的回答内容是否自然流畅？\n* 心理咨询师的回答是否涵盖了患者提问的主要问题或关注点？\n\n## 约束条件\n- 请您避免任何偏见，保持客观，并确保回答呈现的顺序不会影响您的判断；\n- 请不要让回答的长度影响您的评估；\n\n## 工作流程\n请根据以上六个评估指标，分析患者与心理咨询师之间的对话历史，然后给出您的评分。\n请以JSON格式返回评分。参考输出格式如下：\n* 您的分析内容。\n* {{"共情分数": "xx", "辨识分数": "xx", "反思分数": "xx", "策略分数": "xx", "鼓励分数": "xx", "相关性分数": "xx"}} \n\n下面将给出患者与心理咨询师之间具体的对话历史，让我们深呼吸，一步一步地思考！'''
                history = f"用户提问=【{data_dict['question'] + data_dict['description']}】\n\n咨询师回复=【{current_history_content}】"

                message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": history}]
                for _ in range(3):
                    print(f"---{_}/3次")
                    result = gpt4(message)
                    three_time_score_each_response.append(result)
                    time.sleep(2)
                cbt_history_score[str(index)] = three_time_score_each_response
            data_dict["cbt_history_score"] = cbt_history_score
            result_list.append(data_dict)
            print(f'''============{_}：{data_dict['cbt_history_score']}==============================''')
            with open(f"{save_data_path}/fine-score-three-time_{read_file}", 'w', encoding='utf-8') as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)

def psyqa_prompt_scoring_fine(read_file_list=None):
    if read_file_list is None:
        read_file_list = read_json_files()
    for read_file in read_file_list:
        whole_path = f"{read_data_path}/{read_file}"
        with open(whole_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        result_list = []
        for index, data_dict in enumerate(json_list):
            three_time_score_each_response = []
            system_prompt = f'''# 角色：\n您是一位公正的评判员，熟悉心理学知识和心理咨询。\n\n## 注意：\n您的职责是评估AI心理咨询师对患者心理问题的回答质量。您的评估应参照历史对话内容，并仅根据评估标准打分。\n\n## 评估标准：\n### 共情 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的共情程度，即心理咨询师是否在回复内容中表达了对患者焦虑与痛苦的理解和同情，并是否为患者营造了自由表达焦虑与痛苦的安全感。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否正确理解了患者的提问意图？\n* 心理咨询师的回复内容中，是否对患者的焦虑与痛苦表示尊重、理解和同情？\n* 心理咨询师的回复内容中，是否为患者营造了自由表达焦虑与痛苦的安全感？\n\n### 信念 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的认知扭曲程度，即心理咨询师是否在回复内容中识别出了患者焦虑与痛苦背后错误扭曲的认知信念，深入探讨并帮助患者挑战自己错误扭曲的认知信念。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否识别到了患者错误扭曲的认知信念？\n* 心理咨询师的回复内容中，是否深入探讨了患者错误扭曲的认知信念？\n* 心理咨询师的回复内容中，是否帮助患者识别和挑战自己错误扭曲的认知信念？\n\n### 反思 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的反思程度，即心理咨询师是否在回复内容中提出了一些问题，来鼓励患者重新反思考虑他们的初始想法或错误扭曲的认知信念。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否提出了与患者的初始想法相关的问题？\n* 心理咨询师的回复内容中，是否提出了能帮助患者进行深入思考的问题？\n* 心理咨询师的回复内容中，是否提出了能反映患者错误扭曲的认知信念的问题？\n\n### 策略 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的策略可行性，即确定心理咨询师是否在回复内容中，提供了有益于缓解或解决患者当前焦虑与痛苦的策略或见解，来帮助患者解决当前遇到的情况。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，提供的策略或见解是否可行？\n* 心理咨询师的回复内容中，提供的策略或见解能否解决患者当前的问题？\n* 心理咨询师的回复内容中，提供的策略是否为专业的心理治疗方法？\n\n### 鼓励 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的鼓励程度，即心理咨询师是否在回复内容中鼓励患者采用相应的策略或见解，强调这只是开始，鼓励患者一步一步地改变现状。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否鼓励了患者采取行动应对当前的情况？\n* 心理咨询师的回复内容中，是否对患者未来执行策略遇到失败的可能性，进行提醒和分析？\n* 心理咨询师的回复内容中，是否对患者未来执行策略遇到的挫折与痛苦，进行鼓舞和安慰？\n\n### 相关性 (0-7分)：\n您需要确定咨询师的回答与患者的提问之间的相关性。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回答内容与患者的提问内容是否高度相关？\n* 心理咨询师的回答内容是否自然流畅？\n* 心理咨询师的回答是否涵盖了患者提问的主要问题或关注点？\n\n## 约束条件\n- 请您避免任何偏见，保持客观，并确保回答呈现的顺序不会影响您的判断；\n- 请不要让回答的长度影响您的评估；\n\n## 工作流程\n请根据以上六个评估指标，分析患者与心理咨询师之间的对话历史，然后给出您的评分。\n请以JSON格式返回评分。参考输出格式如下：\n* 您的分析内容。\n* {{"共情分数": "xx", "辨识分数": "xx", "反思分数": "xx", "策略分数": "xx", "鼓励分数": "xx", "相关性分数": "xx"}} \n\n下面将给出患者与心理咨询师之间具体的对话历史，让我们深呼吸，一步一步地思考！'''
            history = f"用户提问=【{data_dict['question'] + data_dict['description']}】\n\n咨询师回复=【{data_dict['cbt_answer']}】"

            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": history}]
            for x in range(3):
                print(f"---{x}/3次")
                result = gpt4(message)
                three_time_score_each_response.append(result)
            print(f"XXXXX{index}结束")

            data_dict["cbt_history_score"] = three_time_score_each_response
            result_list.append(data_dict)
            print(f'''============{index}：{data_dict['cbt_history_score']}==============================''')
            with open(f"{save_data_path}/fine-score-three-time-prompt_{read_file}", 'w', encoding='utf-8') as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)

#计算平均总分
def compute_autocbt_score_fine(read_file_list=None):
    if read_file_list is None:
        read_file_list = read_json_files(save_data_path)
    for read_file in read_file_list:
        file_path = f"{save_data_path}/{read_file}"
        with open(file_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        language_is_chinese = True if "psyqa" in read_file else None
        for index, json_dict in enumerate(json_list):
            cbt_history_score = json_dict['cbt_history_score']
            cbt_history_list= []
            for key, value in cbt_history_score.items():
                total_score_dict = {'共情分数': 0, '辨识分数': 0, '反思分数': 0, '策略分数': 0, '鼓励分数': 0, '相关性分数': 0} if language_is_chinese else {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0}
                for json_dict_his_str in value:
                    score_str = json_dict_his_str.replace('\n', '').replace("得分", "分数").replace('：', '').replace('，', ',').replace('“', '"').replace('”', '"').replace("`", "'").replace("\&quot;", "\"")
                    pattern = r'\{.*?\}'
                    matches = re.findall(pattern, score_str)
                    final_match_str = matches[0]
                    try:
                        if len(matches) > 1:
                            for _, match_str in enumerate(matches):
                                final_match_str = match_str if (language_is_chinese and '分数' in match_str) or (not language_is_chinese and "_Score" in match_str) else ""
                        if len(matches) == 0:
                            raise Exception("exception in length of matches")

                        data = json.loads(final_match_str.replace('\\', '').replace("\'", "\"").replace("　", "").replace(" ", ""))

                        if language_is_chinese:
                            if "辨识分数" not in data.keys() and "信念分数" in data.keys():
                                data["辨识分数"] = data["信念分数"]
                            if "辨识分数" not in data.keys() and "识别分数" in data.keys():
                                data["辨识分数"] = data["识别分数"]
                            if float(str(data["共情分数"])) > 7 or float(str(data["辨识分数"])) > 7 or float(str(data["反思分数"])) > 7 or float(str(data["策略分数"])) > 7 or float(str(data["鼓励分数"])) > 7 or float(str(data["相关性分数"])) > 7:
                                raise Exception("chinese exception in score beyond requires max score")
                            total_score_dict["共情分数"] += float(str(data["共情分数"]))
                            total_score_dict["辨识分数"] += float(str(data["辨识分数"]))
                            total_score_dict["反思分数"] += float(str(data["反思分数"]))
                            total_score_dict["策略分数"] += float(str(data["策略分数"]))
                            total_score_dict["鼓励分数"] += float(str(data["鼓励分数"]))
                            total_score_dict["相关性分数"] += float(str(data["相关性分数"]))

                        if not language_is_chinese:
                            if float(str(data["Empathy_Score"])) > 7 or float(str(data["Belief_Score"])) > 7 or float(str(data["Reflection_Score"])) > 7 or float(str(data["Strategy_Score"])) > 7 or float(str(data["Encouragement_Score"])) > 7 or float(str(data["Relevance_Score"])) > 7:
                                raise Exception("english exception in score beyond requires max score")
                            total_score_dict["Empathy_Score"] += float(str(data["Empathy_Score"]))
                            total_score_dict["Belief_Score"] += float(str(data["Belief_Score"]))
                            total_score_dict["Reflection_Score"] += float(str(data["Reflection_Score"]))
                            total_score_dict["Strategy_Score"] += float(str(data["Strategy_Score"]))
                            total_score_dict["Encouragement_Score"] += float(str(data["Encouragement_Score"]))
                            total_score_dict["Relevance_Score"] += float(str(data["Relevance_Score"]))

                    except Exception as e:
                        print(f"{index}出现异常：{matches}")
                total_his_score = 0
                for score_name, score_value in total_score_dict.items():
                    total_score_dict[score_name] = f"{(score_value / 3):.3f}"
                    total_his_score += score_value
                total_score_dict["总分数"] = f"{(total_his_score / 3):.3f}"
                total_score_dict["咨询师出现下标"] = key
                cbt_history_list.append(total_score_dict)
            json_dict['cbt_history_average_score'] = cbt_history_list

        better_score_after_routing_num = 0
        total_num_not_equal = 0
        #计算当存在督导师路由时，路由之前与路由之后的分数对比
        for index, json_dict in enumerate(json_list):
            cbt_history_average_score = json_dict['cbt_history_average_score']
            start_response_score = cbt_history_average_score[0]
            end_response_score = cbt_history_average_score[-1]
            if len(cbt_history_average_score) == 1:
                continue
            total_num_not_equal += 1
            if float(start_response_score["总分数"]) < float(end_response_score["总分数"]):
                better_score_after_routing_num += 1
        print(f"计算当存在督导师路由时，路由之前与路由之后的分数对比：after-better/total-not-equal={better_score_after_routing_num}/{total_num_not_equal}, routing-better-rate={(better_score_after_routing_num/total_num_not_equal):.3f}")

        better_score_in_autocbt_first_response_num = 0
        #计算未开始路由时，路由之前的分数与纯prompt的分数对比
        result_dict = compute_prompt_score_fine(["fine-score-three-time-prompt_psyqa_balanced_qwen_prompt.json"]) if language_is_chinese else compute_prompt_score_fine(["fine-score-three-time-prompt_therapistqa_balanced_Qwen2.5-72B-Instruct_prompt.json"])
        for index, json_dict in enumerate(json_list):
            cbt_history_average_score = json_dict['cbt_history_average_score']
            start_response_score = cbt_history_average_score[0]
            reference_score = result_dict[json_dict['questionID']]
            if float(start_response_score["总分数"]) >= float(reference_score["总分数"]):
                better_score_in_autocbt_first_response_num += 1
        print(f"计算未开始路由时，路由之前的分数与纯prompt的分数对比：after-better/total-not-equal={better_score_in_autocbt_first_response_num}/100, routing-better-rate={(better_score_in_autocbt_first_response_num / 100):.3f}")

        better_score_in_autocbt_first_response_num = 0
        # 计算路由之后的分数与纯prompt的分数对比
        result_dict = compute_prompt_score_fine(["fine-score-three-time-prompt_psyqa_balanced_qwen_prompt.json"]) if language_is_chinese else compute_prompt_score_fine(["fine-score-three-time-prompt_therapistqa_balanced_Qwen2.5-72B-Instruct_prompt.json"])
        for index, json_dict in enumerate(json_list):
            cbt_history_average_score = json_dict['cbt_history_average_score']
            start_response_score = cbt_history_average_score[-1]
            reference_score = result_dict[json_dict['questionID']]
            if float(start_response_score["总分数"]) >= float(reference_score["总分数"]):
                better_score_in_autocbt_first_response_num += 1
        print(f"计算路由之后的分数与纯prompt的分数对比：after-better/total-not-equal={better_score_in_autocbt_first_response_num}/100, routing-better-rate={(better_score_in_autocbt_first_response_num / 100):.3f}")

        better_score_in_autocbt_first_response_num = 0
        # 计算未开始路由时，路由之前的分数与纯prompt的分数差距有多少？以纯prompt分数作为baseline，在baseline之间正负0.5波动的比率有多大？
        result_dict = compute_prompt_score_fine(["fine-score-three-time-prompt_psyqa_balanced_qwen_prompt.json"]) if language_is_chinese else compute_prompt_score_fine(["fine-score-three-time-prompt_therapistqa_balanced_Qwen2.5-72B-Instruct_prompt.json"])
        for index, json_dict in enumerate(json_list):
            cbt_history_average_score = json_dict['cbt_history_average_score']
            start_response_score = cbt_history_average_score[0]
            reference_score = result_dict[json_dict['questionID']]
            if abs(float(start_response_score["总分数"]) - float(reference_score["总分数"])) < 0.5:
                better_score_in_autocbt_first_response_num += 1
        print(f"计算未开始路由时，路由之前的分数与纯prompt的分数差距有多少：after-better/total-not-equal={better_score_in_autocbt_first_response_num}/100, routing-better-rate={(better_score_in_autocbt_first_response_num / 100):.3f}")

        print("===========================================================")
        # 计算最终论文的pure分数结果
        if language_is_chinese:
            total_score_dict_pure = {'共情分数': 0, '辨识分数': 0, '反思分数': 0, '策略分数': 0, '鼓励分数': 0, '相关性分数': 0, '总分数': 0}
            result_dict = compute_prompt_score_fine(["fine-score-three-time-pure_psyqa_balanced_qwen_pure.json"])
            for index, start_response_score in result_dict.items():
                total_score_dict_pure["共情分数"] += float(start_response_score["共情分数"])
                total_score_dict_pure["辨识分数"] += float(start_response_score["辨识分数"])
                total_score_dict_pure["反思分数"] += float(start_response_score["反思分数"])
                total_score_dict_pure["策略分数"] += float(start_response_score["策略分数"])
                total_score_dict_pure["鼓励分数"] += float(start_response_score["鼓励分数"])
                total_score_dict_pure["相关性分数"] += float(start_response_score["相关性分数"])
                total_score_dict_pure["总分数"] += float(start_response_score["总分数"])

        if not language_is_chinese:
            total_score_dict_pure = {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0, '总分数': 0}
            result_dict = compute_prompt_score_fine(["fine-score-three-time-prompt_therapistqa_balanced_Qwen2.5-72B-Instruct_pure.json"])
            for index, start_response_score in result_dict.items():
                total_score_dict_pure["Empathy_Score"] += float(start_response_score["Empathy_Score"])
                total_score_dict_pure["Belief_Score"] += float(start_response_score["Belief_Score"])
                total_score_dict_pure["Reflection_Score"] += float(start_response_score["Reflection_Score"])
                total_score_dict_pure["Strategy_Score"] += float(start_response_score["Strategy_Score"])
                total_score_dict_pure["Encouragement_Score"] += float(start_response_score["Encouragement_Score"])
                total_score_dict_pure["Relevance_Score"] += float(start_response_score["Relevance_Score"])
                total_score_dict_pure["总分数"] += float(start_response_score["总分数"])

        for key, value in total_score_dict_pure.items():
            total_score_dict_pure[key] = f"{(value/100):.3f}"
        print(f"pure sub field score={total_score_dict_pure}")
        print_result_str = ""
        for key, value in total_score_dict_pure.items():
            print_result_str += f"&{value} / 7 "
        print(f"计算pure分数结果={print_result_str}")
        print("===========================================================")


        # 计算最终论文的cbt_prompt分数结果
        if language_is_chinese:
            total_score_dict_prompt = {'共情分数': 0, '辨识分数': 0, '反思分数': 0, '策略分数': 0, '鼓励分数': 0, '相关性分数': 0, '总分数': 0}
            result_dict = compute_prompt_score_fine(["fine-score-three-time-prompt_psyqa_balanced_qwen_prompt.json"])
            for index, start_response_score in result_dict.items():
                total_score_dict_prompt["共情分数"] += float(start_response_score["共情分数"])
                total_score_dict_prompt["辨识分数"] += float(start_response_score["辨识分数"])
                total_score_dict_prompt["反思分数"] += float(start_response_score["反思分数"])
                total_score_dict_prompt["策略分数"] += float(start_response_score["策略分数"])
                total_score_dict_prompt["鼓励分数"] += float(start_response_score["鼓励分数"])
                total_score_dict_prompt["相关性分数"] += float(start_response_score["相关性分数"])
                total_score_dict_prompt["总分数"] += float(start_response_score["总分数"])

        if not language_is_chinese:
            total_score_dict_prompt = {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0, '总分数': 0}
            result_dict = compute_prompt_score_fine(["fine-score-three-time-prompt_therapistqa_balanced_Qwen2.5-72B-Instruct_prompt.json"])
            for index, start_response_score in result_dict.items():
                total_score_dict_prompt["Empathy_Score"] += float(start_response_score["Empathy_Score"])
                total_score_dict_prompt["Belief_Score"] += float(start_response_score["Belief_Score"])
                total_score_dict_prompt["Reflection_Score"] += float(start_response_score["Reflection_Score"])
                total_score_dict_prompt["Strategy_Score"] += float(start_response_score["Strategy_Score"])
                total_score_dict_prompt["Encouragement_Score"] += float(start_response_score["Encouragement_Score"])
                total_score_dict_prompt["Relevance_Score"] += float(start_response_score["Relevance_Score"])
                total_score_dict_prompt["总分数"] += float(start_response_score["总分数"])

        for key, value in total_score_dict_prompt.items():
            total_score_dict_prompt[key] = f"{(value/100):.3f}"
        print(f"prompt sub field score={total_score_dict_prompt}")
        print_result_str = ""
        for key, value in total_score_dict_prompt.items():
            print_result_str += f"&{value} / 7 "
        print(f"计算prompt分数结果={print_result_str}")
        print("===========================================================")

        # 计算最终论文的autocbt路由分数结果
        if language_is_chinese:
            total_score_dict_autocbt = {'共情分数': 0, '辨识分数': 0, '反思分数': 0, '策略分数': 0, '鼓励分数': 0, '相关性分数': 0, '总分数': 0}
            for index, json_dict in enumerate(json_list):
                cbt_history_average_score = json_dict['cbt_history_average_score']
                start_response_score = cbt_history_average_score[-1]
                total_score_dict_autocbt["共情分数"] += float(start_response_score["共情分数"])
                total_score_dict_autocbt["辨识分数"] += float(start_response_score["辨识分数"])
                total_score_dict_autocbt["反思分数"] += float(start_response_score["反思分数"])
                total_score_dict_autocbt["策略分数"] += float(start_response_score["策略分数"])
                total_score_dict_autocbt["鼓励分数"] += float(start_response_score["鼓励分数"])
                total_score_dict_autocbt["相关性分数"] += float(start_response_score["相关性分数"])
                total_score_dict_autocbt["总分数"] += float(start_response_score["总分数"])

        if not language_is_chinese:
            total_score_dict_autocbt = {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0, '总分数': 0}
            for index, json_dict in enumerate(json_list):
                cbt_history_average_score = json_dict['cbt_history_average_score']
                start_response_score = cbt_history_average_score[-1]
                total_score_dict_autocbt["Empathy_Score"] += float(start_response_score["Empathy_Score"])
                total_score_dict_autocbt["Belief_Score"] += float(start_response_score["Belief_Score"])
                total_score_dict_autocbt["Reflection_Score"] += float(start_response_score["Reflection_Score"])
                total_score_dict_autocbt["Strategy_Score"] += float(start_response_score["Strategy_Score"])
                total_score_dict_autocbt["Encouragement_Score"] += float(start_response_score["Encouragement_Score"])
                total_score_dict_autocbt["Relevance_Score"] += float(start_response_score["Relevance_Score"])
                total_score_dict_autocbt["总分数"] += float(start_response_score["总分数"])

        for key, value in total_score_dict_autocbt.items():
            total_score_dict_autocbt[key] = f"{(value/100):.3f}"
        print(f"autocbt sub field score={total_score_dict_autocbt}")
        print_result_str = ""
        for key, value in total_score_dict_autocbt.items():
            print_result_str += f"&{value} / 7 "
        print(f"计算autocbt结果={print_result_str}")
        print("===========================================================")

def compute_prompt_score_fine(read_file_list=None):
    if read_file_list is None:
        read_file_list = read_json_files(save_data_path)
    for read_file in read_file_list:
        file_path = f"{save_data_path}/{read_file}"
        with open(file_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        result_dict = {}

        for index, json_dict in enumerate(json_list):
            cbt_history_score = json_dict['cbt_history_score']
            total_score_dict = {'共情分数': 0, '辨识分数': 0, '反思分数': 0, '策略分数': 0, '鼓励分数': 0, '相关性分数': 0} if 'psyqa' in read_file else {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0}
            for history_score_str in cbt_history_score:
                score_str = history_score_str.replace('\n', '').replace("得分", "分数").replace('：', '').replace('，', ',').replace('“', '"').replace('”', '"').replace("`", "'").replace("\&quot;", "\"")
                pattern = r'\{.*?\}'
                matches = re.findall(pattern, score_str)
                final_match_str = matches[0]
                try:
                    if len(matches) > 1:
                        for _, match_str in enumerate(matches):
                            final_match_str = match_str if ("psyqa" in read_file and '分数' in match_str) or ("therapistqa" in read_file and "_Score" in match_str) else ""
                    if len(matches) == 0:
                        raise Exception("exception in length of matches")

                    data = json.loads(final_match_str.replace('\\', '').replace("\'", "\"").replace("　", "").replace(" ", ""))

                    if "psyqa" in read_file:
                        if "辨识分数" not in data.keys() and "信念分数" in data.keys():
                            data["辨识分数"] = data["信念分数"]
                        if "辨识分数" not in data.keys() and "识别分数" in data.keys():
                            data["辨识分数"] = data["识别分数"]
                        if float(str(data["共情分数"])) > 7 or float(str(data["辨识分数"])) > 7 or float(str(data["反思分数"])) > 7 or float(str(data["策略分数"])) > 7 or float(str(data["鼓励分数"])) > 7 or float(str(data["相关性分数"])) > 7:
                            raise Exception("chinese exception in score beyond requires max score")
                        total_score_dict["共情分数"] += float(str(data["共情分数"]))
                        total_score_dict["辨识分数"] += float(str(data["辨识分数"]))
                        total_score_dict["反思分数"] += float(str(data["反思分数"]))
                        total_score_dict["策略分数"] += float(str(data["策略分数"]))
                        total_score_dict["鼓励分数"] += float(str(data["鼓励分数"]))
                        total_score_dict["相关性分数"] += float(str(data["相关性分数"]))

                    if "therapistqa" in read_file:
                        if float(str(data["Empathy_Score"])) > 7 or float(str(data["Belief_Score"])) > 7 or float(str(data["Reflection_Score"])) > 7 or float(str(data["Strategy_Score"])) > 7 or float(str(data["Encouragement_Score"])) > 7 or float(str(data["Relevance_Score"])) > 7:
                            raise Exception("english exception in score beyond requires max score")
                        total_score_dict["Empathy_Score"] += float(str(data["Empathy_Score"]))
                        total_score_dict["Belief_Score"] += float(str(data["Belief_Score"]))
                        total_score_dict["Reflection_Score"] += float(str(data["Reflection_Score"]))
                        total_score_dict["Strategy_Score"] += float(str(data["Strategy_Score"]))
                        total_score_dict["Encouragement_Score"] += float(str(data["Encouragement_Score"]))
                        total_score_dict["Relevance_Score"] += float(str(data["Relevance_Score"]))

                except Exception as e:
                    print(f"{index}出现异常：{matches}")

            total_his_score = 0
            for score_name, score_value in total_score_dict.items():
                total_score_dict[score_name] = f"{(score_value / 3):.3f}"
                total_his_score += score_value
            total_score_dict["总分数"] = f"{(total_his_score / 3):.3f}"
            json_dict['cbt_history_average_score'] = total_score_dict

            result_dict[json_dict['questionID']] = total_score_dict
        return result_dict

if __name__ == '__main__':
    # psyqa_prompt_scoring_fine(read_file_list)
    # clear_old_score(read_file_list)
    # psyqa_auto_scoring(read_file_list)  # 先让GPT打分，保存GPT的返回信息
    # special_index_scoring(39)
    # compute_score_from_dict_fine()  # 本地根据GPT信息，做二次处理
    compute_autocbt_score_fine(["fine-score-three-time_therapistqa_balanced_Qwen2.5-72B-Instruct_autocbt.json"])
    # compute_prompt_score_fine(["fine-score-three-time-prompt_therapistqa_balanced_Qwen2.5-72B-Instruct_prompt.json"])