import json, datetime, csv, re, os
from openai import OpenAI

def read_json_files(directory):
    # 遍历指定目录下的所有文件
    result_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            result_list.append(filename)
    return result_list

# 读取某一路径下的json格式，然后统计里面的各项子分
def compute_score_from_dict():
    read_path = "/data/xuancheng/final_cbtagency/score"
    read_file_list = read_json_files(read_path)
    for read_file in read_file_list:
        file_path = f"{read_path}/{read_file}"
        with open(file_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        if "therapistqa" in read_file:
            total_score_dict = {'Empathy_score': 0, 'Identification_score': 0, 'Reflection_score': 0, 'Strategy_score': 0, 'Encouragement_score': 0, 'Relevance_score': 0}
        else:
            total_score_dict = {'共情分数': 0, '辨识分数': 0, '反思分数': 0, '策略分数': 0, '鼓励分数': 0, '相关性分数': 0}
        single_file_list = []
        for index, json_dict in enumerate(json_list):
            score_str = json_dict["score"]
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, score_str)
            try:
                if len(matches) != 1:
                    raise Exception("exception in length of matches")
                for match in matches:
                    if "therapistqa" in read_file:
                        # match = match.replace("得分","分数").replace("：","")
                        data = json.loads(match)
                        # print(f"{index} Found JSON:", data)
                        total_score_dict["Empathy_score"] += float(str(data["Empathy_score"]))
                        total_score_dict["Identification_score"] += float(str(data["Identification_score"]))
                        total_score_dict["Reflection_score"] += float(str(data["Reflection_score"]))
                        total_score_dict["Strategy_score"] += float(str(data["Strategy_score"]))
                        total_score_dict["Encouragement_score"] += float(str(data["Encouragement_score"]))
                        total_score_dict["Relevance_score"] += float(str(data["Relevance_score"]))
                        single_file_list.append(data)
                    else:
                        match = match.replace("得分","分数").replace("：","")
                        data = json.loads(match)
                        # print(f"{index} Found JSON:", data)
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
        divisors = [100, 100, 100, 100, 100, 100]
        for i, (key, value) in enumerate(total_score_dict.items()):
            # 根据索引应用不同的除法操作
            modified_value = value / divisors[i]
            modified_data[key] = modified_value
        formatted_dict = {k: f"{v:.3f}" for k, v in modified_data.items()}
        print(f"{file_path} -> {formatted_dict} -> {sum([float(v) for v in formatted_dict.values()]):.3f}")
        print("==========================================================")

def gpt4(message: list):
    model = 'gpt-4'
    openai_api_key = os.environ.get("API_KEY")
    openai_api_base = os.environ.get("API_BASE")
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    completion = client.chat.completions.create(model=model, messages=message)
    return completion.choices[0].message.content

def  psyqa_auto_scoring():
    read_path = "/data/xuancheng/final_cbtagency" # 要读取哪一个路径下的所有文件来评测？文件内容必须是json格式
    save_path = "/data/xuancheng/final_cbtagency/score"  #要将新的结果保存到哪一个路径下？
    read_file_list = read_json_files(read_path)
    for read_file in read_file_list:
        whole_path = f"{read_path}/{read_file}"
        with open(whole_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)

        for index, psyqa_balanced_dict in enumerate(json_list):
            # 多轮的history和单轮qa的history可能不一样，按照你觉得合适的方法，组装这个history就行
            if "therapistqa" in read_file:
                history = f"user's QUESTION=[{psyqa_balanced_dict['question'] + psyqa_balanced_dict['description']}]\npsychological counsellors' ANSWER=[{psyqa_balanced_dict['cbt_answer']}]"
                prompt = f'''# Role:\nYou are an impartial judge, familiar with psychological knowledge and psychological counseling.\n\n## Attention:\nYou are responsible for evaluating the quality of the response provided by the AI Psychological counsellors to the user's psychological problems. Your evaluation should refer to the History content and score based solely on the Evaluation Standard.\n\n## Evaluation Standard:\n### Empathy (0-2 points):\nYou need to determine the level of empathy between the counsellor's ANSWER and the user's QUESTION, that is, whether the counsellor's ANSWER expresses understanding and sympathy for the user's emotions or questions, and creates a sense of security for them.\nIncluding but not limited to the following aspects:\n- 1.1 Did the content of counsellor's ANSWER correctly understand the intention of user's QUESTION?\n- 1.2 Does the content of the counsellor's ANSWER respect the thoughts and emotions of the user's QUESTION?\n\n### Identification (0-4 points):\nYou need to determine the level of cognitive distortion in the recognition between the counsellor's ANSWER and the user's QUESTION, that is, whether the counsellor's ANSWER recognized the user's cognitive distortion through the user's QUESTION description in the conversation.\nIncluding but not limited to the following aspects:\n- 2.1 Has the content of the counsellor's ANSWER recognized the cognitive distortion in the user's QUESTION?\n- 2.2 Can the content of counsellor's ANSWER correctly identify and classify the types of cognitive distortions in user's QUESTION?\n\n### Reflection (0-3 points):\nYou need to determine the level of reflection between the counsellor's ANSWER and the user's QUESTION, that is, whether the counsellor's ANSWER asked some open-ended questions to encourage the user to reconsider their initial ideas or beliefs.\nIncluding but not limited to the following aspects:\n- 3.1 Did the counsellor's ANSWER ask any questions related to the user's initial thoughts or beliefs?\n- 3.2 Can ANSWER's inquiry help the user to engage in deep thinking?\n- 3.3 Is the inquiry from counsellor ANSWER open-ended?\n\n### Strategy (0-4 points):\nYou need to determine the whether the strategies mentioned are correct between the counsellor's ANSWER and the user's QUESTION, that is, whether the counsellor's ANSWER provides practical strategies or insights to help the user solve their current situation.\nIncluding but not limited to the following aspects:\n- 4.1 Is the strategy or insight provided by counsellor ANSWER feasible?\n- 4.2 Can the strategies or insights provided by counsellor ANSWER solve the current problem of the user?\n- 4.3 Is the strategy provided by counsellor ANSWER innovative? Have new perspectives or methods been introduced to help user solve their problems?\n- 4.4 Is the strategy provided by counsellor ANSWER professional? Have established psychological treatment methods been used?\n\n### Encouragement (0-1 points):\nYou need to determine the whether the ANSWER provided by the counsellor has inspired the user to take action and solve their current situation between the counsellor's ANSWER and the user's QUESTION.\nIncluding but not limited to the following aspects:\n- 5.1 Has the counsellor's ANSWER encouraged the user to take action to address their current situation?\n\n### Relevance (0-4 points):\nYou need to determine the relevance of the conversation between the counsellor's ANSWER and the user's QUESTION.\nIncluding but not limited to the following aspects:\n- 6.1 Is the content of the counsellor's ANSWER very relevant to the content of the user's QUESTION?\n- 6.2 Is the conversation between the counsellor's ANSWER and the user's QUESTION natural and smooth?\n- 6.3 Does the counsellor's ANSWER cover the main questions or concerns in the user's QUESTION?\n- 6.4 Does the counsellor's ANSWER avoid including content unrelated to the user's question?\n\n## History\n{history}\n\n## Constraints\n- Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision\n- Do not allow the length of the responses to influence your evaluation\n- Do not favor certain names of the assistants. Be as objective as possible\n\n## Workflow\nPlease first analyze the conversation above history between user and counselor based on the 6 evaluation metrics, and then give your score.\nPlease return the score in JSON format. The reference output format is as follows:\n- your Analysis content.\n- {{"Empathy_score": "xx", "Identification_score": "xx", "Reflection_score": "xx", "Strategy_score": "xx", "Encouragement_score": "xx", "Relevance_score": "xx"}}\n\nTake a deep breath and think step by step! '''
            else:
                history = f"用户提问QUESTION=[{psyqa_balanced_dict['question'] + psyqa_balanced_dict['description']}]\n咨询师回复ANSWER=[{psyqa_balanced_dict['cbt_answer']}]"
                prompt = f'''# 角色：\n您是一位公正的评判员，熟悉心理学知识和心理咨询。\n\n## 注意：\n您的职责是评估AI心理咨询师对用户心理问题的回答质量。您的评估应参照历史对话内容，并仅根据评估标准打分。\n\n## 评估标准：\n### 共情 (0-2分)：\n您需要确定咨询师的回答与用户的提问之间的共情水平，即咨询师的回答是否表达了对用户情绪或问题的理解和同情，并为他们营造了一种安全感。\n包括但不限于以下方面：\n- 1.1 咨询师的回答是否正确理解了用户的提问意图？\n- 1.2 咨询师的回答内容是否尊重了用户的思绪和情感？\n\n### 辨识 (0-4分)：\n您需要确定咨询师的回答与用户的提问之间认知扭曲辨识的程度，即咨询师的回答是否通过用户的提问描述在对话中识别出了用户的认知扭曲。\n包括但不限于以下方面：\n- 2.1 咨询师的回答内容是否识别到了用户的认知扭曲？\n- 2.2 咨询师的回答能否正确地识别并分类出用户提问中的认知扭曲类型？\n\n### 反思 (0-3分)：\n您需要确定咨询师的回答与用户的提问之间的反思程度，即咨询师的回答是否提出了一些开放式的问题来鼓励用户重新考虑他们的初始想法或信念。\n包括但不限于以下方面：\n- 3.1 咨询师的回答是否提出了与用户的初始想法或信念相关的问题？\n- 3.2 咨询师的回答所提的问题是否能帮助用户进行深入思考？\n- 3.3 咨询师的回答提出的疑问是否为开放式问题？\n\n### 策略 (0-4分)：\n您需要确定咨询师的回答所提供的策略是否恰当，即咨询师的回答是否提供了实际可行的策略或见解来帮助用户解决当前的情况。\n包括但不限于以下方面：\n- 4.1 咨询师提供的策略或见解是否可行？\n- 4.2 咨询师提供的策略或见解能否解决用户当前的问题？\n- 4.3 咨询师提供的策略是否有创新性？是否引入了新的视角或方法来帮助用户解决问题？\n- 4.4 咨询师提供的策略是否专业？是否使用了既有的心理治疗方法？\n\n### 鼓励 (0-1分)：\n您需要确定咨询师的回答是否激发了用户采取行动以解决当前的情况。\n包括但不限于以下方面：\n- 5.1 咨询师的回答是否鼓励了用户采取行动应对当前的情况？\n\n### 相关性 (0-4分)：\n您需要确定咨询师的回答与用户的提问之间的相关性。\n包括但不限于以下方面：\n- 6.1 咨询师的回答内容与用户的提问内容是否高度相关？\n- 6.2 咨询师的回答与用户的提问之间的对话是否自然流畅？\n- 6.3 咨询师的回答是否涵盖了用户提问的主要问题或关注点？\n- 6.4 咨询师的回答是否避免了包含与用户提问无关的内容？\n\n## 历史对话内容\n{history}\n\n## 约束条件\n- 避免任何位置偏见，并确保回答呈现的顺序不会影响您的判断；\n- 不要让回答的长度影响您的评估；\n- 不要偏向某些助手的名字。尽可能保持客观；\n\n## 工作流程\n请首先根据以上六个评估指标分析用户与咨询师之间的对话历史，然后给出您的评分。\n请以JSON格式返回评分。参考输出格式如下：\n- 您的分析内容。\n- {{"共情分数": "xx", "辨识分数": "xx", "反思分数": "xx", "策略分数": "xx", "鼓励分数": "xx", "相关性分数": "xx"}} \n\n让我们深呼吸，一步一步地思考！'''
            message = [{"role": "user", "content": prompt}]
            result = gpt4(message)
            psyqa_balanced_dict["score"] = result
            print(f"============{index}：{result}==============================")

        result_str = json.dumps(json_list, ensure_ascii=False, indent=4)
        with open(f"{save_path}/{read_file}", 'a') as file:
            file.write(result_str)
            file.flush()

if __name__ == '__main__':
    psyqa_auto_scoring() # 先让GPT打分，保存GPT的返回信息
    compute_score_from_dict() # 本地根据GPT信息，做二次处理