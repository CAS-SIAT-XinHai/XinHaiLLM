from openai import OpenAI
import time
import json
import os
import argparse

read_data_path = "result"
save_data_path = "score"

def read_json_files(path=read_data_path):
    # 遍历指定目录下的所有文件
    result_list = []
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            result_list.append(filename)
    return result_list

def gpt4(message: list):
    model = "gpt-4o-mini-2024-07-18"
    openai_api_key = os.environ.get("API_KEY")
    client = OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(model=model, messages=message)
    return completion.choices[0].message.content

def psyqa_prompt_scoring_fine(read_file_list=None):
    if read_file_list is None:
        read_file_list = read_json_files()
    for read_file in read_file_list:
        whole_path = f"{read_data_path}/{read_file}"
        with open(whole_path, 'r', encoding='utf-8') as f:
            json_list = json.load(f)
        result_list = []
        print(f"开始处理{read_file}打分评测")
        for index, data_dict in enumerate(json_list):
            three_time_score_each_response = []
            if "psyqa" in read_file:
                system_prompt = f'''# 角色：\n您是一位公正的评判员，熟悉心理学知识和心理咨询。\n\n## 注意：\n您的职责是评估AI心理咨询师对患者心理问题的回答质量。您的评估应参照历史对话内容，并仅根据评估标准打分。\n\n## 评估标准：\n### 共情 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的共情程度，即心理咨询师是否在回复内容中表达了对患者焦虑与痛苦的理解和同情，并是否为患者营造了自由表达焦虑与痛苦的安全感。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否正确理解了患者的提问意图？\n* 心理咨询师的回复内容中，是否对患者的焦虑与痛苦表示尊重、理解和同情？\n* 心理咨询师的回复内容中，是否为患者营造了自由表达焦虑与痛苦的安全感？\n\n### 信念 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的认知扭曲程度，即心理咨询师是否在回复内容中识别出了患者焦虑与痛苦背后错误扭曲的认知信念，深入探讨并帮助患者挑战自己错误扭曲的认知信念。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否识别到了患者错误扭曲的认知信念？\n* 心理咨询师的回复内容中，是否深入探讨了患者错误扭曲的认知信念？\n* 心理咨询师的回复内容中，是否帮助患者识别和挑战自己错误扭曲的认知信念？\n\n### 反思 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的反思程度，即心理咨询师是否在回复内容中提出了一些问题，来鼓励患者重新反思考虑他们的初始想法或错误扭曲的认知信念。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否提出了与患者的初始想法相关的问题？\n* 心理咨询师的回复内容中，是否提出了能帮助患者进行深入思考的问题？\n* 心理咨询师的回复内容中，是否提出了能反映患者错误扭曲的认知信念的问题？\n\n### 策略 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的策略可行性，即确定心理咨询师是否在回复内容中，提供了有益于缓解或解决患者当前焦虑与痛苦的策略或见解，来帮助患者解决当前遇到的情况。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，提供的策略或见解是否可行？\n* 心理咨询师的回复内容中，提供的策略或见解能否解决患者当前的问题？\n* 心理咨询师的回复内容中，提供的策略是否为专业的心理治疗方法？\n\n### 鼓励 (0-7分)：\n您需要确定患者的提问与心理咨询师的回复之间的鼓励程度，即心理咨询师是否在回复内容中鼓励患者采用相应的策略或见解，强调这只是开始，鼓励患者一步一步地改变现状。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回复内容中，是否鼓励了患者采取行动应对当前的情况？\n* 心理咨询师的回复内容中，是否对患者未来执行策略遇到失败的可能性，进行提醒和分析？\n* 心理咨询师的回复内容中，是否对患者未来执行策略遇到的挫折与痛苦，进行鼓舞和安慰？\n\n### 相关性 (0-7分)：\n您需要确定咨询师的回答与患者的提问之间的相关性。您需要关注的点包括但不限于以下方面：\n* 心理咨询师的回答内容与患者的提问内容是否高度相关？\n* 心理咨询师的回答内容是否自然流畅？\n* 心理咨询师的回答是否涵盖了患者提问的主要问题或关注点？\n\n## 约束条件\n- 请您避免任何偏见，保持客观，并确保回答呈现的顺序不会影响您的判断；\n- 请不要让回答的长度影响您的评估；\n\n## 工作流程\n请根据以上六个评估指标，分析患者与心理咨询师之间的对话历史，然后给出您的评分。\n请以JSON格式返回评分。参考输出格式如下：\n* 您的分析内容。\n* {{"共情分数": "xx", "辨识分数": "xx", "反思分数": "xx", "策略分数": "xx", "鼓励分数": "xx", "相关性分数": "xx"}} \n\n下面将给出患者与心理咨询师之间具体的对话历史，让我们深呼吸，一步一步地思考！'''
                history = f"用户提问=【{data_dict['question'] + data_dict['description']}】\n\n咨询师回复=【{data_dict['cbt_answer']}】"
            else:
                system_prompt = f'''# Role:\nYou are an impartial evaluator familiar with psychology and counseling.\n\n## Note:\nYour responsibility is to assess the quality of responses from the AI counselor to the patient's psychological issues. Your evaluation should refer to the historical dialogue content and score based solely on the evaluation criteria.\n\n## Evaluation Criteria:\n### Empathy (0-7 points):\nDetermine the level of empathy between the patient's questions and the counselor's responses, specifically whether the counselor expresses understanding and sympathy for the patient's anxiety and pain, and whether they create a safe space for the patient to express their feelings. Focus on aspects such as:\n* Did the counselor correctly understand the patient's intent?\n* Did the counselor show respect, understanding, and sympathy for the patient's anxiety and pain?\n* Did the counselor create a safe environment for the patient to express their feelings?\n\n### Beliefs (0-7 points):\nAssess the degree of cognitive distortion identified between the patient's questions and the counselor's responses, specifically whether the counselor recognizes and helps challenge the patient's distorted beliefs about their anxiety and pain. Consider the following:\n* Did the counselor identify the patient's distorted beliefs?\n* Did the counselor delve into the patient's distorted beliefs?\n* Did the counselor assist the patient in recognizing and challenging these distorted beliefs?\n\n### Reflection (0-7 points):\nEvaluate the level of reflection between the patient's questions and the counselor's responses, particularly whether the counselor posed questions that encouraged the patient to reconsider their initial thoughts or distorted beliefs. Look for:\n* Did the counselor ask questions related to the patient's initial thoughts?\n* Did the counselor pose questions that facilitated deeper thinking?\n* Did the counselor ask questions reflecting the patient's distorted beliefs?\n\n### Strategies (0-7 points):\nDetermine the feasibility of the strategies provided by the counselor in response to the patient's concerns. Analyze whether the counselor offered helpful insights or strategies to alleviate or resolve the patient's anxiety and pain. Pay attention to:\n* Were the strategies or insights provided by the counselor practical?\n* Could the strategies or insights solve the patient's current problems?\n* Were the strategies based on professional psychological methods?\n\n### Encouragement (0-7 points):\nAssess the level of encouragement in the counselor's responses, specifically whether the counselor encouraged the patient to adopt the strategies or insights shared, emphasizing that this is just the beginning and motivating the patient to make gradual changes. Consider:\n* Did the counselor encourage the patient to take action?\n* Did the counselor address potential failures the patient might encounter while implementing the strategies?\n* Did the counselor provide comfort and encouragement regarding setbacks and challenges?\n\n### Relevance (0-7 points):\nEvaluate the relevance of the counselor's responses to the patient's questions. Focus on:\n* Was the counselor's response highly relevant to the patient's question?\n* Did the counselor's response flow naturally?\n* Did the counselor's answer cover the main issues or concerns raised by the patient?\n\n## Constraints\n- Please avoid any bias, remain objective, and ensure the order of your responses does not affect your judgment;\n- Do not let the length of the response influence your evaluation;\n\n## Workflow\nBased on the six evaluation criteria above, analyze the dialogue history between the patient and the counselor, then provide your scores.\nPlease return your scores in JSON format. Use the following reference output format:\n* Your analysis.\n* {{"Empathy_Score": "xx", "Belief_Score": "xx", "Reflection_Score": "xx", "Strategy_Score": "xx", "Encouragement_Score": "xx", "Relevance_Score": "xx"}} \n\nThe specific dialogue history between the patient and the counselor will now be provided—let's take a deep breath and think step by step!'''
                history = f"patient's question=【{data_dict['question'] + data_dict['description']}】\n\ncounselor's response=【{data_dict['cbt_answer']}】"

            message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": history}]
            for x in range(3):
                print(f"---第{index}条回答的第{x+1}/3次")
                result = gpt4(message)
                three_time_score_each_response.append(result)

            data_dict["cbt_history_score"] = three_time_score_each_response
            result_list.append(data_dict)
            print(f'''============{index}：{data_dict['cbt_history_score']}==============================''')
            with open(f"{save_data_path}/fine-score-three-time-prompt_{read_file}", 'w', encoding='utf-8') as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)

# 为纯cbt promt和无cbt prompt的回答自动打分
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some input files.")
    parser.add_argument('--filename', type=str, required=True, help='The name of the file to process')
    args = parser.parse_args()
    read_file_list = [args.filename]
    psyqa_prompt_scoring_fine(read_file_list)
