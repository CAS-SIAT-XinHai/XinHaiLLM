import json, datetime, pytz
from openai import OpenAI

# read_data_path = "/data/xuancheng/koenshen/data"
read_data_path = "/data/yangmin/autocbt/data"
save_data_path = f"{read_data_path}/result"
beijing_tz = pytz.timezone('Asia/Shanghai')  # 创建北京时间（亚洲/上海）时区对象


def meta_llama31_8b_instruct(message: list):
    model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:40011/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    completion = client.chat.completions.create(model=model, messages=message)
    return completion.choices[0].message.content


def ali_qwen(message: list):
    model = 'Qwen1.5-7B-Chat'
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:40001/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    completion = client.chat.completions.create(model=model, messages=message)
    return completion.choices[0].message.content


def cbt_llm(message: list):
    model = 'baichuan'
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:7861/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    completion = client.chat.completions.create(model=model, messages=message)
    return completion.choices[0].message.content


def generate_cbtllm_answer(file_path=f"{read_data_path}/psyqa_balanced.json",
                           save_path=f"{save_data_path}/psyqa_balanced_cbt.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)
    for index, top_dict in enumerate(qa_list):
        message = [{"role": "user", "content": top_dict["question"] + top_dict["description"]}]
        print(f"psyqa_balanced_cbt-{index}，message={message}")
        result = cbt_llm(message)
        top_dict["cbt_answer"] = result
        top_dict["cbt_history"] = message
        top_dict["cbt_generate_time"] = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, indent=4, ensure_ascii=False)


def generate_cbt_zh_response_with_prompt(file_path=f"{read_data_path}/psyqa_balanced.json",
                                         save_path=f"{save_data_path}/psyqa_balanced_qwen_prompt.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)
    for index, top_dict in enumerate(qa_list):
        message = [{"role": "system", "content":
            '''请你基于咨询者的信息，给出一条专业的，富有同情心和具有助益性的回复。确保你的回复在保持以下认知行为疗法回答结构的基础上，尤其是识别关键思维或信念部分，流畅地将各部分内容相互连接：
              1. 验证和共情：对患者的情感或问题表示理解和同情，创建安全感。
              2. 识别关键思维或信念：通过问题描述，找出可能的认知扭曲或核心信仰。
              3. 提出挑战或反思：提出开放性问题，鼓励患者重新考虑或反思其初始思维或信仰。
              4. 提供策略或见解：提供实用策略或见解，以帮助他们处理当前情况。
              5. 鼓励与前瞻：鼓励患者使用策略，强调这只是开始，并可能需要进一步的支持。'''},
                   {"role": "user", "content": top_dict["question"] + top_dict["description"]}]
        answer = ali_qwen(message)
        print(f"psyqa_balanced_qwen_prompt-{index}，message={message}")
        top_dict["cbt_answer"] = answer
        top_dict["cbt_history"] = message
        top_dict["cbt_generate_time"] = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, indent=4, ensure_ascii=False)


def generate_cbt_zh_response(file_path=f"{read_data_path}/psyqa_balanced.json",
                             save_path=f"{save_data_path}/psyqa_balanced_qwen_pure.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)
    for index, top_dict in enumerate(qa_list):
        message = [{"role": "user", "content": top_dict["question"] + top_dict["description"]}]
        answer = ali_qwen(message)
        print(f"psyqa_balanced_qwen_pure-{index}，message={message}")
        top_dict["cbt_answer"] = answer
        top_dict["cbt_history"] = message
        top_dict["cbt_generate_time"] = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, indent=4, ensure_ascii=False)


def generate_cbt_en_response_with_prompt(file_path=f"{read_data_path}/therapistqa_balanced.json",
                                         save_path=f"{save_data_path}/therapistqa_balanced_llama_prompt.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)
    for index, top_dict in enumerate(qa_list):
        message = [{"role": "system", "content":
            '''Then based on the following question and its description, please provide a professional, compassionate, and helpful response. Ensure your response adheres to the structure of Cognitive Behavioral Therapy (CBT) responses, especially in identifying the key thought or belief, and seamlessly integrates each part:
       1. Validation and Empathy: Show understanding and sympathy for the patient's feelings or issues, creating a sense of safety.
       2. Identify Key Thought or Belief: Through the problem description, identify potential cognitive distortions or core beliefs.
       3. Pose Challenge or Reflection: Raise open-ended questions, encouraging the patient to reconsider or reflect on their initial thoughts or beliefs.
       4.Provide Strategy or Insight: Offer practical strategies or insights to help them deal with the current situation.
       5. Encouragement and Foresight: Encourage the patient to use the strategy, emphasizing that this is just the beginning and further support may be needed.'''},
                   {"role": "user", "content": top_dict["question"] + top_dict["description"]}]
        answer = meta_llama31_8b_instruct(message)
        print(f"therapistqa_balanced_llama_prompt-{index}，message={message}")
        top_dict["cbt_answer"] = answer
        top_dict["cbt_history"] = message
        top_dict["cbt_generate_time"] = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, indent=4, ensure_ascii=False)


def generate_cbt_en_response(file_path=f"{read_data_path}/therapistqa_balanced.json",
                             save_path=f"{save_data_path}/therapistqa_balanced_llama_pure.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_list = json.load(f)
    for index, top_dict in enumerate(qa_list):
        message = [{"role": "user", "content": top_dict["question"] + top_dict["description"]}]
        answer = meta_llama31_8b_instruct(message)
        print(f"therapistqa_balanced_llama_pure-{index}，message={message}")
        top_dict["cbt_answer"] = answer
        top_dict["cbt_history"] = message
        top_dict["cbt_generate_time"] = datetime.datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    generate_cbt_en_response()
    generate_cbt_en_response_with_prompt()
