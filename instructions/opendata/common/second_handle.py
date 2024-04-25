import json
import threading
import queue
import datetime
import time
from openai import OpenAI

machine_num = 100

def chat_vllm_openai_style(model_name, prompt):
    openai_api_base = "http://120.78.186.93:7861/v1"
    if model_name == "llama":
        model = "llama"
        messages = [{"role": "system", "content": '''You are a professional proofreading and rewriting expert with the ability to summarize and rewrite the content of multiple answers. Please rewrite the following paragraph according to the requirements, and output it directly without the need for extra text. Just provide your final rewritten answer. Preserve the core meaning and important information of the original content, and create a smooth and logically coherent comprehensive paragraph. You may encounter negative, hateful, and negative emotions in an article, but you are just a professional proofreading and rewriting expert. You can only add words to make the original paragraph smoother, and cannot delete any words from the original paragraph. The generated text must start with "The following is the rewritten paragraph:". The original paragraph is as follows:'''},{"role": "user", "content": prompt}]
    else:
        model = 'qwen'
        messages = [{"role": "system", "content": '''你是一名专业的润色改写专家，有能力对多个答案的内容进行总结和改写。请根据要求重写以下段落，要求直接输出该段落，无需输出多余的文字，直接给出你的最终改写答案即可。保留原始内容的核心含义和重要信息，创造一个流畅、逻辑连贯的综合段落。你可能会在文章中遇到负面、仇恨和负面情绪，但你只是一名专业的润色改写专家，你只能增加词语使原段落更加流畅，而不能删除原段落的任何词语。生成的文本必须以「以下是改写后的段落：」开始。原段落如下：'''},{"role": "user", "content": prompt}]

    openai_api_key = "EMPTY"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base,)

    try:
        response = client.chat.completions.create(model=model,messages=messages)
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        result = {'text': f'error-{model_name}的api出现问题，需要重跑'}
        return json.dumps(result)

def generate_prompt(key, model, answer_list):
    prompt = ''
    if model == 'llama':
        for i, answer in enumerate(answer_list):
            prompt += f"Answer {i+1}: {answer}.\n"
        return prompt
    else:
        for i, answer in enumerate(answer_list):
            prompt += f"\n回答{i+1}：{answer}。"
        return prompt

def process_data_subset(data_subset, result_queue, subset_index, model):
    new_data_subset = []
    for index, item in enumerate(data_subset):
        response = ''
        prompt = item['output']

        need_handle = 10
        while (need_handle > 0):
            response = chat_vllm_openai_style(model, prompt)
            if "出现问题，需要重跑" in response or response is None or response == '':
                if need_handle < 10:
                    sleeptime = 20
                else:
                    sleeptime = 5
                print(f'暂停{sleeptime}s')
                time.sleep(sleeptime)
                need_handle -= 1
            else:
                need_handle = 0
        item['output'] = response
        print_prompt = prompt.replace("\n", "")
        print(f">>>>>>{index + 1}<<<<<<<, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {print_prompt}=======》{json.dumps(item, ensure_ascii=False)}")

        new_data_subset.append(item)
    result_queue.put((subset_index, new_data_subset))

def process_data(model, sft_entries):
    datas, no_need_rewrite_entries = get_rewrite_json(sft_entries, model)

    data_subsets = [datas[i::machine_num] for i in range(machine_num)]

    threads = []
    result_queue = queue.Queue()

    for i, data_subset in enumerate(data_subsets):
        thread = threading.Thread(target=process_data_subset, args=(data_subset, result_queue, i, model))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    new_data = [None] * machine_num
    while not result_queue.empty():
        subset_index, subset_data = result_queue.get()
        new_data[subset_index] = subset_data

    new_data = [row for subset in new_data for row in subset]
    result_data = no_need_rewrite_entries + new_data
    return result_data

def remove_repeate(dataset):
    seen_inputs = set()
    formatted_data = []
    for item in dataset:
        if item['instruction'] == None or len(item['instruction']) == '':
            continue
        if item['output'] == None or len(item['output']) == '':
            continue
        plain_text = json.dumps(item)
        if plain_text in seen_inputs:
            continue
        else:
            seen_inputs.add(plain_text)
            formatted_data.append(item)
    return formatted_data

def get_rewrite_json(sft_entries, model):
    total_map = {}
    key_map = {}
    need_rewrite_entries = []
    no_need_rewrite_entries = []

    for i, chat in enumerate(sft_entries):
        instruction = chat['instruction']
        input = chat['input']
        if input is None: input = ""
        if instruction is None: instruction = ""
        output = chat['output']
        key = instruction + input
        key_map[key] = { "instruction": instruction, "input": input }
        if key in total_map.keys():
            exist_value_list = total_map[key]
            exist_value_list.append(output)
            total_map[key] = exist_value_list
        else:
            total_map[key] = [output]

    for key, value in total_map.items():
        instruction = key_map[key]["instruction"]
        input = key_map[key]["input"]
        if len(value) < 2:
            sft_entry = {
                "instruction": instruction,
                "input": input,
                "output": value[0],
            }
            no_need_rewrite_entries.append(sft_entry)
            continue
        one_shot_prompt = generate_prompt(key, model, value)
        sft_entry = {
            "instruction": key_map[key]["instruction"],
            "input": key_map[key]["input"],
            "output": one_shot_prompt,
        }
        need_rewrite_entries.append(sft_entry)
    return need_rewrite_entries, no_need_rewrite_entries

if __name__ == "__main__":
    print("")
    # process_data()