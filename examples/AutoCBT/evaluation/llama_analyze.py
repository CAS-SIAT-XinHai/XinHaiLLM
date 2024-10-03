import json, tiktoken, re, os
import time
from openai import OpenAI

read_data_path = "/mnt/c/koenshen/SVN/XinHaiLLM_data_and_db/data/result"
save_data_path = "/mnt/c/koenshen/SVN/XinHaiLLM_data_and_db/data/score"

# 计算autocbt的平均分
def compute_autocbt_single_score(json_list: list):
    for index, json_dict in enumerate(json_list):
        cbt_history_score = json_dict['cbt_history_score']
        cbt_history_list = []
        for key, value in cbt_history_score.items():
            total_score_dict = {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0}
            for json_dict_his_str in value:
                score_str = json_dict_his_str.replace('\n', '').replace("得分", "分数").replace('：', '').replace('，', ',').replace('“', '"').replace('”', '"').replace("`", "'").replace("\&quot;", "\"")
                pattern = r'\{.*?\}'
                matches = re.findall(pattern, score_str)
                final_match_str = matches[0]
                try:
                    if len(matches) > 1:
                        for _, match_str in enumerate(matches):
                            final_match_str = match_str if "_Score" in match_str else ""
                    if len(matches) == 0:
                        raise Exception("exception in length of matches")

                    data = json.loads(final_match_str.replace('\\', '').replace("\'", "\"").replace("　", "").replace(" ", ""))

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
    return json_list


# 分析分数
def analyze_autocbt_score_fine():
    file_path = f"{save_data_path}/{autocbt_file}"
    with open(file_path, 'r', encoding='utf-8') as f:
        json_list_temp = json.load(f)

    json_list = []
    for json_dict in json_list_temp:
        if json_dict['questionID'] not in refuse_response_questionid:
            json_list.append(json_dict)

    json_list = compute_autocbt_single_score(json_list)

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
    result_dict = compute_prompt_score_fine()
    for index, json_dict in enumerate(json_list):
        cbt_history_average_score = json_dict['cbt_history_average_score']
        start_response_score = cbt_history_average_score[0]
        reference_score = result_dict[json_dict['questionID']]
        if float(start_response_score["总分数"]) >= float(reference_score["总分数"]):
            better_score_in_autocbt_first_response_num += 1
    print(f"计算未开始路由时，路由之前的分数与纯prompt的分数对比：after-better/total-not-equal={better_score_in_autocbt_first_response_num}/{len(json_list)}, routing-better-rate={(better_score_in_autocbt_first_response_num / len(json_list)):.3f}")

    better_score_in_autocbt_first_response_num = 0
    # 计算路由之后的分数与纯prompt的分数对比
    result_dict = compute_prompt_score_fine()
    for index, json_dict in enumerate(json_list):
        cbt_history_average_score = json_dict['cbt_history_average_score']
        start_response_score = cbt_history_average_score[-1]
        reference_score = result_dict[json_dict['questionID']]
        if float(start_response_score["总分数"]) >= float(reference_score["总分数"]):
            better_score_in_autocbt_first_response_num += 1
    print(f"计算路由之后的分数与纯prompt的分数对比：after-better/total-not-equal={better_score_in_autocbt_first_response_num}/{len(json_list)}, routing-better-rate={(better_score_in_autocbt_first_response_num / len(json_list)):.3f}")

    better_score_in_autocbt_first_response_num = 0
    # 计算未开始路由时，路由之前的分数与纯prompt的分数差距有多少？以纯prompt分数作为baseline，在baseline之间正负0.5波动的比率有多大？
    result_dict = compute_prompt_score_fine()
    for index, json_dict in enumerate(json_list):
        cbt_history_average_score = json_dict['cbt_history_average_score']
        start_response_score = cbt_history_average_score[0]
        reference_score = result_dict[json_dict['questionID']]
        if abs(float(start_response_score["总分数"]) - float(reference_score["总分数"])) < 0.5:
            better_score_in_autocbt_first_response_num += 1
    print(f"计算未开始路由时，路由之前的分数与纯prompt的分数差距有多少：after-better/total-not-equal={better_score_in_autocbt_first_response_num}/{len(json_list)}, routing-better-rate={(better_score_in_autocbt_first_response_num / len(json_list)):.3f}")

    print("===========================================================")
    # 计算最终论文的pure分数结果

    total_score_dict_pure = {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0, '总分数': 0}
    result_dict = compute_pure_score_fine()
    for index, start_response_score in result_dict.items():
        total_score_dict_pure["Empathy_Score"] += float(start_response_score["Empathy_Score"])
        total_score_dict_pure["Belief_Score"] += float(start_response_score["Belief_Score"])
        total_score_dict_pure["Reflection_Score"] += float(start_response_score["Reflection_Score"])
        total_score_dict_pure["Strategy_Score"] += float(start_response_score["Strategy_Score"])
        total_score_dict_pure["Encouragement_Score"] += float(start_response_score["Encouragement_Score"])
        total_score_dict_pure["Relevance_Score"] += float(start_response_score["Relevance_Score"])
        total_score_dict_pure["总分数"] += float(start_response_score["总分数"])

    for key, value in total_score_dict_pure.items():
        total_score_dict_pure[key] = f"{(value/len(json_list)):.3f}"
    print(f"pure sub field score={total_score_dict_pure}")
    print_result_str = ""
    for key, value in total_score_dict_pure.items():
        print_result_str += f"&{value} / 7 "
    print(f"计算pure分数结果={print_result_str}")
    print("===========================================================")


    # 计算最终论文的cbt_prompt分数结果
    total_score_dict_prompt = {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0, '总分数': 0}
    result_dict = compute_prompt_score_fine()
    for index, start_response_score in result_dict.items():
        total_score_dict_prompt["Empathy_Score"] += float(start_response_score["Empathy_Score"])
        total_score_dict_prompt["Belief_Score"] += float(start_response_score["Belief_Score"])
        total_score_dict_prompt["Reflection_Score"] += float(start_response_score["Reflection_Score"])
        total_score_dict_prompt["Strategy_Score"] += float(start_response_score["Strategy_Score"])
        total_score_dict_prompt["Encouragement_Score"] += float(start_response_score["Encouragement_Score"])
        total_score_dict_prompt["Relevance_Score"] += float(start_response_score["Relevance_Score"])
        total_score_dict_prompt["总分数"] += float(start_response_score["总分数"])

    for key, value in total_score_dict_prompt.items():
        total_score_dict_prompt[key] = f"{(value/len(json_list)):.3f}"
    print(f"prompt sub field score={total_score_dict_prompt}")
    print_result_str = ""
    for key, value in total_score_dict_prompt.items():
        print_result_str += f"&{value} / 7 "
    print(f"计算prompt分数结果={print_result_str}")
    print("===========================================================")

    # 计算最终论文的autocbt路由分数结果
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
        total_score_dict_autocbt[key] = f"{(value/len(json_list)):.3f}"
    print(f"autocbt sub field score={total_score_dict_autocbt}")
    print_result_str = ""
    for key, value in total_score_dict_autocbt.items():
        print_result_str += f"&{value} / 7 "
    print(f"计算autocbt结果={print_result_str}")
    print("===========================================================")

def compute_prompt_score_fine():
    file_path = f"{save_data_path}/{prompt_file}"
    with open(file_path, 'r', encoding='utf-8') as f:
        json_list_temp = json.load(f)

    json_list = []
    for json_dict in json_list_temp:
        if json_dict['questionID'] not in refuse_response_questionid:
            json_list.append(json_dict)
    result_dict = {}

    for index, json_dict in enumerate(json_list):
        cbt_history_score = json_dict['cbt_history_score']
        total_score_dict = {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0}
        for history_score_str in cbt_history_score:
            score_str = history_score_str.replace('\n', '').replace("得分", "分数").replace('：', '').replace('，', ',').replace('“', '"').replace('”', '"').replace("`", "'").replace("\&quot;", "\"")
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, score_str)
            final_match_str = matches[0]
            try:
                if len(matches) > 1:
                    for _, match_str in enumerate(matches):
                        final_match_str = match_str if "_Score" in match_str else ""
                if len(matches) == 0:
                    raise Exception("exception in length of matches")

                data = json.loads(final_match_str.replace('\\', '').replace("\'", "\"").replace("　", "").replace(" ", ""))

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

def compute_pure_score_fine():
    file_path = f"{save_data_path}/{pure_file}"
    with open(file_path, 'r', encoding='utf-8') as f:
        json_list_temp = json.load(f)

    json_list = []
    for json_dict in json_list_temp:
        if json_dict['questionID'] not in refuse_response_questionid:
            json_list.append(json_dict)
    result_dict = {}

    for index, json_dict in enumerate(json_list):
        cbt_history_score = json_dict['cbt_history_score']
        total_score_dict = {'Empathy_Score': 0, 'Belief_Score': 0, 'Reflection_Score': 0, 'Strategy_Score': 0, 'Encouragement_Score': 0, 'Relevance_Score': 0}
        for history_score_str in cbt_history_score:
            score_str = history_score_str.replace('\n', '').replace("得分", "分数").replace('：', '').replace('，', ',').replace('“', '"').replace('”', '"').replace("`", "'").replace("\&quot;", "\"")
            pattern = r'\{.*?\}'
            matches = re.findall(pattern, score_str)
            final_match_str = matches[0]
            try:
                if len(matches) > 1:
                    for _, match_str in enumerate(matches):
                        final_match_str = match_str if "_Score" in match_str else ""
                if len(matches) == 0:
                    raise Exception("exception in length of matches")

                data = json.loads(final_match_str.replace('\\', '').replace("\'", "\"").replace("　", "").replace(" ", ""))

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

def watching_data_if_have_cannot_word(read_file=""):
    file_path = f"{save_data_path}/{read_file}"
    with open(file_path, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
    print("======")
    refuse_response_list = []
    for a, json_dict in enumerate(json_list):
        cbt_history = json_dict['cbt_history']
        for b, cbt_history_dict in enumerate(cbt_history):
            if cbt_history_dict['message']['content'].startswith("I cannot"):
                refuse_response_list.append(json_dict)
                print(f'''第{a}条：{cbt_history_dict['message']['content']}''')
                break
    return [temp_dict['questionID'] for temp_dict in refuse_response_list]


if __name__ == '__main__':
    autocbt_file = "fine-score-three-time_therapistqa_balanced_Meta-Llama-3.1-70B-Instruct_autocbt.json"
    prompt_file = "fine-score-three-time-prompt_therapistqa_balanced_Meta-Llama-3.1-70B-Instruct_prompt.json"
    pure_file = "fine-score-three-time-prompt_therapistqa_balanced_Meta-Llama-3.1-70B-Instruct_pure.json"
    refuse_response_questionid = watching_data_if_have_cannot_word(autocbt_file)
    analyze_autocbt_score_fine()