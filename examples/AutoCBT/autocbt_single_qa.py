import json
import os
import subprocess
import yaml
import chromadb

if __name__ == "__main__":
    # start1()
    language = "en" # zh or en
    dataset_name = "psyqa_balanced" if language == "zh" else "therapistqa_balanced"

    with open(f'/data/xuancheng/koenshen/XinHaiLLM_240821/examples/AutoCBT/configs/xinhai_cbt_{language}.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    with open(f"/data/xuancheng/{dataset_name}.json", 'r', encoding='utf-8') as f:
        ques_descs = json.load(f)
    client = chromadb.PersistentClient(path='/data/pretrained_models/AutoCBT-DB')
    texts = []
    for ques_desc in ques_descs:
        texts.append(ques_desc['question'] + ques_desc['description'])
    work_dir = '/data/xuancheng/koenshen/XinHaiLLM_240821/backend/src'
    result = []
    for i in range(0, 100):
        res = []
        config['arena']['environment']['environment_id'] = f'xinhai_cbt_{language}_single_turn_{i}'
        config['arena']['agents'][0]['role_description'] = texts[i]
        with open(f'/data/xuancheng/final_cbtagency/configs/xinhai_cbt_{language}_single_turn_{i}.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, allow_unicode=True)
        while True:
            try:
                subprocess.run(['python', '-m', 'xinhai.arena.simulation', '--config_path', f'/data/xuancheng/final_cbtagency/configs/xinhai_cbt_{language}_single_turn_{i}.yaml'], cwd=work_dir)
            except Exception as e:
                print(f"An error occurred: {e}")
                pass
            target = client.get_collection(f"xinhai_cbt_{language}_simulation_{i}-0")
            metadatas = target.get(include=['metadatas'])['metadatas']
            for metadata in metadatas:
                aa = json.loads(metadata['message'])
                dialogue = aa['username'] + "：" + aa['content']
                res.append(dialogue)
            ## 这里可以调整控制对话轮数
            if len(res) < 2:
                client.delete_collection(name=f"xinhai_cbt_{language}_simulation_{i}-0")
            else:
                break
        result.append(res)
    with open(f'./result/balanced_psyqa_{i}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    result_str = json.dumps(result, ensure_ascii=False, indent=4)
    with open(f"/data/xuancheng/final_cbtagency/{dataset_name}_cbtagency_llama31.json", 'a') as file:
        file.write(result_str)
        file.flush()
