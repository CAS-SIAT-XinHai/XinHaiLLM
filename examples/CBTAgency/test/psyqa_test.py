import json
import os
import subprocess
import yaml
import chromadb

with open('/data/yangdi/XinHaiLLM-fork/configs/types_of_cognitive_distortions.yaml', 'r', encoding='utf-8')as f:
    config = yaml.safe_load(f)
    
with open('/data/yangdi/data/psyqa_balanced.json', 'r', encoding='utf-8')as f:
    ques_descs = json.load(f)
    
client = chromadb.PersistentClient(path='/data/pretrained_models/CBT-DB')
    
texts = []
for ques_desc in ques_descs:
    texts.append(ques_desc['question']+ques_desc['description'])
    

work_dir = '/data/yangdi/XinHaiLLM-fork/backend/src'

result = []

for i in range(0, 100):
    res = []
    config['arena']['environment']['environment_id'] = f'xinhai_cbt_simulation_{i}'
    config['arena']['agents'][0]['role_description'] = texts[i]
    with open(f'/data/yangdi/config/balanced_test_{i}.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    while True:
        try:
            subprocess.run(['python', '-m', 'xinhai.arena.simulation', '--config_path', f'/data/yangdi/config/balanced_test_{i}.yaml'], cwd=work_dir)
        except Exception as e:
            print(f"An error occurred: {e}")
            pass
        target = client.get_collection(f"xinhai_cbt_simulation_{i}-0")
        metadatas = target.get(include=['metadatas'])['metadatas']
        for metadata in metadatas:
            aa = json.loads(metadata['message'])
            dialogue = aa['username'] + "：" + aa['content']
            res.append(dialogue)
        ## 这里可以调整控制对话轮数
        if len(res) < 2:
            client.delete_collection(name=f"xinhai_cbt_simulation_{i}-0")
        else:
            break
    result.append(res)
    with open(f'./result/balanced_psyqa_{i}.json', 'w', encoding='utf-8')as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    
    
