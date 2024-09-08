import json
import os
import subprocess
import yaml
#读于OCRbench
def read_data(file_path):
    # 打开 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取文件并解析为 Python 对象（通常是字典或列表）
        datas = json.load(file)
    return datas

def agency_answer():
    # 设置命令和参数
    command = [
        '/data/whh/anaconda/envs/agent/bin/python', '-m', 'xinhai.arena.simulation',
        '--config_path', '/home/whh/project/Xinhai/examples/OCRAgencyV1/configs/xinhai_ocr_V1.yaml',
        '--debug'
    ]
    # 运行命令
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("命令输出:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("命令执行失败:\n", e.stderr)
    return
# 打印读取的数据

def write_yaml(image_path):
    # 读取配置文件
    with open('/home/whh/project/Xinhai/examples/OCRAgencyV1/configs/xinhai_ocr_V1.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # 修改配置文件中的某些值
    config['arena']["environment"]["image_path"] = image_path

    # 将修改后的配置写回文件
    with open('/home/whh/project/Xinhai/examples/OCRAgencyV1/configs/xinhai_ocr_V1.yaml', 'w') as file:
        yaml.safe_dump(config, file)

def eval(data,image_path):
    total=len(data)
    acc=0
    for data in datas:
        if(data["type"]=="Regular Text Recognition"):
            dataset_name=data["dataset_name"]
            question=data['question']
            answers = data['answers']
            write_yaml(os.path.join(image_path, data['image_path']))

            agency=agency_answer()
            if answers==answers:
                acc+=1
    return acc/total

if __name__ == '__main__':
    datas=read_data("/home/whh/project/Xinhai/examples/OCRAgencyV1/evaluation/OCRBench.json")
    acc=eval(datas,image_path="/home/whh/project/Xinhai/examples/OCRAgencyV1/evaluation/OCRBench_Images")
