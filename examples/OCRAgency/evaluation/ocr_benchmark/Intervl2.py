import json
from argparse import ArgumentParser
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math
import multiprocessing
from multiprocessing import Pool, Queue, Manager
import time
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal

import base64, requests, json, os, io
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel, Field
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
class ImageURL(BaseModel):
    url: str
class MultimodalInputItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


# https://huggingface.co/OpenGVLab/InternVL2-1B


def save_json(json_list,save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file,indent=4)

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./OCRBench_Images")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--OCRBench_file", type=str, default="./OCRBench/OCRBench.json")
    parser.add_argument("--model_path", type=str, default='OpenGVLab/InternVL2-4B')
    parser.add_argument("--save_name", type=str, default="internvl2-4B")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--model_url",type=str, default='http://localhost:40005/v1/chat/completions')
    args = parser.parse_args()
    return args

OCRBench_score = {"Regular Text Recognition":0,"Irregular Text Recognition":0,"Artistic Text Recognition":0,"Handwriting Recognition":0,
"Digit String Recognition":0,"Non-Semantic Text Recognition":0,"Scene Text-centric VQA":0,"Doc-oriented VQA":0,
"Key Information Extraction":0,"Handwritten Mathematical Expression Recognition":0}
AllDataset_score = {"IIIT5K":0,"svt":0,"IC13_857":0,"IC15_1811":0,"svtp":0,"ct80":0,"cocotext":0,"ctw":0,"totaltext":0,"HOST":0,"WOST":0,"WordArt":0,"IAM":0,"ReCTS":0,"ORAND":0,"NonSemanticText":0,"SemanticText":0,
"STVQA":0,"textVQA":0,"ocrVQA":0,"ESTVQA":0,"ESTVQA_cn":0,"docVQA":0,"infographicVQA":0,"ChartQA":0,"ChartQA_Human":0,"FUNSD":0,"SROIE":0,"POIE":0,"HME100k":0}
num_all = {"IIIT5K":0,"svt":0,"IC13_857":0,"IC15_1811":0,"svtp":0,"ct80":0,"cocotext":0,"ctw":0,"totaltext":0,"HOST":0,"WOST":0,"WordArt":0,"IAM":0,"ReCTS":0,"ORAND":0,"NonSemanticText":0,"SemanticText":0,
"STVQA":0,"textVQA":0,"ocrVQA":0,"ESTVQA":0,"ESTVQA_cn":0,"docVQA":0,"infographicVQA":0,"ChartQA":0,"ChartQA_Human":0,"FUNSD":0,"SROIE":0,"POIE":0,"HME100k":0}

def eval_worker(args, data, eval_id, output_queue):
    model_url=args.model_url
    print(f"Process {eval_id} start.")
    for i in tqdm(range(len(data))):
        img_path = os.path.join(args.image_folder, data[i]['image_path'])
        qs = data[i]['question']
        question = f'<image>\n{qs}'
        txt = MultimodalInputItem(type="text", text=question)
        pie = MultimodalInputItem(type="image_url", image_url=ImageURL(url=img_path))
        # 请求体
        payload = {
            "model": "minicpmv",
            "messages": [{
                "role": "user",
                "content": [txt.dict(), pie.dict()]
            }]
        }
        headers = {'Content-Type': 'application/json'}
        # 发送POST请求
        response = requests.post(model_url, data=json.dumps(payload), headers=headers)
        response = response.json()
        answer = response['choices'][0]['message']['content']
        data[i]['predict'] = answer
        print(answer)
    return data


if __name__=="__main__":
    multiprocessing.set_start_method('spawn')
    args = _get_args()

    data_path = args.OCRBench_file

    with open(data_path, "r") as f:
        data = json.load(f)

    output_queue = []
    data=eval_worker(args, data, 0, output_queue)

    for i in range(len(data)):
        data_type = data[i]["type"]
        dataset_name = data[i]["dataset_name"]
        answers = data[i]["answers"]
        if data[i].get('predict',0)==0:
            continue
        predict = data[i]['predict']
        data[i]['result'] = 0
        if dataset_name == "HME100k":
            if type(answers)==list:
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n"," ").replace(" ","")
                    predict = predict.strip().replace("\n"," ").replace(" ","")
                    if answer in predict:
                        data[i]['result'] = 1
            else:
                answers = answers.strip().replace("\n"," ").replace(" ","")
                predict = predict.strip().replace("\n"," ").replace(" ","")
                if answers in predict:
                    data[i]['result'] = 1
        else:
            if type(answers)==list:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n"," ")
                    predict = predict.lower().strip().replace("\n"," ")
                    if answer in predict:
                        data[i]['result'] = 1
            else:
                answers = answers.lower().strip().replace("\n"," ")
                predict = predict.lower().strip().replace("\n"," ")
                if answers in predict:
                    data[i]['result'] = 1
    save_json(data, os.path.join(args.output_folder,f"{args.save_name}.json"))
    if len(data)==1000:
        for i in range(len(data)):
            if data[i].get("result",100)==100:
                continue
            OCRBench_score[data[i]['type']] += data[i]['result']
        recognition_score = OCRBench_score['Regular Text Recognition']+OCRBench_score['Irregular Text Recognition']+OCRBench_score['Artistic Text Recognition']+OCRBench_score['Handwriting Recognition']+OCRBench_score['Digit String Recognition']+OCRBench_score['Non-Semantic Text Recognition']
        Final_score = recognition_score+OCRBench_score['Scene Text-centric VQA']+OCRBench_score['Doc-oriented VQA']+OCRBench_score['Key Information Extraction']+OCRBench_score['Handwritten Mathematical Expression Recognition']
        print("###########################OCRBench##############################")
        print(f"Text Recognition(Total 300):{recognition_score}")
        print("------------------Details of Recognition Score-------------------")
        print(f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}")
        print(f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}")
        print(f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}")
        print(f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}")
        print(f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}")
        print(f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}")
        print("----------------------------------------------------------------")
        print(f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}")
        print("----------------------------------------------------------------")
        print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}")
        print("----------------------------------------------------------------")
        print(f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}")
        print("----------------------------------------------------------------")
        print(f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}")
        print("----------------------Final Score-------------------------------")
        print(f"Final Score(Total 1000): {Final_score}")
    else:
        for i in range(len(data)):
            num_all[data[i]['dataset_name']] += 1
            if data[i].get("result",100)==100:
                continue
            AllDataset_score[data[i]['dataset_name']] += data[i]['result']
        for key in AllDataset_score.keys():
            print(f"{key}: {AllDataset_score[key]/float(num_all[key])}")
