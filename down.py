from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
import torch
import requests
import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'  # 例如：'https://hf-mirror.com'

url = 'https://cdn-lfs.hf-mirror.com/example'
try:
    response = requests.get(url, timeout=300)  # 增加超时时间为 30 秒
    # 处理响应
except requests.exceptions.Timeout:
    print("请求超时，请检查网络或目标服务器状态。")
except requests.exceptions.ConnectionError as e:
    print(f"连接错误：{e}")
#"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"
#Undi95/Meta-Llama-3-8B-hf,facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b,facebook/opt-125m,openai-community/gpt2
for name in ["BAAI/bge-large-zh-v1.5","facebook/opt-125m"]:
    # 指定模型名称（例如：bert-base-uncased）
    model_name = name
    # 使用AutoTokenizer.from_pretrained从Hugging Face下载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,ignore_mismatched_sizes=True,use_auth_token=True)
    # 使用AutoModelForQuestionAnswering.from_pretrained从Hugging Face下载模型
    model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,ignore_mismatched_sizes=True,use_auth_token=True)
    # 释放模型
    del model
    # 释放分词器
    del tokenizer
    # 执行 Python 垃圾回收
    torch.cuda.empty_cache()
# 可选：如果需要使用模型进行推理，可以进一步处理
# inputs = tokenizer("输入你的文本", return_tensors="pt")
# outputs = model(**inputs)
