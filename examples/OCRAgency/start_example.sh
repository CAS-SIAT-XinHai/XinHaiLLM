#1.先启动controller服务
python -m backend.src.xinhai.controller --host 0.0.0.0 --port 5000

#2.启动llm服务
# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=Qwen1.5-7B-Chat
export WORKER_ADDRESS=http://localhost:40002
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40002
export PYTHONPATH=./backend/src:./LLaMA-Factory/src
# 运行 Python 模块
python -m backend.src.xinhai.workers.llm \
    --model_name_or_path /data/pretrained_models/Qwen1.5-7B-Chat \
    --template qwen \
    --infer_backend vllm \
    --vllm_enforce_eager \
    --infer_dtype float16


#3.启动多模态服务
#多模态脚本
export CUDA_VISIBLE_DEVICES=0
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=minicpmv
export MODEL_PATH=/data/pretrained_models/MiniCPM-Llama3-V-2_5
export WORKER_ADDRESS=http://localhost:40001
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40001
export PYTHONPATH=./backend/src:./LLaMA-Factory/src
python -m xinhai.workers.mllm \
    --model_name_or_path /data/pretrained_models/MiniCPM-Llama3-V-2_5 \
    --template cpm \
    --infer_backend vllm \
    --vllm_enforce_eager \
    --infer_dtype float16


#4.启动storage服务
#storge
export DB_PATH=/data/pretrained_models/StorageDB-bge-1.5-300
export EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5
export RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1
export CUDA_VISIBLE_DEVICES=0
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=storage
export WORKER_ADDRESS=http://localhost:40003
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40003
python -m xinhai.workers.storage


#5.启动OCR服务
export CUDA_VISIBLE_DEVICES=0
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=paddleocr
export WORKER_ADDRESS=http://localhost:40004
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40004
export PYTHONPATH=./backend/src:./LLaMA-Factory/src
python -m xinhai.workers.ocr



#6.然后就可以跑simulation脚本了
export PYTHONPATH=./backend/src:./LLaMA-Factory/src
python -m xinhai.arena.simulation --config_path ../../examples/OCRAgency/configs/xinhai_ocr.yaml --debug

