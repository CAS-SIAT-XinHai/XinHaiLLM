#!/usr/bin/env bash
#./start_example.sh
WORK_DIR=$(dirname $(readlink -f $0))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"
PID=$BASHPID
echo "$PID"
CONDA_HOME=/home/wuhaihong/anaconda3
CONDA_ENV=agent

OUTPUT_DIR="${WORK_DIR}"/output
mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}"/"${UUID}".txt
exec &> >(tee -a "$log_file")

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('${CONDA_HOME}/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
  eval "$__conda_setup"
else
  if [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
    . "${CONDA_HOME}/etc/profile.d/conda.sh"
  else
    export PATH="${CONDA_HOME}/bin:$PATH"
  fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate $CONDA_ENV

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"                   # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" # This loads nvm bash_completion

#前端
#cd /home/whh/project/xinhai/frontend
start_webui_script="cd ${WORK_DIR}/frontend && npm run serve"
echo "$start_webui_script"
screen -dmS start_webui_$PID bash -c "$start_webui_script"
#tmux new-session -d -s xinhai_webui_$PID "$start_webui_script"

#
#python -m backend.src.xinhai.controller --host 0.0.0.0 --port 5000

#python -m backend.src.xinhai.controller --host 0.0.0.0 --port 5000
#cd backend/src
# 设置环境变量
export CUDA_VISIBLE_DEVICES=7
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=Qwen1.5-7B-Chat
export WORKER_ADDRESS=http://localhost:40002
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40002
export PYTHONPATH=/home/whh/project/Xinhai/backend/src:/home/whh/project/Xinhai/LLaMA-Factory/src
# 运行 Python 模块
python -m backend.src.xinhai.workers.llm \
    --model_name_or_path /data/whh/model/hub/models--Qwen--Qwen1.5-7B-Chat/snapshots/5f4f5e69ac7f1d508f8369e977de208b4803444b \
    --template qwen \
    --infer_backend vllm \
    --vllm_enforce_eager \
    --infer_dtype float16

start_llm_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=Qwen1.5-7B-Chat WORKER_ADDRESS=http://localhost:40001 WORKER_HOST=0.0.0.0 WORKER_PORT=40001 python -m xinhai.workers.llm --model_name_or_path /data2/public/pretrained_models/Qwen1.5-7B-Chat --template qwen --infer_backend vllm --vllm_enforce_eager --infer_dtype float16"
echo "$start_llm_script"
#screen -dmS start_llm_$PID bash -c "$start_llm_script"
tmux new-session -d -s xinhai_llm_$PID "$start_llm_script"


#python -m backend.src.xinhai.controller --host 0.0.0.0 --port 5000
#多模态脚本
# 设置 GPU 设备
export CUDA_VISIBLE_DEVICES=1
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=internvl_chat
export MODEL_PATH=/data/whh/model/hub/models--OpenGVLab--InternVL2-4B/snapshots/91b57f9185f33a1303f56b36073cac0c38454d42
export WORKER_ADDRESS=http://localhost:40001
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40001
export PYTHONPATH=/home/whh/project/Xinhai/backend/src:/home/whh/project/Xinhai/LLaMA-Factory/src
python -m xinhai.workers.mllm \
    --model_name_or_path /data/whh/model/hub/models--OpenGVLab--InternVL2-4B/snapshots/91b57f9185f33a1303f56b36073cac0c38454d42 \
    --template cpm \
    --infer_backend vllm \
    --vllm_enforce_eager \
    --infer_dtype float16


start_mllm_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=MiniCPMV WORKER_ADDRESS=http://localhost:40004 WORKER_HOST=0.0.0.0 WORKER_PORT=40004 python -m xinhai.workers.mllm --model_name_or_path /data/xuancheng/MiniCPM-V-2 --template cpm --infer_backend vllm --vllm_enforce_eager --infer_dtype float16"
echo "$start_mllm_script"
#screen -dmS start_mllm_$PID bash -c "$start_mllm_script"
tmux new-session -d -s xinhai_mllm_$PID "$start_mllm_script"


##knowledge
export PRO_KNOWLEDGE_DB_PATH=/data/pretrained_models/ProDB-bge-1.5-300
export SS_KNOWLEDGE_DB_PATH=/data/pretrained_models/KnowDB-bge-1.5-300
export EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5
export RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1
export CUDA_VISIBLE_DEVICES=0
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=knowledge
export WORKER_ADDRESS=http://localhost:40002
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40002
python -m xinhai.workers.knowledge





#storge
export DB_PATH=/data/whh/model/KnowDB-bge-1.5-300
export EMBEDDING_MODEL_PATH=/data/whh/model/hub/models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116
export RERANKER_MODEL_PATH=/data/whh/model/hub/models--maidalun1020--bce-reranker-base_v1/snapshots/09f1802b01b6f99c60d269f717d39793ca690122
export CUDA_VISIBLE_DEVICES=0
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=storage
export WORKER_ADDRESS=http://localhost:40003
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40003
python -m xinhai.workers.storage

cd backend/src
export PYTHONPATH=/home/whh/project/Xinhai/backend/src:/home/whh/project/Xinhai/LLaMA-Factory/src
export PYTHONPATH=/data/whh/anaconda/envs/agent/lib/python3.10:$PYTHONPATH
python -m xinhai.arena.simulation --config_path ../../examples/OCRAgency/configs/xinhai_ocr.yaml --debug
python -m xinhai.arena.simulation --config_path ../../configs/xinhai_cbt.yaml --debug


#OCR功能
export CUDA_VISIBLE_DEVICES=0
export CONTROLLER_ADDRESS=http://localhost:5000
export MODEL_NAME=paddleocr
export WORKER_ADDRESS=http://localhost:40004
export WORKER_HOST=0.0.0.0
export WORKER_PORT=40004
export PYTHONPATH=/home/whh/project/Xinhai/backend/src:/home/whh/project/Xinhai/LLaMA-Factory/src
python -m xinhai.workers.ocr
