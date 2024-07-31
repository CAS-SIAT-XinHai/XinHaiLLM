#!/usr/bin/env bash
WORK_DIR=$(dirname $(readlink -f $0))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"
PID=$BASHPID
echo "$PID"
CONDA_HOME=/home/tanminghuan/anaconda3
CONDA_ENV=AutoInvoice

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

start_webui_script="cd ${WORK_DIR}/frontend && npm run serve"
echo "$start_webui_script"
#screen -dmS start_webui_$PID bash -c "$start_webui_script"
tmux new-session -d -s xinhai_webui_$PID "$start_webui_script"

start_controller_script="cd ${WORK_DIR}/backend/src && python -m xinhai.controller --host 0.0.0.0 --port 5000"
echo "$start_controller_script"
#screen -dmS start_webui_$PID bash -c "$start_webui_script"
tmux new-session -d -s xinhai_controller_$PID "$start_controller_script"

sleep 10

start_ocr_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/tanminghuan/anaconda3/lib/ CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=paddleocr WORKER_ADDRESS=http://localhost:40000 WORKER_HOST=0.0.0.0 WORKER_PORT=40000 python -m xinhai.workers.ocr"
echo "$start_ocr_script"
#screen -dmS start_ocr_$PID bash -c "$start_ocr_script"
tmux new-session -d -s xinhai_ocr_$PID "$start_ocr_script"

start_llm_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=Qwen1.5-7B-Chat WORKER_ADDRESS=http://localhost:40001 WORKER_HOST=0.0.0.0 WORKER_PORT=40001 python -m xinhai.workers.llm --model_name_or_path /data2/public/pretrained_models/Qwen1.5-7B-Chat --template qwen --infer_backend vllm --vllm_enforce_eager --infer_dtype float16"
echo "$start_llm_script"
#screen -dmS start_llm_$PID bash -c "$start_llm_script"
tmux new-session -d -s xinhai_llm_$PID "$start_llm_script"

start_mllm_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=MiniCPMV WORKER_ADDRESS=http://localhost:40004 WORKER_HOST=0.0.0.0 WORKER_PORT=40004 python -m xinhai.workers.mllm --model_name_or_path /data/xuancheng/MiniCPM-V-2 --template cpm --infer_backend vllm --vllm_enforce_eager --infer_dtype float16"
echo "$start_mllm_script"
#screen -dmS start_mllm_$PID bash -c "$start_mllm_script"
tmux new-session -d -s xinhai_mllm_$PID "$start_mllm_script"

PRO_KNOWLEDGE_DB_PATH=/data/pretrained_models/ProDB-bge-1.5-300
SS_KNOWLEDGE_DB_PATH=/data/pretrained_models/KnowDB-bge-1.5-300
EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5
RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1
start_knowledge_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=knowledge WORKER_ADDRESS=http://localhost:40002 WORKER_HOST=0.0.0.0 WORKER_PORT=40002 PRO_KNOWLEDGE_DB_PATH=${PRO_KNOWLEDGE_DB_PATH} SS_KNOWLEDGE_DB_PATH=${SS_KNOWLEDGE_DB_PATH} EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH} RERANKER_MODEL_PATH=${RERANKER_MODEL_PATH} python -m xinhai.workers.knowledge"
echo "$start_knowledge_script"
#screen -dmS start_ocr_$PID bash -c "$start_ocr_script"
tmux new-session -d -s xinhai_knowledge_$PID "$start_knowledge_script"

DB_PATH=/data/pretrained_models/KnowDB-bge-1.5-300
EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5
RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1
start_storage_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=storage WORKER_ADDRESS=http://localhost:40003 WORKER_HOST=0.0.0.0 WORKER_PORT=40003 DB_PATH=${DB_PATH} EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH} python -m xinhai.workers.storage"
echo "$start_storage_script"
tmux new-session -d -s xinhai_storage_$PID "$start_storage_script"

AGENCY_CONFIG_PATH=${WORK_DIR}/configs/xinhai_cbt_agency.yaml
start_agency_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src CONTROLLER_ADDRESS=http://localhost:5000 MODEL_NAME=${AGENCY_CONFIG_PATH} WORKER_ADDRESS=http://localhost:40004 WORKER_HOST=0.0.0.0 WORKER_PORT=40004 AGENCY_CONFIG_PATH=${AGENCY_CONFIG_PATH} python -m xinhai.workers.agency"
echo "$start_agency_script"
tmux new-session -d -s xinhai_agency_$PID "$start_agency_script"
