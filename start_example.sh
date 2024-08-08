#!/usr/bin/env bash
WORK_DIR=$(dirname $(readlink -f $0))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"
PID=$BASHPID
echo "$PID"
CONDA_HOME=/home/tanminghuan/anaconda3
CONDA_ENV=base

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

CONTROLLER_HOST=0.0.0.0
CONTROLLER_PORT=5000
CONTROLLER_ADDRESS=http://localhost:5000

OCR_WORKER_PORT=40000
LLM_WORKER_PORT=40001
MLLM_WORKER_PORT=40002
KNOWLEDGE_WORKER_PORT=40003
STORAGE_WORKER_PORT=40004
AGENCY_WORKER_PORT=40005
FEEDBACK_WORKER_PORT=40005

OCR_DEVICE=0
LLM_DEVICE=0
MLLM_DEVICE=1
KNOWLEDGE_DEVICE=0
STORAGE_DEVICE=0

LD_LIBRARY_PATH=/home/tanminghuan/anaconda3/lib/
LLM_MODEL_PATH=/data/pretrained_models/Qwen1.5-7B-Chat
LLM_MODEL_NAME=Qwen1.5-7B-Chat
LLM_MODEL_TEMPLATE=qwen
MLLM_MODEL_PATH=/data/pretrained_models/MiniCPM-V-2
MLLM_MODEL_NAME=MiniCPM-V-2
MLLM_MODEL_TEMPLATE=cpm

start_controller_script="cd ${WORK_DIR}/backend/src && PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src python -m xinhai.controller --host ${CONTROLLER_HOST} --port ${CONTROLLER_PORT}"
echo "$start_controller_script"
#screen -dmS start_webui_$PID bash -c "$start_webui_script"
tmux new-session -d -s xinhai_controller_$PID "$start_controller_script"

sleep 10

start_ocr_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=${OCR_DEVICE} LD_LIBRARY_PATH=${LD_LIBRARY_PATH} CONTROLLER_ADDRESS=${CONTROLLER_ADDRESS} MODEL_NAME=paddleocr WORKER_ADDRESS=http://localhost:${OCR_WORKER_PORT} WORKER_HOST=0.0.0.0 WORKER_PORT=${OCR_WORKER_PORT} python -m xinhai.workers.ocr"
echo "$start_ocr_script"
#screen -dmS start_ocr_$PID bash -c "$start_ocr_script"
tmux new-session -d -s xinhai_ocr_$PID "$start_ocr_script"

start_llm_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=${LLM_DEVICE} PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src CONTROLLER_ADDRESS=${CONTROLLER_ADDRESS} MODEL_NAME=${LLM_MODEL_NAME} WORKER_ADDRESS=http://localhost:${LLM_WORKER_PORT} WORKER_HOST=0.0.0.0 WORKER_PORT=${LLM_WORKER_PORT} python -m xinhai.workers.llm --model_name_or_path ${LLM_MODEL_PATH} --template ${LLM_MODEL_TEMPLATE} --infer_backend vllm --vllm_enforce_eager --infer_dtype float16"
echo "$start_llm_script"
#screen -dmS start_llm_$PID bash -c "$start_llm_script"
tmux new-session -d -s xinhai_llm_$PID "$start_llm_script"

start_mllm_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=${MLLM_DEVICE} PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src CONTROLLER_ADDRESS=${CONTROLLER_ADDRESS} MODEL_NAME=${MLLM_MODEL_NAME} WORKER_ADDRESS=http://localhost:${MLLM_WORKER_PORT} WORKER_HOST=0.0.0.0 WORKER_PORT=${MLLM_WORKER_PORT} python -m xinhai.workers.mllm --model_name_or_path ${MLLM_MODEL_PATH} --template ${MLLM_MODEL_TEMPLATE} --infer_backend vllm --vllm_enforce_eager --infer_dtype float16"
echo "$start_mllm_script"
#screen -dmS start_mllm_$PID bash -c "$start_mllm_script"
tmux new-session -d -s xinhai_mllm_$PID "$start_mllm_script"

PRO_KNOWLEDGE_DB_PATH=/data/pretrained_models/ProDB-bge-1.5-300
SS_KNOWLEDGE_DB_PATH=/data/pretrained_models/KnowDB-bge-1.5-300
EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5
RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1
start_knowledge_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=${KNOWLEDGE_DEVICE} CONTROLLER_ADDRESS=${CONTROLLER_ADDRESS} MODEL_NAME=knowledge WORKER_ADDRESS=http://localhost:${KNOWLEDGE_WORKER_PORT} WORKER_HOST=0.0.0.0 WORKER_PORT=${KNOWLEDGE_WORKER_PORT} PRO_KNOWLEDGE_DB_PATH=${PRO_KNOWLEDGE_DB_PATH} SS_KNOWLEDGE_DB_PATH=${SS_KNOWLEDGE_DB_PATH} EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH} RERANKER_MODEL_PATH=${RERANKER_MODEL_PATH} python -m xinhai.workers.knowledge"
echo "$start_knowledge_script"
#screen -dmS start_ocr_$PID bash -c "$start_ocr_script"
tmux new-session -d -s xinhai_knowledge_$PID "$start_knowledge_script"

STORAGE_DB_PATH=/data/pretrained_models/StorageDB-bge-1.5-300
EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5
RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1
start_storage_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=${STORAGE_DEVICE} PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src  CONTROLLER_ADDRESS=${CONTROLLER_ADDRESS} MODEL_NAME=storage WORKER_ADDRESS=http://localhost:${STORAGE_WORKER_PORT} WORKER_HOST=0.0.0.0 WORKER_PORT=${STORAGE_WORKER_PORT} DB_PATH=${STORAGE_DB_PATH} EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH} python -m xinhai.workers.storage"
echo "$start_storage_script"
tmux new-session -d -s xinhai_storage_$PID "$start_storage_script"

AGENCY_CONFIG_PATH=${WORK_DIR}/configs/xinhai_cbt_agency.yaml
AGENCY_NAME=xinhai_cbt_agency
start_agency_script="cd ${WORK_DIR}/backend/src && PYTHONPATH=${WORK_DIR}/related_repos/LLaMA-Factory/src CONTROLLER_ADDRESS=${CONTROLLER_ADDRESS} MODEL_NAME=${AGENCY_NAME} WORKER_ADDRESS=http://localhost:${AGENCY_WORKER_PORT} WORKER_HOST=0.0.0.0 WORKER_PORT=${AGENCY_WORKER_PORT} AGENCY_CONFIG_PATH=${AGENCY_CONFIG_PATH} python -m xinhai.workers.agency"
echo "$start_agency_script"
tmux new-session -d -s xinhai_agency_$PID "$start_agency_script"

QA_BANK_DB_PATH=/data/pretrained_models/CPsyExamDB
EMBEDDING_MODEL_PATH=default
RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1
start_feedback_script="cd ${WORK_DIR}/backend/src && CUDA_VISIBLE_DEVICES=0 CONTROLLER_ADDRESS=${CONTROLLER_ADDRESS} MODEL_NAME=feedback WORKER_ADDRESS=http://localhost:${FEEDBACK_WORKER_PORT} WORKER_HOST=0.0.0.0 WORKER_PORT=${FEEDBACK_WORKER_PORT} QA_BANK_DB_PATH=${QA_BANK_DB_PATH} EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH} python -m xinhai.workers.feedback"
echo "$start_feedback_script"
tmux new-session -d -s xinhai_feedback_$PID "$start_feedback_script"
