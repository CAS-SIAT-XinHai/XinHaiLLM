启动命令\
**SMBU**:
* cd /data/yangmin/autocbt/XinHaiLLM/backend/src && PATH=/data/yangmin/envs/autocbt/bin:$PATH python -m xinhai.controller --host 0.0.0.0 --port 5077
* cd /data/yangmin/autocbt/XinHaiLLM/backend/src && PATH=/data/yangmin/envs/autocbt/bin:$PATH CONTROLLER_ADDRESS=http://localhost:5077 DB_PATH=/data/yangmin/autocbt/AutoCBT-DB EMBEDDING_MODEL_PATH=/data/yangmin/models/bge-large-zh-v1.5 MODEL_NAME=storage WORKER_ADDRESS=http://localhost:40023 WORKER_HOST=0.0.0.0 WORKER_PORT=40023 python -m xinhai.workers.storage
* cp /data/yangmin/autocbt/XinHaiLLM/examples/AutoCBT/autocbt_single_qa.py /data/yangmin/autocbt/XinHaiLLM/backend/src/xinhai/arena/ 
* cd /data/yangmin/autocbt/XinHaiLLM/backend/src && PATH=/data/yangmin/envs/autocbt/bin:$PATH python -m xinhai.arena.autocbt_single_qa 

**A6000-2**:
* cd /data/xuancheng/koenshen/XinHaiLLM_240821/backend/src && PATH=/home/xuancheng/miniconda3/envs/vllm_openbmb/bin:$PATH python -m xinhai.controller --host 0.0.0.0 --port 5077
* cd /data/xuancheng/koenshen/XinHaiLLM_240821/backend/src && PATH=/home/xuancheng/miniconda3/envs/vllm_openbmb/bin:$PATH CONTROLLER_ADDRESS=http://localhost:5077 DB_PATH=/data/pretrained_models/AutoCBT-DB EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5 MODEL_NAME=storage WORKER_ADDRESS=http://localhost:40023 WORKER_HOST=0.0.0.0 WORKER_PORT=40023 python -m xinhai.workers.storage
* cp /data/xuancheng/koenshen/XinHaiLLM_240821/examples/AutoCBT/autocbt_single_qa.py /data/xuancheng/koenshen/XinHaiLLM_240821/backend/src/xinhai/arena/
* cd /data/xuancheng/koenshen/XinHaiLLM_240821/backend/src && PATH=/home/xuancheng/miniconda3/envs/vllm_openbmb/bin:$PATH python -m xinhai.arena.autocbt_single_qa

**WSL2**:
* cd /mnt/c/koenshen/SVN/XinHaiLLM_240921/XinHaiLLM/backend/src && PATH=/home/koenshen/miniconda3/envs/autocbt/bin:$PATH python -m xinhai.controller --host 0.0.0.0 --port 5077
* cd /mnt/c/koenshen/SVN/XinHaiLLM_240921/XinHaiLLM/backend/src && PATH=/home/koenshen/miniconda3/envs/autocbt/bin:$PATH CONTROLLER_ADDRESS=http://localhost:5077 DB_PATH=/mnt/c/koenshen/SVN/XinHaiLLM_data_and_db/AutoCBT-DB EMBEDDING_MODEL_PATH=/mnt/c/koenshen/SVN/pretrained_models/bge-large-zh-v1.5 MODEL_NAME=storage WORKER_ADDRESS=http://localhost:40023 WORKER_HOST=0.0.0.0 WORKER_PORT=40023 python -m xinhai.workers.storage