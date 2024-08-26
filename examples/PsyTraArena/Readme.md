## 启动服务
#### controller  
    cd /home/lirenhao/projects/XinHaiLLM/backend/src && python -m xinhai.controller --host 0.0.0.0 --port 7000

#### storage
    cd /home/lirenhao/projects/XinHaiLLM/backend/src && CUDA_VISIBLE_DEVICES=0 CONTROLLER_ADDRESS=http://localhost:7000 MODEL_NAME=storage WORKER_ADDRESS=http://localhost:48888 WORKER_HOST=0.0.0.0 WORKER_PORT=48888 DB_PATH=/data/lirenhao/XinHaiLLM/AgentMem_psychoarena EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5 python -m xinhai.workers.storage

#### knowledge
    cd /home/lirenhao/projects/XinHaiLLM/backend/src && CUDA_VISIBLE_DEVICES=0 CONTROLLER_ADDRESS=http://localhost:7000 MODEL_NAME=knowledge WORKER_ADDRESS=http://localhost:48882 WORKER_HOST=0.0.0.0 WORKER_PORT=48882 PRO_KNOWLEDGE_DB_PATH=/data/lirenhao/XinHaiLLM/ProDB-bge-1.5-1024-zh SS_KNOWLEDGE_DB_PATH=/data/lirenhao/XinHaiLLM/KnowDB-bge-1.5-300 EMBEDDING_MODEL_PATH=/data/pretrained_models/bge-large-zh-v1.5 RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1 python -m xinhai.workers.knowledge

#### feedback
    cd /home/lirenhao/projects/XinHaiLLM/backend/src && CUDA_VISIBLE_DEVICES=0 CONTROLLER_ADDRESS=http://localhost:7000 MODEL_NAME=feedback WORKER_ADDRESS=http://localhost:48883 WORKER_HOST=0.0.0.0 WORKER_PORT=48883 BOOK_CATALOGS_DB_PATH=/data/lirenhao/XinHaiLLM/ProDB-bge-1.5-300 QA_BANK_DB_PATH=/data/lirenhao/XinHaiLLM/CPsyExamDB EMBEDDING_MODEL_PATH=default RERANKER_MODEL_PATH=/data/pretrained_models/maidalun/bce-reranker-base_v1 python -m xinhai.workers.feedback

## 启动模拟
    python -m xinhai.arena.simulation --config_path ../../examples/PsyTraArena/configs/psyschool_4players_zh.yaml --debug