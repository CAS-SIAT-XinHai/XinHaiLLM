# XinHai
XinHai: Sea of Minds Framework for Multimodal Multi-agent Simulation and Evolution

## Quick Start

### Docker Compose

Build a local docker image in `docker` directory.
```shell
docker build --add-host raw.githubusercontent.com:185.199.108.133 --build-context root=.. -t xinhai .
```

Modify env file
```shell
cp .env.example .env
```

Change `docker-compose.yml` accordingly.

Start docker compose in project root directory.
```shell
docker-compose up
```

Check status of docker images:
```shell
(base) vimos@vimos-Z270MX-Gaming5 XinHaiLLM  % docker-compose images
      Container          Repository    Tag       Image Id       Size
----------------------------------------------------------------------
xinhaillm_bridge_1       xinhai       latest   ad1a7389388c   17.56 GB
xinhaillm_controller_1   xinhai       latest   ad1a7389388c   17.56 GB
xinhaillm_frontend_1     xinhai       latest   ad1a7389388c   17.56 GB
xinhaillm_llm_1          xinhai       latest   ad1a7389388c   17.56 GB
xinhaillm_storage_1      xinhai       latest   ad1a7389388c   17.56 GB
```

If you need to stop the services, use `docker-compose down`.

## Acknowledgment
XinHai learns a lot from the following repos:
* [AgentVerse](https://github.com/OpenBMB/AgentVerse)
* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
* [MGM](https://github.com/dvlab-research/MGM)
* [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
