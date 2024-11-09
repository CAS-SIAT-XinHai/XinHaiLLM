ARG IMG_TAG=2.4.0-cuda12.1-cudnn9-devel
ARG IMG_REPO=pytorch

FROM pytorch/$IMG_REPO:$IMG_TAG

RUN apt-get clean
RUN apt-get update

RUN apt-get -y install curl git

RUN conda install -y -c conda-forge gxx_linux-64

COPY pip.conf /root/.config/pip/pip.conf

RUN pip install python-multipart sse_starlette chromadb sentence_transformers

RUN pip install vllm
# Install vLLM with CUDA 11.8.
#ARG VLLM_VERSION=0.6.3.post1
#ARG PYTHON_VERSION=311
#RUN pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu121-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install aiofiles peft trl datasets accelerate

ARG WORK_DIR=/usr/share/xinhai

RUN mkdir -p ${WORK_DIR}/related_repos

COPY --from=root related_repos/  ${WORK_DIR}/related_repos

# WORKDIR ${WORK_DIR}/related_repos/vllm
# RUN python use_existing_torch.py
# RUN git config --global http.version HTTP/1.1
# RUN pip install -r requirements-build.txt
# RUN pip install -e . --no-build-isolation

WORKDIR ${WORK_DIR}/related_repos/transformers
RUN pip install -e .

RUN pip install timm==0.9.10

RUN pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

RUN pip install paddleocr

RUN pip install df2img kaleido==0.0.3

RUN conda install -y -c fastai opencv-python-headless

RUN conda install -y cudnn

WORKDIR ${WORK_DIR}/backend/src
