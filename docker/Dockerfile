ARG IMG_TAG=2.4.0-cuda12.1-cudnn9-devel
ARG IMG_REPO=pytorch

FROM pytorch/$IMG_REPO:$IMG_TAG

RUN apt-get update

RUN apt-get -y install curl

RUN pip install vllm==0.5.4

RUN pip install aiofiles peft transformers==4.43.4 trl datasets==2.20.0 accelerate==0.32.0

RUN pip install python-multipart sse_starlette

RUN pip install chromadb sentence_transformers

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 20.14.0

# Install nvm with node and npm
RUN curl https://raw.githubusercontent.com/creationix/nvm/v0.30.1/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

ARG WORK_DIR=/usr/share/xinhai

RUN mkdir ${WORK_DIR}

WORKDIR ${WORK_DIR}/backend/src
