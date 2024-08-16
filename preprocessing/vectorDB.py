from langchain.document_loaders import JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoModel, AutoTokenizer
from typing import Literal
from langchain.embeddings import HuggingFaceBgeEmbeddings
import os
import sys
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
"""
该功能实现专业书籍和社科类书籍的向量数据库构建
python vectorsDB.py <BOOK_PATH> <EMDEDDING_MODEL_PATH> <CHROMADB_PATH>
原始书籍是json文件，格式为：
{
    “instruction”：
    “input”：
    “output”：
}
"""
batch_size_limit = 41665

def get_book_path(bookpath: str):
    paths = os.listdir(bookpath)
    paths = [os.path.join(bookpath, path) for path in paths]
    return paths

def get_split(raw_documents):
    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    )
    documents = r_splitter.split_documents(raw_documents)
    return documents


def embedding_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    device = device or "auto"
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device

def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"

def check_db(db_path):
    if not os.path.exists(db_path):
        os.mkdir(db_path)
    checks = os.listdir(db_path)
    for check in checks:
        if "chroma" in check:
            return True
    return False
    

def get_db(paths, MODEL_PATH, CHROMA_PATH):
    query_instruction = "为这个句子生成表示以用于检索相关文章："
    device = embedding_device()
    if 'bge-large-zh' in MODEL_PATH:
        embeddings = HuggingFaceBgeEmbeddings(model_name=MODEL_PATH,
                                          model_kwargs={'device': detect_device()},
                                          query_instruction=query_instruction)
    else:
        embeddings = SentenceTransformerEmbeddings(model_name=MODEL_PATH)
    for path in paths:
        raw_documents = JSONLoader(path, jq_schema=".[].output").load()
        documents = get_split(raw_documents)
        print(f"Begin Handle a Book {path}")
        if not check_db(CHROMA_PATH):
            # 判断是否需要分批添加文档
            if len(documents) > batch_size_limit:
                batch = documents[0:batch_size_limit]
                db = Chroma.from_documents(batch, embedding=embeddings, persist_directory=CHROMA_PATH)
                for i in range(batch_size_limit, len(documents), batch_size_limit):
                    batch = documents[i:min(i + batch_size_limit, len(documents))]
                    db.add_documents(documents=batch)
            else:
                db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=CHROMA_PATH)
        else:
            # 判断是否需要分批添加文档
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            if len(documents) > batch_size_limit:
                for i in range(0, len(documents), batch_size_limit):
                    batch = documents[i:min(i + batch_size_limit, len(documents))]
                    db.add_documents(documents=batch)
            else:
                db.add_documents(documents=documents)

        print("Already Finish a Book")
    return

if __name__ == "__main__":
    if len(sys.argv) == 4:
        BOOK_PATH = sys.argv[1]
        MODEL_PATH = sys.argv[2]
        CHROMA_PATH = sys.argv[3]
    paths = get_book_path(BOOK_PATH)
    get_db(paths, MODEL_PATH, CHROMA_PATH)
    