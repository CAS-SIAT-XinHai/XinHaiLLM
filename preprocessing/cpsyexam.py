import os
import sys
import json
import zipfile
import uuid
from typing import Literal, Dict, Any, List
import chromadb
from chromadb.utils import embedding_functions


def extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"


def concatenate_document(item: Dict[str, Any]) -> str:
    subject = item.get("subject_name", "")
    question = item.get("question", "")
    options = " ".join(f"{k}: {v}" for k, v in item.get("options", {}).items() if v)
    explanation = item.get("explanation", "")
    return f"{subject} {question} {options} {explanation}"


def process_files(files: List[str], folder_path: str, collection, embedding_function, collection_name: str):
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {collection_name} with file {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            contents = [json.dumps(item) for item in data]
            metadatas = [{"concatenated_text": concatenate_document(item)} for item in data]
            texts_to_embed = [metadata["concatenated_text"] for metadata in metadatas]
            if texts_to_embed:
                embeddings = embedding_function(texts_to_embed)
                ids = [str(uuid.uuid4()) for _ in range(len(contents))]
                collection.add(
                    documents=contents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            print(f"Processed {collection_name} with file {file_path}")


def process_folder(folder_path, split_name, chroma_client, embedding_function):
    files = os.listdir(folder_path)
    ca_files = [f for f in files if f.startswith("CA")]
    kg_files = [f for f in files if f.startswith("KG")]

    def create_or_replace_collection(collection_name):
        existing_collections = [col.name for col in chroma_client.list_collections()]
        if collection_name in existing_collections:
            print(f"Deleting existing collection: {collection_name}")
            chroma_client.delete_collection(collection_name)
        print(f"Creating collection: {collection_name}")
        return chroma_client.create_collection(collection_name,embedding_function=embedding_function)

    if ca_files:
        ca_collection_name = f"CA_{split_name}"
        ca_collection = create_or_replace_collection(ca_collection_name)
        process_files(ca_files, folder_path, ca_collection, embedding_function, ca_collection_name)

    if kg_files:
        kg_collection_name = f"KG_{split_name}"
        kg_collection = create_or_replace_collection(kg_collection_name)
        process_files(kg_files, folder_path, kg_collection, embedding_function, kg_collection_name)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <zipfile_path> <model_path> <chroma_path> <unzip_path> <embedding_path>")
        sys.exit(1)

    zip_file_path = sys.argv[1]
    model_path = sys.argv[2]
    chroma_path = sys.argv[3]
    unzip_path = sys.argv[4]
    # zip_file_path = "../cpsyexam.zip"
    # chroma_path = "/data/pretrained_models/CPsyExamDB"
    # unzip_path = "../cpsyexam"

    extract_zip(zip_file_path, unzip_path)
    device = detect_device()

    # 初始化 Chroma 客户端
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    # chroma_client.reset()
    # 使用默认嵌入函数
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_path)

    for split in ["train", "dev", "test"]:
        folder_path = os.path.join(unzip_path, split)
        process_folder(folder_path, split, chroma_client, embedding_function)