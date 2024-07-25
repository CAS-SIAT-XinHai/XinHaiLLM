import os
import sys
import json
import zipfile
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import Literal

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

def concatenate_document(item):
    subject = item["subject_name"]
    question = item["question"]
    options = " ".join(f"{k}: {v}" for k, v in item["options"].items() if v)
    explanation = item["explanation"]
    return f"{subject} {question} {options} {explanation}"

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata else {}

def process_files(files, folder_path, model_path, chroma_path, db_name):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_path, model_kwargs={"device": detect_device()}
    )
    db_path = os.path.join(chroma_path, db_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            documents = [Document(concatenate_document(item)) for item in data]
            for document in documents:
                db.add_documents(documents=[document])
        print(f"Processing {db_name} with file {file_path}")
    print(f"Processed {db_name}")

def process_folder(folder_path: str, model_path: str, chroma_path: str):
    files = os.listdir(folder_path)
    # Sort files into CA and KG groups
    ca_files = [f for f in files if f.startswith("CA")]
    kg_files = [f for f in files if f.startswith("KG")]
    # Process each group into the appropriate database
    if ca_files:
        db_name = f"CA_{os.path.basename(folder_path)}"
        process_files(ca_files, folder_path, model_path, chroma_path, db_name)
    if kg_files:
        db_name = f"KG_{os.path.basename(folder_path)}"
        process_files(kg_files, folder_path, model_path, chroma_path, db_name)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python cpsyexam.py <zipfile_path> <model_path> <chroma_path> <unzip_path>"
        )
        sys.exit(1)

    zip_file_path = sys.argv[1]
    model_path = sys.argv[2]
    chroma_path = sys.argv[3]
    unzip_path = sys.argv[4]

    extract_zip(zip_file_path, unzip_path)
    for split in ["train", "dev", "test"]:
        folder_path = os.path.join(unzip_path, split)
        process_folder(folder_path, model_path, chroma_path)
