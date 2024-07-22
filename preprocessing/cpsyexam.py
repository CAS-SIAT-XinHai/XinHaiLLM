import os
import sys
import json
from zipfile import ZipFile
from langchain.vectorstores import Chroma
'''
python script.py cpsyexam.zip chroma_path
'''
def unzip_files(zip_path, extract_to):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def get_files(zip_path):
    unzip_files(zip_path, "cpsyexam")
    base_path = "cpsyexam"
    for split in ['train', 'dev', 'test']:
        folder_path = os.path.join(base_path, split)
        for filename in os.listdir(folder_path):
            if filename.startswith('CA-') or filename.startswith('KG-'):
                yield os.path.join(folder_path, filename), f"{'CA' if filename.startswith('CA-') else 'KG'}_{split}"


def process_files(zip_path, chroma_path):
    for file_path, collection_name in get_files(zip_path):
        print(f"Processing file: {file_path} for collection: {collection_name}")

        # Load the JSON documents
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        # Define the path for Chroma database
        db_path = os.path.join(chroma_path, collection_name)

        # Check if the Chroma DB for this collection already exists
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
            db = Chroma.from_documents(documents, persist_directory=db_path)
        else:
            # Load the existing Chroma DB
            db = Chroma(persist_directory=db_path)
        db.add_documents(documents)

        print(f"Finished processing for collection: {collection_name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <ZIP_PATH> <CHROMA_PATH>")
    else:
        zip_path, chroma_path = sys.argv[1:]
        process_files(zip_path, chroma_path)