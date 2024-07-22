import os
import sys
import json
from zipfile import ZipFile
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


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


def concatenate_fields(doc):
    # Combine subject_name, question, options, and explanation into one string
    options_str = " ".join(f"{key}: {value}" for key, value in doc["options"].items() if value.strip())
    combined_str = f"{doc['subject_name']} {doc['Question']} {options_str} {doc['explanation']}"
    return combined_str


def process_files(zip_path, chroma_path, model_path):
    # Initialize embeddings with the specified model
    embeddings = SentenceTransformerEmbeddings(model_name=model_path)

    for file_path, collection_name in get_files(zip_path):
        print(f"Processing file: {file_path} for collection: {collection_name}")

        # Load the JSON documents
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        # Extract, concatenate fields, and embed explanations
        explanations = [(doc['id'], embeddings.embed(concatenate_fields(doc))) for doc in documents]

        # Define the path for Chroma database
        db_path = os.path.join(chroma_path, collection_name)

        # Check if the Chroma DB for this collection already exists
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
            db = Chroma.from_documents([(doc[0], doc[1]) for doc in explanations], persist_directory=db_path)
        else:
            # Load the existing Chroma DB
            db = Chroma(persist_directory=db_path)
            db.add_documents([(doc[0], doc[1]) for doc in explanations])

        print(f"Finished processing for collection: {collection_name}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <ZIP_PATH> <MODEL_PATH> <CHROMA_PATH> ")
    else:
        zip_path, model_path, chroma_path = sys.argv[1:]
        process_files(zip_path, chroma_path, model_path)