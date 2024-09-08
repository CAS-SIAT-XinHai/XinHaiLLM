import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import logger
from typing import List
import logging
from config import MODEL_PATH

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s - %(message)s',
                    datefmt='%d/%b/%Y %H:%M:%S')
logger = logging.getLogger('memory')

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_PATH)


class MemoryStorage(object):

    def __init__(self, db_path: str, user_id: int, embedding_fn=embedding_fn):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_fn
        self.user_id = user_id
        self.collection = self.client.get_or_create_collection(name="User_"+str(self.user_id), embedding_function=self.embedding_fn)
        
    
    def insert_data(self, documents: List, metadatas: List):
        ### metadatas的格式：[{"source": "Human"}, {"source": "AI"}]
        ### documents格式: ['one', 'tow']
        res = self.collection.count()
        ids = [f'{self.user_id}_{i + res}' for i in range(len(documents))]
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f'User_{self.user_id}\'s memory_storage adds a message')
        
    def get_data(self, k=0) -> List:
        ### 返回第k次及之后的对话
        dialogues = self.collection.get(include=['documents'])['documents']
        sources = self.collection.get(include=['metadatas'])['metadatas']
        res = self.collection.count()
        results = []
        for i in range(k, res):
            results.append(sources[i]['source:  '] + dialogues[i])
        return results
        
    def search_similar(self, query, k=4) -> List:
        search = self.collection.query(query_texts=query, n_results=k)
        dialogues = search['documents'][0]
        sources = search['metadatas'][0]
        results = []
        for i in range(k):
            results.append(sources[i]['source:  '] + dialogues[i])
        return results
    
    def delete(self):
        self.client.delete_collection(name="User_" + str(self.user_id))
        logger.info(f'User_{self.user_id}\'s memory_storage has been deleted!')
