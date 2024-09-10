import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
from argparse import ArgumentParser
import json

DB_PATH = "/data/lirenhao/XinHaiLLM/AgentMem_psychoarena"
EMBEDDING_MODEL_PATH = "/data/pretrained_models/bge-large-zh-v1.5"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_PATH)


def extract_agent_memory(args):
    target_id = args.id
    memory_type = args.type

    agent_ids= ["0","1","2","3"]
    arena_name="psyschool_4p_zh_20240908"

    client = chromadb.PersistentClient(path=DB_PATH)
    res_dict = {}

    for aid in agent_ids:
        if memory_type == "short":
            storage_key = f"{arena_name}-{aid}"
        elif memory_type == "long":
            storage_key = f"{arena_name}-{aid}_summary"
        else:
            raise NotImplementedError
        
        collection = client.get_or_create_collection(name=f"{storage_key}",
                                            embedding_function=embedding_fn)
        total_num = collection.count()
        print(f"Agent {aid} have {total_num} {memory_type}-term memories")

        target_mem = collection.get(ids=str(target_id))["metadatas"][0]
        summary = eval(target_mem["summary"])
        res_dict[f"agent-{aid}"] = summary
    
    return res_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", type=str, default="long")
    parser.add_argument("--id", type=int, default=55)
    args = parser.parse_args()

    output_path = f"./examples/PsyTraArena/scripts/{args.type}-term-memory_indexId-{args.id}.json"
    res_dict = extract_agent_memory(args)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False)