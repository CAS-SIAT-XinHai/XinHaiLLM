import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
from argparse import ArgumentParser


DB_PATH = "/data/lirenhao/XinHaiLLM/AgentMem_psychoarena"
EMBEDDING_MODEL_PATH = "/data/pretrained_models/bge-large-zh-v1.5"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_PATH)


def reset_db():
    print(f"Reset DB: {DB_PATH}...")
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(allow_reset=True))
    client.reset()
    print(f"Done!")


def clear_agent_memory(memory_type="short"):
    agent_ids= ["0","1","2","3"]
    # arena_name="psychool_4p_zh_20240904"
    arena_name="psyschool_4p_zh_20240908"

    client = chromadb.PersistentClient(path=DB_PATH)
    for aid in agent_ids:
        if memory_type == "short":
            storage_key = f"{arena_name}-{aid}"
        elif memory_type == "long":
            storage_key = f"{arena_name}-{aid}_summary"
        else:
            raise NotImplementedError
        
        collection = client.get_or_create_collection(name=f"{storage_key}",
                                            embedding_function=embedding_fn)
        num_delete = collection.count()
        print(f"Agent {aid}: delete {num_delete} {memory_type}-term memories...")
        client.delete_collection(name=storage_key)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", type=str, default="short")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        reset_db()
    else:
        clear_agent_memory(args.type)
