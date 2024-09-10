import json


def analyze_records(record_path):
    with open(record_path, "r") as f:
        res = json.load(f)

    target_key = "retrieved_qa_ids"
    collection_names = ["KG", "CA"]
    top_k_list = [3, 2]

    total_turn = 0
    retrieved_item = {}
    for turn in res:
        target_content = turn[target_key]
        assert len(target_content) == sum(top_k_list)
        for c_name, top_k in zip(collection_names, top_k_list):
            first_k_item = target_content[:top_k]
            target_content = target_content[top_k:]
            if c_name not in retrieved_item:
                retrieved_item[c_name] = first_k_item
            else:
                retrieved_item[c_name] += first_k_item
        total_turn += 1
    
    retrieved_count = {}
    for c_name in collection_names:
        total_count = len(retrieved_item[c_name])
        non_dup_count = len(set(retrieved_item[c_name]))
        retrieved_count[c_name] = (total_count, non_dup_count)
    print(retrieved_count)


if __name__ == "__main__":
    record_path = "./examples/PsyTraArena/cache/simulation_records.json"
    analyze_records(record_path)