import argparse
import json
import os
import random
import string
import uuid
from time import sleep
import numpy as np
from tqdm import trange
import requests
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PsyEnEvaluator:
    def __init__(self, args):
        self.n_shot = args.n_shot
        self.model_name = args.model_name
        self.api_base = args.api_base
        self.save_dir = args.save_dir
        self.train_file = args.train_file
        self.test_file = args.test_file

        # Generate a UUID folder to save all results
        self.unique_id = str(uuid.uuid4())
        self.save_path = os.path.join(self.save_dir, self.unique_id)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def eval(self):
        train_dataset = self.load_json(self.train_file)
        test_dataset = self.load_json(self.test_file)

        results = []
        labels = []

        for i in trange(len(test_dataset), desc="Evaluating", position=1, leave=False):
            support_set = random.sample(train_dataset, min(self.n_shot, len(train_dataset)))
            
            row = test_dataset[i]
            choices = [c for c in string.ascii_lowercase if c in row["options"]]

            query, resp, history = self.format_example(row, support_set, choices)

            messages = [
                item for user, assistant in history
                for item in [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant}
                ]
            ]

            response_str = self.get_llm_response(messages, query, choices)
            ans = self.extract_ans(response_str, choices)

            if ans:
                results.append(ans)
                labels.append(row['answer'])

        corrects = (np.array(results) == np.array(labels))
        self._save_results(corrects, results)

    def format_example(self, target_data, support_set, choices):
        query, resp = self.parse_example(target_data, choices)
        history = [self.parse_example(example, choices, with_answer=True) for example in support_set]
        return query.strip(), resp, history

    def parse_example(self, example, choices, with_answer=False):
        candidates = [f"\n{ch}. {example['options'].get(ch, '')}" for ch in choices if example['options'].get(ch)]
        question = example["question"]
        if not with_answer:
            return "".join([question] + candidates), example['answer']
        else:
            return "".join([question] + candidates), f"Answer: {example['answer']}"

    def get_llm_response(self, messages, query, choices, max_retries=5):
        url = self.api_base
        params = {
            "knowledge": "knowledge",
            "model": self.model_name,
            "question": query,
            "messages": messages,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=params, timeout=30)
                response.raise_for_status()
                result = response.json()

                if isinstance(result, dict) and 'ans' in result and result['ans']:
                    return result

                logger.warning(f"Attempt {attempt + 1}: Invalid or empty response: {result}")
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1}: Request failed: {e}")

            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                sleep(sleep_time)

        logger.error("Max retries reached. Returning empty response.")
        return {"ans": ""}

    def extract_ans(self, response_str, choices):
        if not isinstance(response_str, dict) or 'ans' not in response_str:
            logger.error(f"Invalid response format: {response_str}")
            return None

        ans = response_str.get('ans', '').strip().lower()
        if not ans:
            logger.warning("Empty answer received")
            return None

        # Check if the answer is a single letter choice
        if ans in choices:
            return ans

        # Check if the answer is in the format "(a)" or "a)"
        match = re.search(r'\(([a-z])\)|\b([a-z])\)', ans)
        if match:
            extracted = match.group(1) or match.group(2)
            if extracted in choices:
                return extracted

        logger.warning(f"Could not extract a valid answer from: {ans}")
        return None

    def _save_results(self, corrects, results):
        score = 100 * np.mean(corrects)
        logger.info(f"Accuracy: {score:.2f}%")

        results_path = os.path.join(self.save_path, f"results_psy_en.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": score,
                "results": results
            }, f, indent=2)

def main(args):
    evaluator = PsyEnEvaluator(args)
    evaluator.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shot", type=int, default=5, help="Number of shots (examples)")
    parser.add_argument("--api_base", type=str, help="Base API endpoint",
                        default="http://localhost:5555/api/storage-chat-completion")
    parser.add_argument("--model_name", type=str, help="Model name to be used for evaluation",
                        default="Qwen/Qwen2-72B-Instruct")
    parser.add_argument("--save_dir", type=str, help="Directory to save the results", default="results")
    parser.add_argument("--train_file", type=str, help="Path to the training data file", 
                        default="llmeval/processed_data/merged_train.json")
    parser.add_argument("--test_file", type=str, help="Path to the test data file", 
                        default="llmeval/processed_data/merged_test.json")
    args = parser.parse_args()

    main(args)
