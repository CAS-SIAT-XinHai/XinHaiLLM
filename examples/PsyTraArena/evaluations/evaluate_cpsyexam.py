import argparse
import json
import os
import re
import string
import uuid
from time import sleep
import numpy as np
from tqdm import trange
import requests
import random
import time
from requests.exceptions import RequestException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIEvaluator:
    def __init__(self, args):
        self.n_shot = args.n_shot
        self.task_dir = args.task_dir
        self.task = args.task
        self.model_name = args.model_name
        self.api_base = args.api_base
        self.save_dir = args.save_dir
        self.default_question_type = 'single'

        # Load the mapping table
        with open(os.path.join(self.task_dir, self.task, 'mapping.json'), 'r', encoding='utf-8') as f:
            self.task_mapping = json.load(f)

        # Generate a UUID folder to save all results
        self.unique_id = str(uuid.uuid4())
        print(self.unique_id)
        self.save_path = os.path.join(self.save_dir, self.unique_id)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def load_json(self, dataset_dir, dataset_name):
        """
        Used to load the specified train or test dataset
        """
        file_path = os.path.join(dataset_dir, f"{dataset_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    def eval(self):
        # Iterate through all categories
        for split in self.task_mapping:
            dataset_name = self.task_mapping[split]["name"]
            logger.info(f"Evaluating split: {split} ({dataset_name})")

            # Load dataset train and test splits
            train_dir = os.path.join(self.task_dir, self.task, 'train')
            test_dir = os.path.join(self.task_dir, self.task, 'test')
            try:
                # Load train and test splits
                train_dataset = self.load_json(train_dir, dataset_name)
                test_dataset = self.load_json(test_dir, dataset_name)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON for {dataset_name}: {e}")
                continue
            except FileNotFoundError as e:
                logger.error(f"File not found for {dataset_name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error loading {dataset_name}: {e}")
                continue

            if not train_dataset or not test_dataset:
                logger.warning(f"Empty dataset for {split}, skipping")
                continue

            results = []
            labels = []
            # if split == "CA-心理咨询-单项选择题":
            #     continue
            # Iterate through samples in the test set
            for i in trange(len(test_dataset), desc=f"Evaluating {split}", position=1, leave=False):
                # Randomly select n_shot examples from train dataset as few-shot support set
                support_set = random.sample(train_dataset, min(self.n_shot, len(train_dataset)))

                # Get choices for the question
                choices = [c for c in string.ascii_uppercase if c in test_dataset[i]["options"]]
                row = test_dataset[i]
                subject_name = row.get('subject_name', 'Unknown Subject')

                # Prepare query, response, system, and history
                query, resp, history = self.format_example_with_choices(
                    target_data=row,
                    support_set=support_set,
                    subject_name=subject_name,
                    choices=choices
                )

                # Create complete prompt
                messages = [
                    item for user, assistant in history
                    for item in [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant}
                    ]
                ]

                # Get LLM response with retry mechanism
                response_str = self.get_llm_response(messages, query, choices,
                                                     row.get('question_type', self.default_question_type))

                # Extract answer
                ans_list = self.extract_ans(response_str, choices, row.get('question_type', self.default_question_type))

                # If we still got an empty answer after all retries, log it and continue
                if not ans_list:
                    print(f"Warning: Empty answer for question {i} in {split} after all retries")
                    continue

                results.append(''.join(ans_list))
                labels.append(row['answer'])  # Ground truth

            # Evaluate results
            corrects = (np.array(results) == np.array(labels))
            self._save_results(corrects, results, split)

    def format_example_with_choices(self, target_data, support_set, subject_name, choices):
        """
        Format the question, answer, system message, and few-shot support set.
        """
        # Parse the main question and options
        query, resp = self.parse_example_with_choices(target_data, choices)

        # Parse the examples from the few-shot support set
        history = [self.parse_example_with_choices(support_set[k], choices, with_answer=True) for k in
                   range(len(support_set))]

        return query.strip(), resp, history

    def parse_example_with_choices(self, example, choices, with_answer=False):
        """
        Create a formatted string with the question and options. Optionally include the correct answer.
        """
        candidates = [f"\n{ch}. {example['options'].get(ch, '')}" for ch in choices if example['options'].get(ch)]
        question = example["question"]
        if not with_answer:
            return "".join([question] + candidates), example['answer']
        else:
            return "".join([question] + candidates), f"答案：{example['answer']}"

    def get_llm_response(self, messages, query, choices, question_type, max_retries=5, initial_delay=1, max_delay=16):
        """
        Send the complete prompt to the LLM API and return the response.
        Implements exponential backoff for retries and ensures a non-empty answer.
        """
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

                # Check if the response is valid and non-empty
                if isinstance(result, dict) and 'ans' in result and result['ans']:
                    ans_list = self.extract_ans(result, choices, question_type)
                    if ans_list:  # If we got a non-empty answer, return it
                        return result

                print(f"Attempt {attempt + 1}: Invalid or empty response: {result}")
            except RequestException as e:
                print(f"Attempt {attempt + 1}: Request failed: {e}")

            # If we're here, the request failed or returned an invalid/empty response
            if attempt < max_retries - 1:  # don't sleep after the last attempt
                delay = min(initial_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

        print("Max retries reached. Returning empty response.")
        return {"ans": ""}

    def extract_ans(self, response_str, choices, question_type):
        """
        Extract the answer from the LLM response.
        Determine if it's a single-choice or multiple-choice question based on question_type and extract the answer accordingly.
        """
        if not isinstance(response_str, dict) or 'ans' not in response_str:
            logger.error(f"Invalid response format: {response_str}")
            return []

        ans = response_str.get('ans', '')
        if not ans:
            logger.warning("Empty answer received")
            return []

        # If it's a single-choice question, return the answer directly
        if question_type == "single" or question_type == "单项选择题":
            # Ensure the returned answer is one of the valid options
            # Handle both formats: single letter and letter in parentheses
            cleaned_ans = ans.strip().strip('()').upper()
            if cleaned_ans in choices:
                return [cleaned_ans]
            elif ans.strip().upper() in [f'({c.upper()})' for c in choices]:
                return [ans.strip().strip('()').upper()]
            else:
                return []

        # If it's a multiple-choice question, handle compact format (like 'ACD') or answers with separators
        elif question_type == "multi" or question_type == "多项选择题":
            # 1. First try the separator case, handling common separators like commas, Chinese commas, spaces, etc.
            multiple_ans = []
            try:
                split_ans = re.split(r'[，,、/\s]+', ans)  # Match Chinese/English commas, enumeration comma, and spaces
                multiple_ans = [a.strip() for a in split_ans if a.strip() in choices]
            except:
                logger.warning(f"Error splitting answer: {ans}")

            # 2. If splitting by separators doesn't work, try compact format
            if len(multiple_ans) == 0:
                # Assume the answer is in compact form like 'ACD', check character by character
                multiple_ans = [char for char in ans if char in choices]

            return multiple_ans

        # If question_type is not recognized, return empty
        return []

    def _save_results(self, corrects, results, split):
        """
        Save results and output accuracy. All results are stored in the same UUID folder.
        """
        # Check if results and corrects are empty
        if len(results) == 0 or len(corrects) == 0:
            logger.warning(f"No valid results for {split}, skipping accuracy calculation.")
            score = float('nan')  # If empty, set accuracy to NaN
        else:
            score = 100 * np.mean(corrects)
            logger.info(f"Accuracy for {split}: {score:.2f}%")

        # All results are saved to the same UUID folder
        results_path = os.path.join(self.save_path, f"results_{self.task}_{split}.json")

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": score,
                "results": results
            }, f, indent=2)


def main(args):
    evaluator = APIEvaluator(args)
    evaluator.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shot", type=int, default=5, help="Number of shots (examples)")
    parser.add_argument("--api_base", type=str, help="Base API endpoint",
                        default="http://localhost:5555/api/storage-chat-completion")
    parser.add_argument("--model_name", type=str, help="Model name to be used for evaluation",
                        default="Qwen/Qwen2-72B-Instruct")
    parser.add_argument("--task_dir", type=str, help="Directory of the task data", default="llmeval")
    parser.add_argument("--task", type=str, help="Name of the task", default="cpsyexam")
    parser.add_argument("--save_dir", type=str, help="Directory to save the results", default="results")
    args = parser.parse_args()

    main(args)
