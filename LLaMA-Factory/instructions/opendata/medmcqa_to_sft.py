import argparse
import json
import logging

from datasets import load_dataset
from tqdm import tqdm
from common.second_handle import *

"""
{
    "question":"A 40-year-old man presents with 5 days of productive cough and fever. Pseudomonas aeruginosa is isolated from a pulmonary abscess. CBC shows an acute effect characterized by marked leukocytosis (50,000 mL) and the differential count reveals a shift to left in granulocytes. Which of the following terms best describes these hematologic findings?",
    "exp": "Circulating levels of leukocytes and their precursors may occasionally reach very high levels (>50,000 WBC mL). These extreme elevations are sometimes called leukemoid reactions because they are similar to the white cell counts observed in leukemia, from which they must be distinguished. The leukocytosis occurs initially because of the accelerated release of granulocytes from the bone marrow (caused by cytokines, including TNF and IL-1) There is a rise in the number of both mature and immature neutrophils in the blood, referred to as a shift to the left. In contrast to bacterial infections, viral infections (including infectious mononucleosis) are characterized by lymphocytosis Parasitic infestations and certain allergic reactions cause eosinophilia, an increase in the number of circulating eosinophils. Leukopenia is defined as an absolute decrease in the circulating WBC count.",
    "cop":1,
    "opa":"Leukemoid reaction",
    "opb":"Leukopenia",
    "opc":"Myeloid metaplasia",
    "opd":"Neutrophilia",
    "subject_name":"Pathology",
    "topic_name":"Basic Concepts and Vascular changes of Acute Inflammation",
    "id":"4e1715fe-0bc3-494e-b6eb-2d4617245aef",
    "choice_type":"single"
}
"""


def convert(opts):
    dataset = load_dataset("medmcqa", split='train')
    formatted_data = []
    for entry in tqdm(dataset):
        if entry['topic_name'] is not None:
            prompt = f"This is a question about {entry['subject_name']} from real-world medical entrance exam.\nThe topic is {entry['topic_name']}.\n\n"
        else:
            prompt = f"This is a question about {entry['subject_name']} from real-world medical entrance exam.\n\n"
        choices = ['A', 'B', 'C', 'D']

        if entry['exp'] is not None:
            output = f"{entry['exp']}\nThe answer is {choices[entry['cop']]} ."
        else:
            output = f"The answer is {choices[entry['cop']]} ."

        formatted_entry = dict(instruction=prompt,
                               input=f"{entry['question']}\nA: {entry['opa']}\nB: {entry['opb']}\nC: {entry['opc']}\nD: {entry['opd']}\n",
                               output=output)

        formatted_data.append(formatted_entry)

    result_data = remove_repeate(formatted_data)
    with open(opts.output_file, 'w', encoding='utf-8') as out_file:
        json.dump(result_data, out_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='MedMCQA', description='')
    parser.add_argument("--output_file", type=str, default="../../data/medmcqa.json")
    args = parser.parse_args()
    convert(args)
