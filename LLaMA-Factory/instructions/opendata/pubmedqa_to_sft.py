import argparse
import json
import logging

from datasets import load_dataset
from tqdm import tqdm
from common.second_handle import *

"""
All PubMed articles with question titles:
pqa_labeled:  manually labeled 1k of them for cross-validation and testing. 
pqa_unlabeled: 61.2k yes/no/answerable QA instances compose of the unlabeled subset which can be used for semisupervised learning. 
pqa_artificial: 211k automatically convert statement titles of 211.3k PubMed articles to questions and label them with yes/no answers using a simple heuristic. These artificially generated instances can be used for pre-training.

{
  "pubid": 21645374,
  "question": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
  "context": {
    "contexts": [
      "Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.",
      "The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (\u0394\u03a8m). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells."
    ],
    "labels": [
      "BACKGROUND",
      "RESULTS"
    ],
    "meshes": [
      "Alismataceae",
      "Apoptosis",
      "Cell Differentiation",
      "Mitochondria",
      "Plant Leaves"
    ],
    "reasoning_required_pred": [
      "y",
      "e",
      "s"
    ],
    "reasoning_free_pred": [
      "y",
      "e",
      "s"
    ]
  },
  "long_answer": "Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation of this organelle with other organelles during developmental PCD. To the best of our knowledge, this is the first report of mitochondria and chloroplasts moving on transvacuolar strands to form a ring structure surrounding the nucleus during developmental PCD. Also, for the first time, we have shown the feasibility for the use of CsA in a whole plant system. Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.",
  "final_decision": "yes"
}
"""


def convert(opts):
    formatted_data = []
    for name in ['pqa_labeled', 'pqa_unlabeled', 'pqa_artificial']:
        dataset = load_dataset("pubmed_qa", name, split='train')
        for i, entry in enumerate(tqdm(dataset)):
            if i == 0:
                print(json.dumps(entry, indent=2))

            if 'final_decision' in entry:
                output = f"{entry['long_answer']}\nThe final decision is {entry['final_decision']} ."
            else:
                output = entry['long_answer']

            context = '\n'.join(entry['context']['contexts'])
            formatted_entry = dict(instruction="Read the context and answer the question:\n\n",
                                   input=f"{context}\n{entry['question']}\n",
                                   output=output)

            formatted_data.append(formatted_entry)

    result_data = remove_repeate(formatted_data)
    with open(f"{opts.output_file}.json", 'w', encoding='utf-8') as out_file:
        json.dump(result_data, out_file, ensure_ascii=False, indent=4)
    print("==================Start generate v2")
    sft_entries_v2 = process_data('llama', result_data)
    sft_entries_v2 = remove_llm_other(sft_entries_v2)
    with open(f"{opts.output_file}_v2.json", 'w', encoding='utf-8') as json_file_v2:
        json.dump(sft_entries_v2, json_file_v2, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(prog='PubMedQA', description='')
    parser.add_argument("--output_file", type=str, default="../../data/pubmedqa")
    args = parser.parse_args()
    convert(args)
