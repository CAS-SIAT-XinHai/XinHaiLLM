"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import asyncio
import logging
import os
from argparse import ArgumentParser
from typing import List, Dict

import datasets
import jsonlines
import yaml
from pydantic import BaseModel

from xinhai.rag.methods import RAG_REGISTRY, XinHaiRAGMethodBase
# from xinhai.rag.metrics import XinHaiRAGMetricBase, METRIC_REGISTRY
from xinhai.types.rag import XinHaiRAGDocumentIn, XinHaiRAGMethodTypes

logger = logging.getLogger(__name__)


class Dataset(BaseModel):
    splits: Dict[str, List[dict]]
    corpus_path: str


class Evaluation:
    def __init__(self, dataset: Dataset,
                 rag: XinHaiRAGMethodBase,
                 # metrics: List[XinHaiRAGMetricBase]
                 ):
        self.dataset = dataset
        self.rag = rag
        # self.metrics = metrics

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method."""
        logger.info("Creating index")
        documents = [XinHaiRAGDocumentIn(id=str(item['id']),
                                         text=item['contents'],
                                         metadata=item) for item in
                     datasets.load_dataset("json", data_files=self.dataset.corpus_path, split="train", num_proc=4)]
        self.rag.retriever.indexer.build_index(documents)
        logger.info("Finish!")

    @staticmethod
    def load_dataset(config):
        """Load dataset from config."""

        dataset_path = config["path"]
        all_split = config["splits"]

        split_dict = {split: [] for split in all_split}

        for split in all_split:
            split_path = os.path.join(dataset_path, f"{split}.jsonl")
            if not os.path.exists(split_path):
                print(f"{split} file not exists!")
                continue

            with jsonlines.open(split_path, "r") as f:
                split_dict[split] = [item for item in f]

        return Dataset(splits=split_dict, corpus_path=config['corpus_path'])

    @classmethod
    def from_config(cls, config_path):
        config = yaml.safe_load(open(config_path))

        dataset_config = config['dataset']
        dataset = cls.load_dataset(dataset_config)

        rag_config = config["rag"]
        rag_method = XinHaiRAGMethodTypes(rag_config.pop("method"))
        rag = RAG_REGISTRY[rag_method](rag_config)

        # eval_config = config["evaluation"]
        # metrics = [METRIC_REGISTRY[XinHaiRAGMetricTypes(m)](eval_config) for m in eval_config["metrics"]]
        return cls(
            dataset,
            rag,
            # metrics
        )

    def run(self):
        """Run the environment from scratch until it is done."""
        self.build_index()
        for split, data in self.dataset.splits.items():
            for item in data:
                logger.debug("============================================================================")
                ret = asyncio.run(
                    self.rag(XinHaiRAGDocumentIn(id=str(item['id']), text=item['question'], metadata=item)))
                logger.debug(ret)

            # result_dict = {}
            # for metric in self.metrics:
            #     metric_result, metric_scores = metric.calculate_metric(data)
            #     result_dict.update(metric_result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/xinhai.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%Y-%m-%d:%H:%M:%S',
                            level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(level=logging.INFO)

    evaluator = Evaluation.from_config(args.config_path)
    evaluator.run()
