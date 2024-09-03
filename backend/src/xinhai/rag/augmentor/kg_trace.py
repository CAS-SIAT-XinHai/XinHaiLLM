"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import copy
import logging
import re
from string import Template
from typing import List

import jsonlines
import torch

from xinhai.rag.generator import XinHaiRAGGeneratorBase
from xinhai.rag.indexer import XinHaiRAGDenseIndexer, INDEXER_REGISTRY
from xinhai.rag.augmentor import register_augmentor, XinHaiRAGAugmentorBase
from xinhai.types.kg import XinHaiKGTriplet, XinHaiKGReasoningChain
from xinhai.types.rag import XinHaiRAGAugmentorTypes, XinHaiRAGAugmentedResult, XinHaiRAGDocumentOut, XinHaiRAGIndexerTypes, \
    XinHaiRAGDocumentIn

logger = logging.getLogger(__name__)


class TripletExtraction:

    def __init__(self, config):
        self.num_references = config['num_references']

        self.indexer_type = XinHaiRAGIndexerTypes(config['indexer'].pop('type'))
        self.indexer: XinHaiRAGDenseIndexer = INDEXER_REGISTRY[self.indexer_type](config['indexer'])

        self.reference_for_index_template = Template(config['reference_for_index_template'])
        with jsonlines.open(config['reference_path']) as reader:
            documents = [XinHaiRAGDocumentIn(id=str(i),
                                             text=self.reference_for_index_template.safe_substitute(item),
                                             metadata=item) for i, item in enumerate(reader)]
            self.indexer.build_index(documents)

        self.system_prompt_template = Template(config['system_prompt_template'])
        self.user_prompt_template = Template(config['user_prompt_template'])
        self.reference_template = Template(config['reference_template'])

        self.triplet_pattern = re.compile(config['triplet_pattern'])

    def format_reference(self, documents: List[XinHaiRAGDocumentOut]):
        format_reference = []
        for idx, doc_item in enumerate(documents):
            metadata = doc_item.document.metadata
            mapping = {
                'id': doc_item.document.id,
                'idx': idx,
                'text': doc_item.document.text,
            }
            metadata.update(mapping)
            format_reference.append(self.reference_template.safe_substitute(metadata))

        return "\n\n".join(format_reference)

    def __call__(self, document: XinHaiRAGDocumentIn, generator: XinHaiRAGGeneratorBase):

        def reference_ranking(query):
            docs_with_scores = self.indexer.vectorstore.similarity_search_with_score(query,
                                                                                     k=self.num_references)

            return [XinHaiRAGDocumentOut(
                document=XinHaiRAGDocumentIn(
                    id=str(doc.metadata.get('id', doc.id)),
                    metadata=doc.metadata,
                    text=doc.page_content
                ),
                score=score
            ) for doc, score in docs_with_scores]

        references = reference_ranking(document.text)
        formatted_reference = self.format_reference(references)
        input_params = {"text": document.text, "reference": formatted_reference}
        input_params.update(document.metadata)

        system_prompt = self.system_prompt_template.safe_substitute(input_params)
        logger.debug(system_prompt)

        user_prompt = self.user_prompt_template.safe_substitute(input_params)
        logger.debug(user_prompt)

        prompts = XinHaiRAGAugmentedResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        while True:
            triplets_text = generator.generate(prompts, max_tokens=512)
            logger.debug(triplets_text)

            triplets = []
            for item in self.triplet_pattern.finditer(triplets_text):
                triplets.append(
                    XinHaiKGTriplet.model_validate(item.groupdict())
                )
            logger.debug(triplets)
            if triplets:
                break
        return triplets


class ReasoningChain:

    def __init__(self, config):
        self.num_chains = config["num_chains"]
        self.num_choices = config["num_choices"]
        self.max_chain_length = config["max_chain_length"]
        self.num_references = config["num_references"]
        self.max_chain_length = config["max_chain_length"]

        self.indexer_type = XinHaiRAGIndexerTypes(config['indexer'].pop('type'))
        self.indexer: XinHaiRAGDenseIndexer = INDEXER_REGISTRY[self.indexer_type](config['indexer'])

        self.reference_for_index_template = Template(config['reference_for_index_template'])
        with jsonlines.open(config['reference_reasoning_chain_path']) as chain_reader, jsonlines.open(
                config['reference_reasoning_steps_path']) as steps_reader:
            documents = []
            for i, (chain_item, steps_item) in enumerate(zip(chain_reader, steps_reader)):
                metadata = chain_item
                metadata.update({"reasoning_steps": steps_item})
                documents.append(XinHaiRAGDocumentIn(id=str(i),
                                                     text=self.reference_for_index_template.safe_substitute(metadata),
                                                     metadata=metadata))
            self.indexer.build_index(documents)

        self.system_prompt_template = Template(config['system_prompt_template'])
        self.user_prompt_template = Template(config['user_prompt_template'])
        self.reference_template = Template(config['reference_template'])

    def format_reference(self, documents: List[XinHaiRAGDocumentOut], hop):
        format_reference = []
        for idx, doc_item in enumerate(documents):
            metadata = doc_item.document.metadata
            if len(metadata['reasoning_steps']) > hop:
                mapping = {
                    'id': doc_item.document.id,
                    'idx': idx,
                    'hop': hop,
                    'text': doc_item.document.text,
                }
                metadata.update(mapping)
                metadata.update(metadata['reasoning_steps'][hop])
                format_reference.append(self.reference_template.safe_substitute(metadata))

        return "\n".join(format_reference)

    def retrieve_triplet_candidates(self, triplets, chains):
        logger.debug(triplets)
        logger.debug(chains)
        triplets_embedding = self.indexer.vectorstore.embeddings.embed_documents([str(t) for t in triplets])
        queries_embeddings = self.indexer.vectorstore.embeddings.embed_documents([str(c) for c in chains])
        queries_triples_similarities = torch.matmul(torch.tensor(queries_embeddings),
                                                    torch.tensor(triplets_embedding).T)  # n_path, n_triples

        candidate_triples_mask = torch.ones_like(queries_triples_similarities)
        for k, chain in enumerate(chains):
            path = [triplets.index(t) for t in chain.triplets]
            candidate_triples_mask[k, path] = 0.0
        queries_triples_similarities = queries_triples_similarities + \
                                       torch.finfo(queries_triples_similarities.dtype).min * (
                                               1.0 - candidate_triples_mask)
        topk_most_relevant_triples_indices = \
            torch.topk(queries_triples_similarities, k=min(self.num_choices, len(triplets)), dim=1)[1].tolist()
        logger.debug(topk_most_relevant_triples_indices)

        candidate_triples = [
            [triplets[idx] for idx in candidate_triples_indices] \
            for candidate_triples_indices in topk_most_relevant_triples_indices
        ]
        logger.debug(candidate_triples)
        return candidate_triples

    @staticmethod
    def format_triplet_candidates(candidate_triples):
        return "\n".join(["A. no need for additional knowledge triples"] \
                         + ["{}. {}".format(chr(ord('B') + k), triple) for k, triple in enumerate(candidate_triples)])

    def __call__(self, query, triplets: List[XinHaiKGTriplet], generator: XinHaiRAGGeneratorBase):
        def reference_ranking():
            docs_with_scores = self.indexer.vectorstore.similarity_search_with_score(query,
                                                                                     k=self.num_references)

            return [XinHaiRAGDocumentOut(
                document=XinHaiRAGDocumentIn(
                    id=str(doc.metadata.get('id', doc.id)),
                    metadata=doc.metadata,
                    text=doc.page_content
                ),
                score=score
            ) for doc, score in docs_with_scores]

        references = reference_ranking()
        logger.debug(references)

        chains = [XinHaiKGReasoningChain(query=query)]
        for j in range(self.max_chain_length):
            if sum([c.is_complete for c in chains]) == self.num_chains or all([c.is_complete for c in chains]):
                break

            # TODO: beam search over different candidate score
            new_chains = []
            for chain, candidate_triplets in zip(chains, self.retrieve_triplet_candidates(triplets, chains)):
                if chain.is_complete:
                    new_chains.append(chain)
                    continue

                existing_triplets = chain.triplets
                logger.debug(existing_triplets)
                logger.debug(candidate_triplets)

                input_params = {
                    "query": query,
                    "hop": j,
                    "reference": self.format_reference(references, hop=j),
                    "existing_triplets": existing_triplets,
                    "candidate_triplets": self.format_triplet_candidates(candidate_triplets),
                }
                system_prompt = self.system_prompt_template.safe_substitute(input_params)
                logger.debug(system_prompt)

                user_prompt = self.user_prompt_template.safe_substitute(input_params)
                logger.debug(user_prompt)

                prompts = XinHaiRAGAugmentedResult(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                generate_output = generator.generate(prompts, max_tokens=32, return_dict=True)

                prob = 1
                if generate_output.startswith('A'):
                    chain.is_complete = True
                    chain.score *= prob
                    new_chains.append(chain)
                else:
                    option = ord(generate_output[0]) - ord('B')
                    if len(candidate_triplets) > option:
                        new_chain = copy.deepcopy(chain)
                        new_chain.score *= prob
                        new_chain.triplets.append(candidate_triplets[option])
                        new_chains.append(new_chain)
                    else:
                        new_chains.append(chain)

            chains = new_chains

        return [chain for chain in chains if chain.is_complete]


@register_augmentor(XinHaiRAGAugmentorTypes.KG_TRACE)
class KGTraceAugmentor(XinHaiRAGAugmentorBase):
    name = XinHaiRAGAugmentorTypes.KG_TRACE
    share_generator = True
    generator: XinHaiRAGGeneratorBase

    def __init__(self, config):
        super().__init__(config)
        # load demonstrations for generating triples and reasoning chain
        self.triplets_extraction = TripletExtraction(config['triplets_extraction'])
        self.reasoning_chain = ReasoningChain(config['reasoning_chain'])

    def _augment(self, query: str, retrieved_documents: List[XinHaiRAGDocumentOut], *args,
                 **kwargs) -> XinHaiRAGAugmentedResult:
        logger.debug("Begin extracting triples")
        triplets = []
        for doc in retrieved_documents:
            logger.debug(doc)
            triplets.extend(self.triplets_extraction(doc.document, self.generator))
        logger.debug(triplets)
        logger.debug("Finish extracting triples")

        logger.debug("Begin generating reasoning chain")
        chains = self.reasoning_chain(query, triplets, self.generator)
        logger.debug(chains)
        sentences = []
        for j, chain in enumerate(chains):
            sentences.extend([f"{j}-{i + 1}. {triplet.as_sentence()}" for i, triplet in enumerate(chain.triplets)])
        context = "\n".join(sentences)
        logger.debug("Finish generating reasoning chain")

        input_params = {
            "query": query,
            "context": context,
        }

        system_prompt = self.system_prompt_template.safe_substitute(input_params)
        logger.debug(system_prompt)

        user_prompt = self.user_prompt_template.safe_substitute(input_params)
        logger.debug(user_prompt)
        return XinHaiRAGAugmentedResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
