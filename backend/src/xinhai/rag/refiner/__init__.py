"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import importlib
import logging
import os
from string import Template
from typing import List, Dict

from xinhai.types.rag import XinHaiRAGRefinerTypes, XinHaiRAGRetrievedResult

logger = logging.getLogger(__name__)

REFINER_REGISTRY = {}


def register_refiner(name, subname=None):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
        :param name:
        :param subname:
    """

    def register_refiner_cls(cls):
        if subname is None:
            if name in REFINER_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            REFINER_REGISTRY[name] = cls
        else:
            if name in REFINER_REGISTRY and subname in REFINER_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            REFINER_REGISTRY.setdefault(name, {})
            REFINER_REGISTRY[name][subname] = cls
        return cls

    return register_refiner_cls


class XinHaiRAGRefinerBase:
    r"""Base object of Refiner method"""
    name: XinHaiRAGRefinerTypes

    def __init__(self, config):
        # self.config = config
        # self.model_path = config["refiner_model_path"]
        # self.device = config["device"]
        # self.input_prompt_flag = config["refiner_input_prompt_flag"] if "refiner_input_prompt_flag" in config else False
        # DEFAULT_PATH_DICT = {
        #     "recomp_abstractive_nq": "fangyuan/nq_abstractive_compressor",
        #     "recomp:abstractive_tqa": "fangyuan/tqa_abstractive_compressor",
        #     "recomp:abstractive_hotpotqa": "fangyuan/hotpotqa_abstractive",
        # }
        #
        # refiner_path = config["refiner_model_path"]
        #
        # if refiner_path is None:
        #     refiner_path = DEFAULT_PATH_DICT.get(refiner_name)
        #
        # if refiner_path is None:
        #     raise ValueError("Refiner path is not specified and no default path available!")
        #
        # model_config = AutoConfig.from_pretrained(refiner_path)
        # arch = model_config.architectures[0].lower()
        #
        # if "recomp" in refiner_name or "recomp" in refiner_path or "bert" in arch:
        #     if model_config.model_type == "t5":
        #         refiner_class = "AbstractiveRecompRefiner"
        #     else:
        #         refiner_class = "ExtractiveRefiner"
        # elif "lingua" in refiner_name:
        #     refiner_class = "LLMLinguaRefiner"
        # elif "selective-context" in refiner_name or "sc" in refiner_name:
        #     refiner_class = "SelectiveContextRefiner"
        # elif "kg-trace" in refiner_name:
        #     return getattr(REFINER_MODULE, "KGTraceRefiner")(config, retriever, generator)
        # else:
        #     raise ValueError("No implementation!")
        self.system_prompt_template = Template(config['system_prompt_template'])
        self.user_prompt_template = Template(config['user_prompt_template'])
        self.reference_template = Template(config['reference_template'])

    def _refine(self, query: str, retrieval_result: XinHaiRAGRetrievedResult, *args, **kwargs) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """
        pass

    def refine(self, *args, **kwargs):
        return self._refine(*args, **kwargs)


# automatically import any Python files in the models/ directory
refiner_dir = os.path.dirname(__file__)
for file in os.listdir(refiner_dir):
    path = os.path.join(refiner_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.rag.refiner.{model_name}')
