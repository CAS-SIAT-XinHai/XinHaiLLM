"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import os
import shutil
import subprocess
import tempfile

from xinhai.rag.indexer import XinHaiRAGIndexerBase
from xinhai.rag.indexer import register_indexer
from xinhai.types.rag import XinHaiRAGIndexerTypes


@register_indexer(XinHaiRAGIndexerTypes.LUCENE)
class XinHaiRAGLuceneIndexer(XinHaiRAGIndexerBase):
    name = XinHaiRAGIndexerTypes.LUCENE

    def __init__(self, config):
        super().__init__(config)

    def build_index(self, corpus_path):
        """Building BM25 index based on Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """
        # to use pyserini pipeline, we first need to place jsonl file in the folder
        with tempfile.TemporaryDirectory() as tmp_dirname:
            shutil.copyfile(corpus_path, os.path.join(tmp_dirname, 'temp.jsonl'))

            print("Start building bm25 index...")
            pyserini_args = [
                "--collection",
                "JsonCollection",
                "--input",
                tmp_dirname,
                "--index",
                self.index_path,
                "--generator",
                "DefaultLuceneDocumentGenerator",
                "--threads",
                "1",
            ]

            subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)
