"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from typing import List

from pydantic import BaseModel


class XinHaiKGTriplet(BaseModel):
    head: str
    relation: str
    tail: str

    def __str__(self):
        return f"<{self.head}; {self.relation}; {self.tail}>"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, XinHaiKGTriplet):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def as_sentence(self):
        return f"{self.head} {self.relation} {self.tail}"


class XinHaiKGReasoningChain(BaseModel):
    query: str
    triplets: List[XinHaiKGTriplet] = []
    score: float = 1
    is_complete: bool = False

    def __str__(self):
        return (f"Knowledge triplets: {' '.join(map(str, self.triplets))}"
                f"Query: {self.query}")

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, XinHaiKGReasoningChain):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))
