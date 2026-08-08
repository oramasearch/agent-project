"""Few-shot retrieval over the training corpus.

The corpus was built to train a model. Until that model exists it earns its keep a different way:
as the source of the examples this agent is shown at inference time. Same artifact, two uses — and
the retrieval path is a cheap, honest check on whether the corpus is any good, because a corpus
that retrieves useless neighbours will not train a useful model either.

Similarity is over the SITUATION — industry, table names, column names, capture state — not over
the answer. Retrieving by answer would show the model what it is supposed to conclude, which is
leakage in the shape of a helpful prompt.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _terms(text: str) -> Counter[str]:
    return Counter(word for word in re.findall(r"[a-z_]{3,}", text.lower()))


def _situation_text(example: dict[str, Any]) -> str:
    """The situation only.

    Includes the INTENT and, for a mutation, the pipeline being changed — both are inputs this
    agent is handed, so matching on them is matching on the situation. Deliberately excludes the
    trajectory, the pipeline it produced and the reply: those are the answer, and retrieving by
    the answer shows the model what it is supposed to conclude.
    """
    profile = example.get("profile") or example
    intent_and_before = [str(example.get("intent", ""))]
    for key, value in (example.get("before") or {}).items():
        if value:
            intent_and_before.append(str(key))
    parts = [str(profile.get("industry", ""))]
    for connection in profile.get("connections", []) or []:
        parts.append(f"{connection.get('kind','')} {connection.get('state','')}")
    for table in profile.get("tables", []) or []:
        parts.append(f"{table.get('database','')} {table.get('name','')}")
        for column in table.get("columns", []) or []:
            parts.append(column["name"] if isinstance(column, dict) else str(column))
    for constraint in (profile.get("persona") or {}).get("constraints", []) or []:
        parts.append(str(constraint))
    return " ".join(parts + intent_and_before)


class Corpus:
    """A retrievable corpus of worked examples."""

    def __init__(self, path: Path) -> None:
        self.examples: list[dict[str, Any]] = []
        self.vectors: list[Counter[str]] = []
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.examples.append(example)
            self.vectors.append(_terms(_situation_text(example)))
        # Inverse document frequency, so `account_id` appearing everywhere counts for less than
        # `hydraulic_temp` appearing twice. Without it every example looks equally similar and the
        # retrieval degenerates to "the first k".
        documents = len(self.vectors) or 1
        appearances = Counter(term for vector in self.vectors for term in vector)
        self.idf = {term: math.log(documents / (1 + count)) + 1.0
                    for term, count in appearances.items()}

    def _score(self, query: Counter[str], other: Counter[str]) -> float:
        shared = set(query) & set(other)
        if not shared:
            return 0.0
        numerator = sum(query[t] * other[t] * self.idf.get(t, 1.0) ** 2 for t in shared)
        norm = math.sqrt(sum((query[t] * self.idf.get(t, 1.0)) ** 2 for t in query)) * \
               math.sqrt(sum((other[t] * self.idf.get(t, 1.0)) ** 2 for t in other))
        return numerator / norm if norm else 0.0

    def nearest(self, profile: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
        """The k most similar SITUATIONS. Empty when the corpus is empty — the caller must then
        decide whether to proceed few-shot-less rather than be handed silence."""
        if not self.examples:
            return []
        query = _terms(_situation_text(profile))
        ranked = sorted(
            ((self._score(query, vector), index) for index, vector in enumerate(self.vectors)),
            reverse=True)
        return [self.examples[index] for score, index in ranked[:k] if score > 0.0]
