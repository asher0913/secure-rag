from __future__ import annotations

from dataclasses import dataclass

from .models import AccessContext
from .service import SecureRAGService


@dataclass(frozen=True)
class EvalCase:
    query: str
    context: AccessContext
    relevant_document_id: str | None
    forbidden_document_ids: frozenset[str] = frozenset()


def evaluate(service: SecureRAGService, cases: list[EvalCase], top_k: int = 4) -> dict[str, float]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    leaks = 0
    returned = 0
    for case in cases:
        result = service.query(case.query, case.context, top_k=top_k)
        ids = [hit.chunk.document_id for hit in result.hits]
        returned += len(ids)
        leaks += sum(item in case.forbidden_document_ids for item in ids)
        if case.relevant_document_id is not None:
            recalls.append(float(case.relevant_document_id in ids))
            reciprocal_ranks.append(
                1 / (ids.index(case.relevant_document_id) + 1)
                if case.relevant_document_id in ids
                else 0.0
            )
    return {
        "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "authorization_leak_rate": leaks / max(returned, 1),
    }

