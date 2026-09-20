from __future__ import annotations

import json

from .evaluation import EvalCase, evaluate
from .models import AccessContext, Document
from .service import SecureRAGService


def main() -> None:
    service = SecureRAGService()
    service.ingest(
        Document(
            id="public-runbook",
            tenant_id="acme",
            title="Checkout latency runbook",
            text=(
                "If checkout P95 exceeds 800 ms, inspect database pool saturation "
                "and cache misses."
            ),
            allowed_groups=frozenset({"engineering"}),
        )
    )
    service.ingest(
        Document(
            id="finance-plan",
            tenant_id="acme",
            title="Confidential finance plan",
            text="The confidential acquisition budget is 42 million dollars.",
            allowed_groups=frozenset({"finance"}),
        )
    )
    engineer = AccessContext("alice", "acme", frozenset({"engineering"}))
    result = service.query("How should I debug checkout latency?", engineer)
    print(result.answer)
    metrics = evaluate(
        service,
        [
            EvalCase(
                query="checkout latency database cache",
                context=engineer,
                relevant_document_id="public-runbook",
                forbidden_document_ids=frozenset({"finance-plan"}),
            )
        ],
    )
    print(json.dumps(metrics, indent=2))
