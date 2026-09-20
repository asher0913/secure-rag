# SecureRAG

[![CI](https://github.com/asher0913/secure-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/asher0913/secure-rag/actions/workflows/ci.yml)

Authorization-aware hybrid retrieval for enterprise knowledge bases. SecureRAG prevents text from
another tenant, department or revoked document from entering retrieval ranking or LLM context.

## Why this is not another PDF chatbot

```text
identity -> tenant/ACL pre-filter -> hybrid retrieval -> live ACL post-check -> citations -> audit
```

- Fail-closed tenant, user and group authorization.
- Permission checks both before retrieval and before context assembly.
- Version-aware revocation behavior.
- BM25-style lexical retrieval plus a swappable embedding interface.
- Recall@K, MRR and authorization-leakage evaluation.
- Citation and denial audit events for every query.

The default implementation is dependency-light and fully local. Replace the in-memory store and
hashing embedder with Postgres/pgvector, Milvus, BGE or E5 without changing the security boundary.

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run secure-rag-demo
uv run uvicorn secure_rag.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API explorer.

## Example

```python
from secure_rag import AccessContext, Document, SecureRAGService

service = SecureRAGService()
service.ingest(Document(
    id="runbook",
    tenant_id="acme",
    title="Checkout runbook",
    text="Inspect DB pool saturation when P95 latency spikes.",
    allowed_groups=frozenset({"engineering"}),
))
result = service.query(
    "How do I debug checkout latency?",
    AccessContext("alice", "acme", frozenset({"engineering"})),
)
print(result.answer)
```

## Security regression suite

The tests cover cross-tenant isolation, department ACLs, permission revocation, Recall@K and
forbidden-document leakage. The included fixture currently reports leakage `0.0`; this is a small
regression test, not a production security certification.

## Production roadmap

- OIDC identity and external policy decisions through OPA.
- Postgres row-level security and immutable audit storage.
- Vector-database metadata filters plus permission-aware refill.
- Prompt-injection scanning and grounded generation evaluation.

## License

MIT

