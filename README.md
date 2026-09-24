# SecureRAG

[![CI](https://github.com/asher0913/secure-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/asher0913/secure-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Permission-aware retrieval for multi-tenant RAG. Text a user may not read never reaches ranking
or the model's context. That still holds when the index and the user's session lag behind the
directory of record.

```text
session token ─► pre-filter on the index's ACL copy ─► hybrid BM25 + embedding ranking ─► live re-check ─► context + citations
 (user, groups)   tenant + explicit grant, fail-closed    statistics from the authorized     against the directory     audit event:
                                                          view only                          of record, with refill    returned + denied ids
```

## Results

Where should permissions be enforced? Five designs run on the same seeded corpus: two tenants
with the same five departments, 72 users, 440 documents (30 of them readable only by named
people), and eight project codenames that recur across teams and tenants. There are three kinds
of query:

- **specific**: "budget forecast falcon q3 2024";
- **broad**: "latest on project falcon", answered by any readable document about the project;
- **probes**, written from documents the user may **not** read.

Leakage is always judged against the directory of record.

**Everything in sync** (648 queries):

| Design | Queries that leak | Cross-tenant | Broad questions answered | Empty context | Recall@4 |
|---|---:|---:|---:|---:|---:|
| no filter | 98.1% | 89.0% | 20.1% | 0% | 1.000 |
| post-filter (rank everything, drop forbidden) | 0% | 0% | 20.1% | **25.2%** | 1.000 |
| post-filter, 4× over-fetch | 0% | 0% | 81.2% | 4.4% | 1.000 |
| pre-filter on the index's ACLs | 0% | 0% | **100%** | 0% | 1.000 |
| pre-filter + live re-check (SecureRAG) | 0% | 0% | **100%** | 0% | 1.000 |

**After changes, before the next index sync** (634 queries). 13 people moved teams, 37 documents
were restricted to a named owner, and 18 were deleted. The index still has the old ACLs and
documents, and sessions still carry the old groups. Probes target exactly what users lost
access to.

| Design | Queries that leak | Forbidden chunks per 100 queries | Broad questions answered | Recall@4 |
|---|---:|---:|---:|---:|
| no filter | 97.6% | 277 | 25.0% | 0.997 |
| post-filter | 35.5% | 46 | 20.8% | 0.882 |
| post-filter, 4× over-fetch | 45.1% | 73 | 75.0% | 0.885 |
| pre-filter on the index's ACLs | **54.1%** | 92 | 99.3% | 0.885 |
| **pre-filter + live re-check** | **0%** | 0 | 100% | 0.885 |

- **Post-filtering starves the context.** When a user can read a small share of the corpus,
  the global top-k is mostly other people's documents. Dropping them leaves a quarter of
  questions with an empty context, and broad questions are answered 20% of the time.
  Over-fetching helps (81%) but does not fix it. Filtering before ranking does (100%).
- **A pre-filter is only as fresh as the index.** Once permissions change, the best-ranking
  design leaks the most: 54% of queries return text the user just lost access to, because it
  faithfully ranks everything the stale ACL copy allows.
- **The live re-check closes the gap.** Before context assembly, each candidate is checked
  against the directory of record: the current document ACL, whether the document still exists,
  and the user's current groups. The context is then refilled from further down the list. Leaks
  drop to zero and broad questions are still all answered. If the directory cannot be reached,
  the candidate is dropped.
- **Stale sessions fail closed.** Recall@4 falls from 1.0 to 0.885 for every filtered design,
  because people who moved teams still carry their old groups. They cannot see their new team's
  documents until the session is refreshed, which is the safe direction to be wrong in.

The corpus is synthetic and small; it isolates the enforcement question from retrieval quality.
The "no filter" recall of 1.0 is an artefact of lexical queries naming their document; it is
there to show the leak rate, not as a retrieval baseline.

## Design details

- **Fail-closed ACLs.** A document is readable only by an explicit grant to the user, one of
  their groups, or `everyone` in the same tenant. An empty ACL grants nobody, and tenants never
  cross.
- **Unauthorized text is never ranked.** The retriever only sees the caller's authorized chunks,
  and BM25 statistics come from that view. With statistics from the whole shared index, a user's
  scores would move when another tenant adds documents, which is a side channel. A test checks
  that scores are unchanged when another tenant adds twenty documents.
- **Audit.** Every query records the returned and the denied document ids, with a trace id.
- **Swappable parts.** The store, retriever and authority are interfaces. Postgres with
  row-level security, a vector database's metadata filters, BGE or E5 embeddings, and OPA or an
  IdP as the authority slot in without moving the security boundary.

## Usage

```bash
pip install -e '.[dev]'

secure-rag demo                                 # two documents, one engineer, the audit record
secure-rag benchmark --out results/benchmark.json
uvicorn secure_rag.api:app                      # POST /v1/documents, POST /v1/query, GET /v1/audit
```

```python
from secure_rag import AccessContext, Document, SecureRAGService

service = SecureRAGService(authority=my_directory)   # .document(id) and .groups(tenant, user)
service.ingest(Document("runbook", "acme", "Checkout runbook", "Inspect DB pool saturation ...",
                        allowed_groups=frozenset({"engineering"})))
result = service.query("checkout latency", AccessContext("alice", "acme", frozenset({"engineering"})))
```

## Tests

`pytest -q` runs 12 tests:

- ACLs fail closed;
- the retriever never receives unauthorized chunks;
- the live re-check enforces a revoked grant, a deleted document and a team move that the index
  and the session have not seen, and records the denials;
- an unreachable directory denies everything;
- refill keeps the context full after denials;
- scores do not depend on other tenants;
- the Recall@K/MRR/leakage regression suite passes;
- the benchmark headline holds;
- the HTTP API round-trips.

## Limitations

- The live re-check costs one directory lookup per candidate. In production, batch it and cache
  it for seconds, not minutes; the cache TTL is the new staleness window.
- Retrieval is lexical plus hashed embeddings. With a dense index, pre-filtering needs a vector
  database that filters during the ANN search, or recall falls the way post-filtering's does here.
- Chunk text is trusted once authorized. Prompt injection inside authorized documents is out of
  scope.

## License

MIT
