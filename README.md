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

## Quick start

```bash
git clone https://github.com/asher0913/secure-rag && cd secure-rag
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
secure-rag demo && secure-rag benchmark --out runs/benchmark.json
```

This needs Python 3.10+ and nothing else: no model download, database or GPU. `secure-rag demo`
prints one query and its audit record. `secure-rag benchmark` reruns both tables below. CI runs the
benchmark on every push and requires it to equal `results/benchmark.json` exactly.

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

## Evidence and CI coverage

| Result | Kind of evidence | File | Rerun in CI? |
|---|---|---|---|
| Both design tables | seeded synthetic corpus (2 tenants, 72 users, 440 documents), leakage judged against the directory of record | `results/benchmark.json` | Yes: must match exactly |
| ACLs fail closed; unauthorized text never ranked; revocation, deletion and team moves enforced | unit tests | `tests/` (12 tests) | Yes, on every push |
| Scores unchanged when another tenant adds documents | unit test | `tests/` | Yes |

## Design trade-offs

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Where to filter | before ranking, on the index's ACL copy | after ranking | Post-filtering leaves 25% of contexts empty (table above). |
| Freshness | live re-check of every candidate against the directory of record, then refill | trust the index until the next sync | A stale pre-filter leaked on 54% of queries after permission changes; the re-check brings that to 0%. It costs one directory lookup per candidate. |
| Ranking statistics | BM25 statistics from the caller's authorized view | statistics from the shared index | Shared statistics make scores depend on other tenants' documents, which is a side channel. |
| Failure mode | deny when the directory cannot be reached | serve from the index | A retrieval system that fails open leaks during outages. |

## Code map

| File | What to look at |
|---|---|
| `src/secure_rag/service.py` | `SecureRAGService.query`: pre-filter → hybrid ranking → `live_check` → refill → audit event |
| `src/secure_rag/models.py` | `authorized()`, the fail-closed ACL rule; `AccessContext`, `Document` |
| `src/secure_rag/retrieval.py` | `HybridRetriever.search`: BM25 + hashed embeddings over the authorized chunks only |
| `src/secure_rag/benchmark.py` | the seeded world, the permission changes and the five designs (`_run_design`) |
| `src/secure_rag/api.py` | FastAPI endpoints for ingest, query and audit |

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

## Known issues

The retrieval core above is what the benchmark measures. The HTTP layer is a demo and is not yet
safe to expose:

- **Identity is taken from the request body.** `POST /v1/query` trusts the `user_id`, `tenant_id`
  and `groups` the client sends. In production they must come from a verified session token.
  The benchmark and the service's `query()` already take an `AccessContext` built by the caller.
- **The write and audit endpoints have no access control.** Anyone who can reach the service can
  ingest documents or read every tenant's audit log.
- **The live re-check compares ACLs and existence, not content.** A document that was edited
  after indexing passes the re-check with its old text, because document versions are not
  compared.

## License

MIT
