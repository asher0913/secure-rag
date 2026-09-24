import json
from pathlib import Path

import pytest

from secure_rag import EVERYONE, AccessContext, Document, SecureRAGService, authorized
from secure_rag.evaluation import EvalCase, evaluate
from secure_rag.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
ENGINEER = AccessContext("alice", "acme", frozenset({"engineering"}))
FINANCE = AccessContext("bob", "acme", frozenset({"finance"}))


def _docs():
    return [
        Document("runbook", "acme", "Checkout runbook", "Checkout latency: database pool saturation and cache misses.",
                 allowed_groups=frozenset({"engineering"})),
        Document("plan", "acme", "Acquisition plan", "Checkout expansion budget and board approval notes.",
                 allowed_groups=frozenset({"finance"})),
        Document("memo", "acme", "All hands", "Checkout launch date announced to all staff.",
                 allowed_groups=frozenset({EVERYONE})),
        Document("globex", "globex", "Globex checkout", "Checkout database credentials rotation for globex.",
                 allowed_groups=frozenset({EVERYONE})),
    ]  # fmt: skip


def _service(**kwargs) -> SecureRAGService:
    service = SecureRAGService(**kwargs)
    for doc in _docs():
        service.ingest(doc)
    return service


class Authority:
    """A directory of record that can change without the index being re-synced."""

    def __init__(self, docs, fail=False):
        self.docs = {d.id: d for d in docs}
        self.fail = fail
        self.groups_of = {}

    def document(self, document_id):
        if self.fail:
            raise ConnectionError("directory unavailable")
        return self.docs.get(document_id)

    def groups(self, tenant_id, user_id):
        return self.groups_of.get(user_id)


def test_acl_is_fail_closed():
    assert not authorized("acme", frozenset(), frozenset(), ENGINEER)  # no grant, no access
    assert authorized("acme", frozenset(), frozenset({EVERYONE}), ENGINEER)
    assert not authorized("globex", frozenset(), frozenset({EVERYONE}), ENGINEER)
    assert authorized("acme", frozenset({"alice"}), frozenset(), ENGINEER)


def test_unauthorized_text_is_never_ranked():
    seen = []

    class Spy(HybridRetriever):
        def search(self, query, chunks, top_k=5, statistics_from=None):
            seen.extend(c.document_id for c in chunks)
            return super().search(query, chunks, top_k, statistics_from)

    ids = {h.chunk.document_id for h in _service(retriever=Spy()).query("checkout budget database", ENGINEER).hits}
    assert ids == {"runbook", "memo"}
    assert set(seen) == {"runbook", "memo"}


def test_live_recheck_enforces_changes_the_index_has_not_seen():
    authority = Authority(_docs())
    service = _service(authority=authority)
    assert "plan" in {h.chunk.document_id for h in service.query("expansion budget", FINANCE).hits}
    authority.docs["plan"] = Document("plan", "acme", "Acquisition plan", "...", allowed_groups=frozenset({"board"}))
    del authority.docs["memo"]
    hits = {h.chunk.document_id for h in service.query("checkout expansion budget launch", FINANCE).hits}
    assert hits.isdisjoint({"plan", "memo"})
    assert service.store.audit_events()[-1].denied_document_ids == ("memo", "plan")
    authority.groups_of["alice"] = frozenset({"sales"})  # moved team; her session token still says engineering
    assert "runbook" not in {h.chunk.document_id for h in service.query("checkout latency", ENGINEER).hits}


def test_unreachable_authority_fails_closed():
    service = _service(authority=Authority(_docs(), fail=True))
    result = service.query("checkout latency", ENGINEER)
    assert not result.hits and result.answer == "No authorized evidence was found."


def test_refill_after_denials_keeps_the_context_full():
    grant = frozenset({"engineering"})
    docs = [Document(f"d{i}", "acme", f"Doc {i}", f"checkout latency note {i}", allowed_groups=grant) for i in range(8)]
    authority = Authority(docs)
    service = SecureRAGService(authority=authority)
    for doc in docs:
        service.ingest(doc)
    top = [h.chunk.document_id for h in service.query("checkout latency note", ENGINEER, top_k=4).hits]
    for doc_id in top[:2]:
        del authority.docs[doc_id]
    again = [h.chunk.document_id for h in service.query("checkout latency note", ENGINEER, top_k=4).hits]
    assert len(again) == 4 and not set(again) & set(top[:2])


def test_scores_do_not_depend_on_other_tenants():
    service = _service()
    before = [(h.chunk.id, h.score) for h in service.query("checkout database latency", ENGINEER).hits]
    for i in range(20):
        service.ingest(Document(f"g{i}", "globex", "Globex", "database database latency checkout",
                                allowed_groups=frozenset({EVERYONE})))  # fmt: skip
    after = [(h.chunk.id, h.score) for h in service.query("checkout database latency", ENGINEER).hits]
    assert before == after


def test_regression_suite_metrics():
    metrics = evaluate(
        _service(), [EvalCase("checkout latency cache database", ENGINEER, "runbook", frozenset({"plan", "globex"}))]
    )
    assert metrics["recall_at_k"] == 1.0 and metrics["authorization_leak_rate"] == 0.0


def test_benchmark_headline():
    result = json.loads((ROOT / "results" / "benchmark.json").read_text())
    in_sync = result["scenarios"]["in sync"]["designs"]
    stale = result["scenarios"]["after changes, before the next sync"]["designs"]
    assert in_sync["no filter"]["queries_with_a_leak_pct"] > 90
    assert in_sync["post-filter"]["broad_questions_answered_pct"] < 30
    assert in_sync["pre-filter on index ACLs"]["broad_questions_answered_pct"] == 100
    assert stale["pre-filter on index ACLs"]["queries_with_a_leak_pct"] > 50
    assert stale["pre-filter + live re-check"]["queries_with_a_leak_pct"] == 0
    assert all(d["cross_tenant_leak_pct"] == 0 for name, d in stale.items() if name != "no filter")


@pytest.mark.parametrize("path", ["/v1/documents", "/v1/query", "/v1/audit"])
def test_api_routes_exist(path):
    from secure_rag.api import create_app

    assert path in {route.path for route in create_app().routes}


def test_api_round_trip():
    from fastapi.testclient import TestClient

    from secure_rag.api import create_app

    client = TestClient(create_app())
    client.post("/v1/documents", json={"id": "r", "tenant_id": "acme", "title": "Runbook", "text": "checkout latency",
                                       "allowed_groups": ["engineering"]})  # fmt: skip
    body = {"query": "checkout latency", "user_id": "alice", "tenant_id": "acme", "groups": ["engineering"]}
    assert [c["document_id"] for c in client.post("/v1/query", json=body).json()["citations"]] == ["r"]
    body["groups"] = ["sales"]
    assert client.post("/v1/query", json=body).json()["citations"] == []
    assert len(client.get("/v1/audit").json()) == 2
