from secure_rag.evaluation import EvalCase, evaluate
from secure_rag.models import AccessContext, Document
from secure_rag.service import SecureRAGService


def _service() -> SecureRAGService:
    service = SecureRAGService()
    service.ingest(
        Document(
            id="runbook",
            tenant_id="acme",
            title="Checkout runbook",
            text="Checkout latency is often caused by database pool saturation and cache misses.",
            allowed_groups=frozenset({"engineering"}),
        )
    )
    service.ingest(
        Document(
            id="finance",
            tenant_id="acme",
            title="Acquisition plan",
            text="Confidential acquisition budget and board approval notes.",
            allowed_groups=frozenset({"finance"}),
        )
    )
    service.ingest(
        Document(
            id="other-tenant",
            tenant_id="globex",
            title="Globex secrets",
            text="Checkout database password is secret.",
        )
    )
    return service


def test_acl_is_applied_before_and_after_retrieval() -> None:
    service = _service()
    engineer = AccessContext("alice", "acme", frozenset({"engineering"}))
    result = service.query("checkout database latency", engineer)
    ids = {hit.chunk.document_id for hit in result.hits}
    assert "runbook" in ids
    assert "finance" not in ids
    assert "other-tenant" not in ids


def test_permission_revocation_fails_closed() -> None:
    service = _service()
    finance = AccessContext("bob", "acme", frozenset({"finance"}))
    assert service.query("acquisition budget", finance).hits
    service.ingest(
        Document(
            id="finance",
            tenant_id="acme",
            title="Acquisition plan",
            text="Confidential acquisition budget and board approval notes.",
            allowed_groups=frozenset({"board"}),
            version=2,
        )
    )
    assert not service.query("acquisition budget", finance).hits


def test_evaluation_reports_zero_authorization_leakage() -> None:
    service = _service()
    metrics = evaluate(
        service,
        [
            EvalCase(
                "checkout latency cache database",
                AccessContext("alice", "acme", frozenset({"engineering"})),
                "runbook",
                frozenset({"finance", "other-tenant"}),
            )
        ],
    )
    assert metrics["recall_at_k"] == 1.0
    assert metrics["authorization_leak_rate"] == 0.0

