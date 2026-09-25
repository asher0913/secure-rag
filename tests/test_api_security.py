"""The HTTP layer: who the caller is comes from a signed token, and groups from the directory."""

import pytest
from fastapi.testclient import TestClient

from secure_rag import SecureRAGService
from secure_rag.api import create_app
from secure_rag.auth import TokenError, TokenSigner
from secure_rag.service import StoreAuthority
from secure_rag.store import InMemoryStore

SECRET = "test-secret-at-least-16-bytes"
QUERY = {"query": "checkout latency budget database"}


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def world():
    clock = Clock()
    signer = TokenSigner(SECRET, clock=clock)
    store = InMemoryStore()
    memberships = {
        ("acme", "alice"): frozenset({"engineering"}),
        ("acme", "bob"): frozenset({"finance"}),
        ("globex", "gina"): frozenset({"engineering", "finance"}),
    }
    service = SecureRAGService(store=store, authority=StoreAuthority(store, memberships))
    client = TestClient(create_app(service, signer))

    def auth(user, tenant, *roles):
        return {"Authorization": "Bearer " + signer.issue(user, tenant, roles or ("reader",))}

    writer = auth("wendy", "acme", "writer")
    for doc_id, text, group in (
        ("runbook", "Checkout latency: database pool saturation.", "engineering"),
        ("plan", "Checkout budget for the acquisition.", "finance"),
    ):
        body = {"id": doc_id, "title": doc_id, "text": text, "allowed_groups": [group]}
        assert client.post("/v1/documents", json=body, headers=writer).status_code == 200
    return {"client": client, "auth": auth, "memberships": memberships, "clock": clock, "signer": signer}


def cited(response):
    assert response.status_code == 200, response.text
    return {c["document_id"] for c in response.json()["citations"]}


def test_requests_without_a_valid_token_are_rejected(world):
    client, auth = world["client"], world["auth"]
    assert client.post("/v1/query", json=QUERY).status_code == 401
    assert client.post("/v1/query", json=QUERY, headers={"Authorization": "Bearer nonsense"}).status_code == 401
    forged = TokenSigner("a-different-secret-16b").issue("alice", "acme")
    assert client.post("/v1/query", json=QUERY, headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    # Swap in another user's claims but keep alice's signature.
    alice = auth("alice", "acme")["Authorization"].split()[1]
    bob = auth("bob", "acme")["Authorization"].split()[1]
    spliced = bob.split(".")[0] + "." + alice.split(".")[1]
    assert client.post("/v1/query", json=QUERY, headers={"Authorization": f"Bearer {spliced}"}).status_code == 401


def test_expired_tokens_are_rejected(world):
    headers = world["auth"]("alice", "acme")
    world["clock"].now += 3601
    assert world["client"].post("/v1/query", json=QUERY, headers=headers).status_code == 401
    with pytest.raises(TokenError):
        world["signer"].verify(headers["Authorization"].split()[1])


def test_the_body_cannot_claim_an_identity(world):
    impersonation = {**QUERY, "user_id": "bob", "tenant_id": "acme", "groups": ["finance"]}
    response = world["client"].post("/v1/query", json=impersonation, headers=world["auth"]("alice", "acme"))
    assert response.status_code == 422


def test_groups_come_from_the_directory_not_the_token(world):
    client, auth = world["client"], world["auth"]
    assert cited(client.post("/v1/query", json=QUERY, headers=auth("alice", "acme"))) == {"runbook"}
    assert cited(client.post("/v1/query", json=QUERY, headers=auth("bob", "acme"))) == {"plan"}
    assert cited(client.post("/v1/query", json=QUERY, headers=auth("mallory", "acme"))) == set()  # not in the directory


def test_revocation_takes_effect_without_a_new_token(world):
    client, headers = world["client"], world["auth"]("alice", "acme")
    assert cited(client.post("/v1/query", json=QUERY, headers=headers)) == {"runbook"}
    world["memberships"][("acme", "alice")] = frozenset({"sales"})  # moved team; same token
    assert cited(client.post("/v1/query", json=QUERY, headers=headers)) == set()


def test_tenants_are_isolated(world):
    client, auth = world["client"], world["auth"]
    assert cited(client.post("/v1/query", json=QUERY, headers=auth("gina", "globex"))) == set()
    takeover = {"id": "runbook", "title": "x", "text": "globex text", "allowed_groups": ["engineering"]}
    assert client.post("/v1/documents", json=takeover, headers=auth("gary", "globex", "writer")).status_code == 403
    assert cited(client.post("/v1/query", json=QUERY, headers=auth("alice", "acme"))) == {"runbook"}


def test_roles_guard_writes_and_the_audit_log(world):
    client, auth = world["client"], world["auth"]
    doc = {"id": "new", "title": "t", "text": "checkout", "allowed_groups": ["engineering"]}
    assert client.post("/v1/documents", json=doc, headers=auth("alice", "acme")).status_code == 403
    assert client.get("/v1/audit", headers=auth("alice", "acme")).status_code == 403
    assert client.post("/v1/query", json=QUERY, headers=auth("wendy", "acme", "writer")).status_code == 403


def test_the_audit_log_is_scoped_to_the_callers_tenant(world):
    client, auth = world["client"], world["auth"]
    client.post("/v1/query", json=QUERY, headers=auth("alice", "acme"))
    client.post("/v1/query", json=QUERY, headers=auth("gina", "globex"))
    events = client.get("/v1/audit", headers=auth("ada", "acme", "auditor")).json()
    assert [e["user_id"] for e in events] == ["alice"]


def test_the_server_assigns_versions(world):
    client, writer = world["client"], world["auth"]("wendy", "acme", "writer")
    body = {
        "id": "runbook",
        "title": "runbook",
        "text": "Checkout latency, revised.",
        "allowed_groups": ["engineering"],
    }
    assert client.post("/v1/documents", json=body, headers=writer).json()["version"] == 2


def test_an_unavailable_directory_answers_nothing(world):
    class Down(StoreAuthority):
        def groups(self, tenant_id, user_id):
            raise ConnectionError("directory unavailable")

    store = InMemoryStore()
    service = SecureRAGService(store=store, authority=Down(store, {}))
    client = TestClient(create_app(service, world["signer"]))
    assert client.post("/v1/query", json=QUERY, headers=world["auth"]("alice", "acme")).status_code == 503


def test_the_app_refuses_to_start_without_a_secret(monkeypatch):
    monkeypatch.delenv("SECURE_RAG_TOKEN_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        create_app()
