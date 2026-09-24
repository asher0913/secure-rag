"""Retrieval behind two authorization checks: a pre-filter on the index and a live re-check.

The index holds a copy of every document's ACL taken when it was last synced,
and the caller's groups come from a session token. Both can be stale. So the
index copy only decides what may be *ranked*; before anything enters the
context, each candidate is checked again against the authority (the document
system and directory of record). If the authority cannot answer, the candidate
is dropped: the service fails closed.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Protocol

from .models import AccessContext, AuditEvent, Chunk, Document, QueryResult, RetrievalHit, authorized
from .retrieval import HybridRetriever
from .store import InMemoryStore


class Authority(Protocol):
    def document(self, document_id: str) -> Document | None: ...

    def groups(self, tenant_id: str, user_id: str) -> frozenset[str] | None: ...


class StoreAuthority:
    """Default authority: the store's current documents, and the caller's groups as given."""

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def document(self, document_id: str) -> Document | None:
        return self.store.get_document(document_id)

    def groups(self, tenant_id: str, user_id: str) -> frozenset[str] | None:
        return None  # unknown: trust the session's groups


def chunk_document(document: Document, max_chars: int = 420, overlap: int = 60) -> list[Chunk]:
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")
    paragraphs = [part.strip() for part in document.text.split("\n") if part.strip()]
    windows: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > max_chars:
            windows.append(current)
            current = current[-overlap:] + "\n" + paragraph
        else:
            current = (current + "\n" + paragraph).strip()
    if current:
        windows.append(current)
    return [
        Chunk(
            id=f"{document.id}:{index}",
            document_id=document.id,
            tenant_id=document.tenant_id,
            title=document.title,
            text=text,
            allowed_users=document.allowed_users,
            allowed_groups=document.allowed_groups,
            version=document.version,
        )
        for index, text in enumerate(windows)
    ]


class SecureRAGService:
    def __init__(
        self,
        store: InMemoryStore | None = None,
        retriever: HybridRetriever | None = None,
        generator: Callable[[str, list[RetrievalHit]], str] | None = None,
        authority: Authority | None = None,
        overfetch: int = 3,
    ) -> None:
        self.store = store or InMemoryStore()
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or self._citation_first_answer
        self.authority = authority or StoreAuthority(self.store)
        self.overfetch = overfetch

    def ingest(self, document: Document) -> None:
        self.store.upsert_document(document, chunk_document(document))

    def live_check(self, chunk: Chunk, context: AccessContext) -> bool:
        try:
            latest = self.authority.document(chunk.document_id)
            live_groups = self.authority.groups(context.tenant_id, context.user_id)
        except Exception:
            return False  # the authority is unavailable: fail closed
        if latest is None:
            return False  # deleted at the source
        if live_groups is not None:  # the directory's current groups override the session token's
            context = AccessContext(context.user_id, context.tenant_id, live_groups)
        return authorized(latest.tenant_id, latest.allowed_users, latest.allowed_groups, context)

    def query(self, query: str, context: AccessContext, top_k: int = 4) -> QueryResult:
        trace_id = uuid.uuid4().hex
        # 1. Pre-filter on the index's ACL copy, so unauthorized text is never ranked.
        visible = [
            c for c in self.store.all_chunks() if authorized(c.tenant_id, c.allowed_users, c.allowed_groups, context)
        ]
        candidates = self.retriever.search(query, visible, top_k=top_k * self.overfetch)
        # 2. Live re-check before context assembly; refill from further down the candidate list.
        allowed: list[RetrievalHit] = []
        denied: set[str] = set()
        for hit in candidates:
            if self.live_check(hit.chunk, context):
                allowed.append(hit)
            else:
                denied.add(hit.chunk.document_id)
            if len(allowed) == top_k:
                break
        self.store.append_audit(
            AuditEvent(
                trace_id=trace_id,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                query=query,
                candidate_count=len(candidates),
                returned_document_ids=tuple(dict.fromkeys(hit.chunk.document_id for hit in allowed)),
                denied_document_ids=tuple(sorted(denied)),
            )
        )
        return QueryResult(answer=self.generator(query, allowed), hits=tuple(allowed), trace_id=trace_id)

    @staticmethod
    def _citation_first_answer(query: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "No authorized evidence was found."
        evidence = " ".join(f"[{hit.chunk.document_id}] {hit.chunk.text[:180]}" for hit in hits[:3])
        return f"Question: {query}\nAuthorized evidence: {evidence}"
