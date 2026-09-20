from __future__ import annotations

import uuid
from collections.abc import Callable

from .models import AccessContext, AuditEvent, Chunk, Document, QueryResult, RetrievalHit
from .retrieval import HybridRetriever
from .store import InMemoryStore


def _authorized(chunk: Chunk, context: AccessContext) -> bool:
    if chunk.tenant_id != context.tenant_id:
        return False
    return (
        not chunk.allowed_users
        and not chunk.allowed_groups
        or context.user_id in chunk.allowed_users
        or bool(context.groups & chunk.allowed_groups)
    )


def _chunk(document: Document, max_chars: int = 420, overlap: int = 60) -> list[Chunk]:
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
    ) -> None:
        self.store = store or InMemoryStore()
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or self._citation_first_answer

    def ingest(self, document: Document) -> None:
        self.store.upsert_document(document, _chunk(document))

    def query(self, query: str, context: AccessContext, top_k: int = 4) -> QueryResult:
        trace_id = uuid.uuid4().hex
        all_chunks = self.store.all_chunks()
        # Snapshot pre-filter keeps unauthorized text out of the retriever/model boundary.
        visible = [chunk for chunk in all_chunks if _authorized(chunk, context)]
        candidates = self.retriever.search(query, visible, top_k=max(top_k * 2, top_k))
        # Real-time post-filter fails closed if permissions changed after the snapshot.
        allowed: list[RetrievalHit] = []
        denied: set[str] = set()
        for hit in candidates:
            latest = self.store.get_document(hit.chunk.document_id)
            if latest is None:
                denied.add(hit.chunk.document_id)
                continue
            latest_chunk = Chunk(
                id=hit.chunk.id,
                document_id=latest.id,
                tenant_id=latest.tenant_id,
                title=latest.title,
                text=hit.chunk.text,
                allowed_users=latest.allowed_users,
                allowed_groups=latest.allowed_groups,
                version=latest.version,
            )
            if _authorized(latest_chunk, context):
                allowed.append(hit)
            else:
                denied.add(hit.chunk.document_id)
            if len(allowed) == top_k:
                break
        event = AuditEvent(
            trace_id=trace_id,
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            query=query,
            candidate_count=len(candidates),
            returned_document_ids=tuple(dict.fromkeys(hit.chunk.document_id for hit in allowed)),
            denied_document_ids=tuple(sorted(denied)),
        )
        self.store.append_audit(event)
        return QueryResult(
            answer=self.generator(query, allowed),
            hits=tuple(allowed),
            trace_id=trace_id,
        )

    @staticmethod
    def _citation_first_answer(query: str, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "No authorized evidence was found."
        evidence = " ".join(
            f"[{hit.chunk.document_id}] {hit.chunk.text[:180]}" for hit in hits[:3]
        )
        return f"Question: {query}\nAuthorized evidence: {evidence}"

