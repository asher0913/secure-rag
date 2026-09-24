from __future__ import annotations

from threading import RLock

from .models import AuditEvent, Chunk, Document


class InMemoryStore:
    """Thread-safe store used by the demo; replaceable with Postgres + a vector DB."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._audit: list[AuditEvent] = []
        self._lock = RLock()

    def upsert_document(self, document: Document, chunks: list[Chunk]) -> None:
        with self._lock:
            old = [key for key, value in self._chunks.items() if value.document_id == document.id]
            for key in old:
                del self._chunks[key]
            self._documents[document.id] = document
            self._chunks.update({chunk.id: chunk for chunk in chunks})

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._documents.pop(document_id, None)
            self._chunks = {key: value for key, value in self._chunks.items() if value.document_id != document_id}

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks.values())

    def get_document(self, document_id: str) -> Document | None:
        with self._lock:
            return self._documents.get(document_id)

    def append_audit(self, event: AuditEvent) -> None:
        with self._lock:
            self._audit.append(event)

    def audit_events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._audit)
