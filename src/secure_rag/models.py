from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class AccessContext:
    user_id: str
    tenant_id: str
    groups: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Document:
    id: str
    tenant_id: str
    title: str
    text: str
    allowed_users: frozenset[str] = field(default_factory=frozenset)
    allowed_groups: frozenset[str] = field(default_factory=frozenset)
    version: int = 1


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    tenant_id: str
    title: str
    text: str
    allowed_users: frozenset[str]
    allowed_groups: frozenset[str]
    version: int


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    lexical_score: float
    semantic_score: float


@dataclass(frozen=True)
class QueryResult:
    answer: str
    hits: tuple[RetrievalHit, ...]
    trace_id: str


@dataclass(frozen=True)
class AuditEvent:
    trace_id: str
    user_id: str
    tenant_id: str
    query: str
    candidate_count: int
    returned_document_ids: tuple[str, ...]
    denied_document_ids: tuple[str, ...]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

