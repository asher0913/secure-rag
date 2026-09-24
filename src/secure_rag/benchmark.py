"""Where should a RAG system enforce permissions? Five designs on a seeded two-tenant corpus.

The directory of record (users' groups, documents' ACLs, deletions) is the
authority. The retrieval index holds a copy of each document's ACL from its
last sync, and a user's groups come from a session token issued at login.
Two scenarios: everything in sync, and after a round of changes (people
moving teams, documents being restricted or deleted) that the index and the
tokens have not caught up with yet. Leakage is always judged against the
directory of record.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, replace

from .models import EVERYONE, AccessContext, Chunk, Document, authorized
from .retrieval import HybridRetriever
from .service import SecureRAGService, chunk_document
from .store import InMemoryStore

TENANTS = ("acme", "globex")
DEPARTMENTS = {
    "engineering": (
        ("incident postmortem", "architecture review", "on-call runbook", "capacity plan"),
        ("latency", "database", "deploy", "cache", "rollback", "service", "outage", "cluster"),
    ),
    "finance": (
        ("budget forecast", "quarterly revenue report", "vendor contract review", "cost audit"),
        ("budget", "revenue", "forecast", "invoice", "margin", "spend", "quarter", "cost"),
    ),
    "hr": (
        ("salary band review", "performance review summary", "hiring plan", "employee relations case"),
        ("salary", "compensation", "hiring", "review", "promotion", "headcount", "benefits", "leave"),
    ),
    "legal": (
        ("litigation memo", "contract dispute summary", "regulatory filing", "patent assessment"),
        ("contract", "liability", "counsel", "dispute", "filing", "clause", "regulator", "settlement"),
    ),
    "sales": (
        ("pipeline review", "customer renewal plan", "pricing proposal", "territory plan"),
        ("pipeline", "renewal", "pricing", "customer", "deal", "quota", "discount", "territory"),
    ),
}
ALL_STAFF = (
    ("company announcement", "office policy", "holiday schedule", "security training"),
    ("policy", "office", "announcement", "training", "holiday", "travel"),
)
CODENAMES = ("falcon", "orion", "kestrel", "atlas", "juniper", "nimbus", "quartz", "sierra")  # shared across teams
TOP_K = 4


@dataclass
class Directory:
    """The authority: current group memberships and current documents."""

    groups_of: dict[tuple[str, str], frozenset[str]]
    documents: dict[str, Document]

    def document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    def groups(self, tenant_id: str, user_id: str) -> frozenset[str] | None:
        return self.groups_of.get((tenant_id, user_id), frozenset())

    def context(self, tenant_id: str, user_id: str) -> AccessContext:
        return AccessContext(user_id, tenant_id, self.groups_of[(tenant_id, user_id)])

    def can_read(self, document_id: str, context: AccessContext) -> bool:
        doc = self.documents.get(document_id)
        live = self.context(context.tenant_id, context.user_id)
        return doc is not None and authorized(doc.tenant_id, doc.allowed_users, doc.allowed_groups, live)


@dataclass(frozen=True)
class Query:
    text: str
    tenant_id: str
    user_id: str
    kind: str  # specific | broad | probe
    relevant: tuple[str, ...]  # documents the user may read that answer it; empty for probes


def build_world(seed: int = 0, users_per_tenant: int = 36, docs_per_department: int = 40):
    rng = random.Random(seed)
    docs: dict[str, Document] = {}
    groups_of: dict[tuple[str, str], frozenset[str]] = {}
    for tenant in TENANTS:
        depts = list(DEPARTMENTS)
        members: dict[str, list[str]] = {d: [] for d in depts}
        for u in range(users_per_tenant):
            user = f"{tenant}-u{u:02d}"
            groups = {depts[u % len(depts)]}
            if rng.random() < 0.2:
                groups.add(rng.choice(depts))
            groups_of[(tenant, user)] = frozenset(groups)
            for g in groups:
                members[g].append(user)
        for dept, (types, vocab) in [*DEPARTMENTS.items(), (EVERYONE, ALL_STAFF)]:
            count = docs_per_department if dept != EVERYONE else docs_per_department // 2
            for i in range(count):
                doc_type, codename = rng.choice(types), rng.choice(CODENAMES)
                period = f"{rng.choice(['q1', 'q2', 'q3', 'q4'])} {rng.choice([2023, 2024, 2025])}"
                words = rng.sample(vocab, 4)
                text = (
                    f"{doc_type} for project {codename}, {period}.\n"
                    f"The {words[0]} for {codename} moved after the {words[1]} in {period}.\n"
                    f"Owner notes: {words[2]} and {words[3]} need follow-up; figure {rng.randint(10, 999)}."
                )
                users: frozenset[str] = frozenset()
                grant: frozenset[str] = frozenset({dept})
                if dept in ("hr", "legal") and rng.random() < 0.2 and members[dept]:
                    users, grant = frozenset(rng.sample(members[dept], 1)), frozenset()  # a named-person case file
                doc_id = f"{tenant}-{dept}-{i:02d}"
                docs[doc_id] = Document(doc_id, tenant, f"{doc_type} {codename}", text, users, grant)
    return Directory(groups_of, docs)


def apply_changes(directory: Directory, seed: int = 0) -> tuple[Directory, dict[str, int]]:
    """People move teams, some documents are restricted to named people, some are deleted."""
    rng = random.Random(f"changes:{seed}")
    groups_of = dict(directory.groups_of)
    docs = dict(directory.documents)
    moved = restricted = deleted = 0
    depts = list(DEPARTMENTS)
    for key, groups in sorted(groups_of.items()):
        if rng.random() < 0.2:
            old = rng.choice(sorted(groups))
            new = rng.choice([d for d in depts if d not in groups])
            groups_of[key] = frozenset((groups - {old}) | {new})
            moved += 1
    for doc_id, doc in sorted(docs.items()):
        roll = rng.random()
        if roll < 0.05:
            del docs[doc_id]
            deleted += 1
        elif roll < 0.15 and EVERYONE not in doc.allowed_groups and doc.allowed_groups:
            owners = sorted(u for (t, u), g in groups_of.items() if t == doc.tenant_id and g & doc.allowed_groups)
            docs[doc_id] = replace(doc, allowed_users=frozenset(owners[:1]), allowed_groups=frozenset(), version=2)
            restricted += 1
    return Directory(groups_of, docs), {"users_moved_team": moved, "documents_restricted": restricted,
                                         "documents_deleted": deleted}  # fmt: skip


def _query_for(doc: Document, rng: random.Random) -> str:
    """How people actually ask: half name the kind of document and the project, half just the project
    and a topic, which matches documents across every team and both tenants."""
    doc_type, codename = doc.title.rsplit(" ", 1)
    first_line, second_line = doc.text.split("\n")[:2]
    topic = second_line.split()[1]
    period = first_line.rstrip(".").split(", ")[-1]
    if rng.random() < 0.5:
        return f"{doc_type} {codename} {period}"
    return f"what happened with the {topic} on project {codename} in {period}"


def build_queries(
    live: Directory, before: Directory | None = None, seed: int = 0, specific: int = 4, broad: int = 2, probes: int = 3
) -> list[Query]:
    """Per user: specific questions about documents they may read, broad questions about a project
    ("latest on project falcon", answered by any readable document about it), and probes written from
    documents they may not read. With ``before``, probes prefer documents the user could read before
    the changes and cannot now.
    """
    rng = random.Random(f"queries:{seed}")
    out = []
    for (tenant, user), _ in sorted(live.groups_of.items()):
        ctx = live.context(tenant, user)
        readable = [d for d in sorted(live.documents) if live.can_read(d, ctx)]
        forbidden = [d for d in sorted(live.documents) if not live.can_read(d, ctx)]
        revoked = []
        if before is not None:
            old_ctx = before.context(tenant, user)
            revoked = [d for d in sorted(before.documents) if before.can_read(d, old_ctx) and not live.can_read(d, ctx)]
        for doc_id in rng.sample(readable, min(specific, len(readable))):
            out.append(Query(_query_for(live.documents[doc_id], rng), tenant, user, "specific", (doc_id,)))
        for codename in rng.sample(CODENAMES, broad):
            about = tuple(d for d in readable if live.documents[d].title.endswith(f" {codename}"))
            if about:
                out.append(Query(f"latest on project {codename}", tenant, user, "broad", about))
        pool = revoked if revoked else forbidden
        for doc_id in rng.sample(pool, min(probes, len(pool))):
            source = (before or live).documents[doc_id]
            out.append(Query(_query_for(source, rng), tenant, user, "probe", ()))
    return out


class Index:
    """The retrieval index: chunks with the ACL copied at the last sync."""

    def __init__(self, snapshot: Directory, retriever: HybridRetriever) -> None:
        self.store = InMemoryStore()
        for doc in snapshot.documents.values():
            self.store.upsert_document(doc, chunk_document(doc))
        self.chunks = self.store.all_chunks()
        self.retriever = retriever

    def visible(self, context: AccessContext) -> list[Chunk]:
        return [c for c in self.chunks if authorized(c.tenant_id, c.allowed_users, c.allowed_groups, context)]


def _run_design(design: str, index: Index, live: Directory, token: AccessContext, text: str) -> list[Chunk]:
    r = index.retriever
    if design == "no filter":
        return [h.chunk for h in r.search(text, index.chunks, TOP_K)]
    if design == "post-filter":
        hits = r.search(text, index.chunks, TOP_K)
        return [h.chunk for h in hits if authorized(h.chunk.tenant_id, h.chunk.allowed_users, h.chunk.allowed_groups,
                                                     token)]  # fmt: skip
    if design == "post-filter, 4x over-fetch":
        hits = r.search(text, index.chunks, TOP_K * 4)
        ok = [h.chunk for h in hits if authorized(h.chunk.tenant_id, h.chunk.allowed_users, h.chunk.allowed_groups,
                                                   token)]  # fmt: skip
        return ok[:TOP_K]
    if design == "pre-filter on index ACLs":
        return [h.chunk for h in r.search(text, index.visible(token), TOP_K)]
    if design == "pre-filter + live re-check":
        service = SecureRAGService(store=index.store, retriever=r, authority=live)
        return [h.chunk for h in service.query(text, token, TOP_K).hits]
    raise ValueError(design)


DESIGNS = (
    "no filter",
    "post-filter",
    "post-filter, 4x over-fetch",
    "pre-filter on index ACLs",
    "pre-filter + live re-check",
)


def _score(design: str, index: Index, live: Directory, tokens: Directory, queries: list[Query]) -> dict:
    recall, rr, broad_answered, empty, answerable = [], [], [], 0, 0
    leaks = cross = forbidden_chunks = 0
    for q in queries:
        token = tokens.context(q.tenant_id, q.user_id)
        chunks = _run_design(design, index, live, token, q.text)
        ids = [c.document_id for c in chunks]
        bad = [c for c in chunks if not live.can_read(c.document_id, token)]
        leaks += bool(bad)
        cross += any(c.tenant_id != q.tenant_id for c in bad)
        forbidden_chunks += len(bad)
        if q.kind == "specific":
            recall.append(q.relevant[0] in ids)
            rr.append(1 / (ids.index(q.relevant[0]) + 1) if q.relevant[0] in ids else 0.0)
        elif q.kind == "broad":
            broad_answered.append(any(d in q.relevant for d in ids))
        if q.kind != "probe":
            answerable += 1
            empty += not chunks
    return {
        "recall_at_4": round(statistics.fmean(recall), 3),
        "mrr": round(statistics.fmean(rr), 3),
        "broad_questions_answered_pct": round(100 * statistics.fmean(broad_answered), 1),
        "empty_context_pct": round(100 * empty / answerable, 1),
        "queries_with_a_leak_pct": round(100 * leaks / len(queries), 1),
        "cross_tenant_leak_pct": round(100 * cross / len(queries), 1),
        "forbidden_chunks_per_100_queries": round(100 * forbidden_chunks / len(queries), 1),
    }


def run(seed: int = 0) -> dict:
    retriever = HybridRetriever()
    before = build_world(seed)
    after, changes = apply_changes(before, seed)
    index = Index(before, retriever)  # synced before the changes
    scenarios = {
        "in sync": (before, before, build_queries(before, seed=seed)),
        # the directory changed; the index and the users' session tokens have not caught up
        "after changes, before the next sync": (after, before, build_queries(after, before, seed=seed)),
    }
    out = {
        "corpus": {
            "tenants": len(TENANTS),
            "users": len(before.groups_of),
            "documents": len(before.documents),
            "named_person_documents": sum(bool(d.allowed_users) for d in before.documents.values()),
        },
        "changes": changes,
        "scenarios": {},
    }
    for name, (live, token_source, queries) in scenarios.items():
        out["scenarios"][name] = {
            "queries": len(queries),
            "by_kind": {k: sum(q.kind == k for q in queries) for k in ("specific", "broad", "probe")},
            "designs": {d: _score(d, index, live, token_source, queries) for d in DESIGNS},
        }
    return out
