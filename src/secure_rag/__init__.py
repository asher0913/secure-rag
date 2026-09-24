"""Permission-aware retrieval for multi-tenant RAG: pre-filter, live re-check, audit."""

from .models import EVERYONE, AccessContext, Document, authorized
from .service import SecureRAGService

__all__ = ["EVERYONE", "AccessContext", "Document", "SecureRAGService", "authorized"]
