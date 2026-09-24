"""secure-rag demo | benchmark"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run
from .models import AccessContext, Document
from .service import SecureRAGService


def demo() -> None:
    service = SecureRAGService()
    service.ingest(
        Document(
            "checkout-runbook",
            "acme",
            "Checkout latency runbook",
            "If checkout P95 exceeds 800 ms, inspect database pool saturation and cache misses.",
            allowed_groups=frozenset({"engineering"}),
        )
    )
    service.ingest(
        Document(
            "finance-plan",
            "acme",
            "Acquisition plan",
            "The acquisition budget for checkout expansion is 42 million dollars.",
            allowed_groups=frozenset({"finance"}),
        )
    )
    engineer = AccessContext("alice", "acme", frozenset({"engineering"}))
    print(service.query("checkout budget and latency", engineer).answer)
    print(json.dumps(service.store.audit_events()[-1].__dict__, default=str, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="secure-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="two documents, one engineer, and the audit record")
    bench = sub.add_parser("benchmark", help="five enforcement designs, in sync and after ACL changes")
    bench.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "demo":
        demo()
        return
    result = run()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    for scenario, data in result["scenarios"].items():
        print(scenario)
        for design, m in data["designs"].items():
            leak, broad = m["queries_with_a_leak_pct"], m["broad_questions_answered_pct"]
            print(f"  {design:28s} leak {leak:5.1f}%  broad answered {broad:5.1f}%  recall@4 {m['recall_at_4']:.3f}")


if __name__ == "__main__":
    main()
