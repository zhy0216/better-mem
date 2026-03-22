import asyncio
import json
import sys
import uuid
from datetime import datetime

import click
import structlog

from src.config import settings
from src.models.message import Message
from src.services.embedding import get_model
from src.store.cache import close_redis, create_redis
from src.store.database import close_pool, create_pool

logger = structlog.get_logger(__name__)


async def _startup():
    await create_pool()
    await create_redis()
    get_model()


async def _shutdown():
    await close_pool()
    await close_redis()


def run_async(coro):
    """Run an async coroutine with full lifecycle management."""

    async def _wrapped():
        await _startup()
        try:
            return await coro
        finally:
            await _shutdown()

    return asyncio.run(_wrapped())


@click.group()
def cli():
    """Better-Mem: proposition-centric long-term memory for conversational AI."""
    pass


# ---------------------------------------------------------------------------
# memorize
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--user-id", default="unknown", help="User ID for the messages.")
@click.option("--group-id", required=True, help="Conversation / group ID.")
@click.option("--tenant-id", default="default", help="Tenant ID.")
@click.option("--file", "input_file", type=click.Path(exists=True), default=None,
              help="JSON file with messages array. Reads stdin if omitted.")
def memorize(user_id, group_id, tenant_id, input_file):
    """Memorize messages: extract propositions, update beliefs, build profile.

    Input JSON format (array of messages):

    \b
    [
      {"role": "user", "content": "I love hiking"},
      {"role": "assistant", "content": "That's great!"}
    ]
    """
    if input_file:
        with open(input_file) as f:
            raw = json.load(f)
    else:
        raw = json.load(sys.stdin)

    if isinstance(raw, dict) and "messages" in raw:
        raw = raw["messages"]

    messages = [Message(**m) for m in raw]
    if not messages:
        click.echo("No messages provided.", err=True)
        raise SystemExit(1)

    async def _run():
        from src.services import memorize_service
        from src.store import buffer_store

        batch_id = uuid.uuid4()
        buffers = await buffer_store.insert_messages(
            messages=messages,
            group_id=group_id,
            tenant_id=tenant_id,
            batch_id=batch_id,
            user_id=user_id,
        )
        await buffer_store.mark_processing([b.id for b in buffers])

        try:
            saved = await memorize_service.process_buffered_messages(
                buffers=buffers,
                tenant_id=tenant_id,
                group_id=group_id,
                user_id=user_id,
            )
            await buffer_store.mark_consumed([b.id for b in buffers])
        except Exception:
            await buffer_store.mark_pending([b.id for b in buffers])
            raise

        result = [
            {
                "id": str(p.id),
                "canonical_text": p.canonical_text,
                "proposition_type": p.proposition_type,
                "semantic_key": p.semantic_key,
                "tags": p.tags,
            }
            for p in saved
        ]
        return result

    saved = run_async(_run())
    click.echo(json.dumps({"status": "completed", "propositions": saved}, indent=2, default=str))


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--user-id", required=True, help="User ID to recall memories for.")
@click.option("--tenant-id", default="default", help="Tenant ID.")
@click.option("--top-k", default=20, type=int, help="Max candidates to retrieve.")
@click.option("--assemble/--no-assemble", default=True, help="Assemble context via LLM.")
@click.option("--include-profile/--no-profile", default=True, help="Include user profile.")
def recall(query, user_id, tenant_id, top_k, assemble, include_profile):
    """Recall memories relevant to QUERY for a given user."""

    async def _run():
        from src.retrieve import assembler, searcher
        from src.store import profile_store, proposition_store

        from src.models.proposition import SearchFilters

        filters = SearchFilters()

        candidates, elapsed_ms = await searcher.hybrid_search(
            query=query,
            user_id=user_id,
            tenant_id=tenant_id,
            top_k=top_k,
            filters=filters,
        )

        hit_ids = [p.id for p in candidates]
        if hit_ids:
            try:
                await proposition_store.track_access(hit_ids)
            except Exception:
                pass

        profile = None
        if include_profile:
            profile = await profile_store.get(user_id, tenant_id)

        if assemble and candidates:
            assembled = await assembler.assemble_context(query, candidates, profile)
            selected_ids = set(assembled.selected_proposition_ids)
            filtered_props = [p for p in candidates if str(p.id) in selected_ids] or candidates

            profile_snippet = None
            if profile:
                relevant_traits = [
                    t.get("trait", "") for t in profile.profile_data.personality[:3]
                    if t.get("trait")
                ]
                profile_snippet = {
                    "summary": profile.profile_data.summary,
                    "relevant_traits": relevant_traits,
                }

            return {
                "context": assembled.context,
                "propositions": [
                    {
                        "id": str(p.id),
                        "canonical_text": p.canonical_text,
                        "confidence": round(p.confidence, 4),
                        "score": round(p.score, 4),
                        "source": p.source,
                    }
                    for p in filtered_props
                ],
                "profile_snippet": profile_snippet,
                "total_candidates": len(candidates),
                "search_time_ms": round(elapsed_ms, 1),
            }

        return {
            "propositions": [
                {
                    "id": str(p.id),
                    "canonical_text": p.canonical_text,
                    "proposition_type": p.proposition_type,
                    "confidence": round(p.confidence, 4),
                    "score": round(p.score, 4),
                    "source": p.source,
                }
                for p in candidates
            ],
            "total_candidates": len(candidates),
            "search_time_ms": round(elapsed_ms, 1),
        }

    result = run_async(_run())
    click.echo(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# propositions
# ---------------------------------------------------------------------------

@cli.group()
def propositions():
    """Manage propositions."""
    pass


@propositions.command("list")
@click.option("--user-id", required=True, help="User ID.")
@click.option("--tenant-id", default="default", help="Tenant ID.")
@click.option("--type", "proposition_type", default=None, help="Filter by proposition type.")
@click.option("--status", default="active", help="Filter by belief status.")
@click.option("--limit", default=50, type=int, help="Max results.")
@click.option("--offset", default=0, type=int, help="Offset for pagination.")
def list_propositions(user_id, tenant_id, proposition_type, status, limit, offset):
    """List propositions for a user."""

    async def _run():
        from src.store import proposition_store

        props = await proposition_store.list_propositions(
            user_id=user_id,
            tenant_id=tenant_id,
            proposition_type=proposition_type,
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "propositions": [
                {
                    "id": str(p.id),
                    "canonical_text": p.canonical_text,
                    "proposition_type": p.proposition_type,
                    "semantic_key": p.semantic_key,
                    "first_observed_at": p.first_observed_at.isoformat() if p.first_observed_at else None,
                    "last_observed_at": p.last_observed_at.isoformat() if p.last_observed_at else None,
                    "tags": p.tags,
                    "metadata": p.metadata,
                }
                for p in props
            ],
            "count": len(props),
        }

    result = run_async(_run())
    click.echo(json.dumps(result, indent=2, default=str))


@propositions.command("delete")
@click.argument("proposition_id")
@click.option("--tenant-id", default="default", help="Tenant ID.")
def delete_proposition(proposition_id, tenant_id):
    """Soft-delete a proposition by ID."""

    async def _run():
        from src.store import proposition_store

        ok = await proposition_store.soft_delete(uuid.UUID(proposition_id), tenant_id)
        if not ok:
            click.echo(f"Proposition {proposition_id} not found.", err=True)
            raise SystemExit(1)
        return {"status": "deleted", "id": proposition_id}

    result = run_async(_run())
    click.echo(json.dumps(result, indent=2))


@propositions.command("evidence")
@click.argument("proposition_id")
@click.option("--limit", default=10, type=int, help="Max evidence entries.")
def get_evidence(proposition_id, limit):
    """Show evidence for a proposition."""

    async def _run():
        from src.store import proposition_store

        evidences = await proposition_store.get_evidence(uuid.UUID(proposition_id), limit=limit)
        return {
            "evidence": [
                {
                    "id": str(e.id),
                    "evidence_type": e.evidence_type,
                    "direction": e.direction,
                    "source_type": e.source_type,
                    "speaker_id": e.speaker_id,
                    "quoted_text": e.quoted_text,
                    "observed_at": e.observed_at.isoformat() if e.observed_at else None,
                    "weight": e.weight,
                    "created_at": e.created_at.isoformat(),
                }
                for e in evidences
            ],
            "count": len(evidences),
        }

    result = run_async(_run())
    click.echo(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("user_id")
@click.option("--tenant-id", default="default", help="Tenant ID.")
@click.option("--scope", default="global", help="Profile scope.")
@click.option("--group-id", default=None, help="Group ID.")
def profile(user_id, tenant_id, scope, group_id):
    """Show the synthesized profile for a user."""

    async def _run():
        from src.store import profile_store

        p = await profile_store.get(user_id, tenant_id, scope, group_id)
        if not p:
            click.echo(f"Profile not found for user {user_id}.", err=True)
            raise SystemExit(1)
        return {
            "user_id": p.user_id,
            "scope": p.scope,
            "version": p.version,
            "fact_count": p.fact_count,
            "profile_data": p.profile_data.model_dump(),
            "updated_at": p.updated_at.isoformat(),
        }

    result = run_async(_run())
    click.echo(json.dumps(result, indent=2, default=str))


def main():
    cli()


if __name__ == "__main__":
    main()
