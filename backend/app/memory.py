import json
import re
from typing import Any

from backend.app.config import settings
from backend.app.db import database
from backend.app.ollama import OllamaError, ollama_client
from backend.app.schemas import ChatMessage

MEMORY_SYSTEM_PROMPT = """You are Vega's local memory classifier.
Review recent conversation turns and decide whether durable memory should be stored.
Return only valid JSON. Do not include markdown.

Memory should be saved only for stable, reusable information:
- user preferences
- stable user identity or environment facts
- durable project constraints or decisions
- recurring procedures
- corrections to assistant behavior
- possible reusable skill/workflow candidates

Do not save one-off questions, temporary wording, generic facts, jokes, or content that is not useful later.
"""

MEMORY_VERIFIER_PROMPT = """You are Vega's local memory verifier.
Review a candidate memory and the most similar existing memories.
Return only valid JSON. Do not include markdown.

Choose exactly one action:
- save_new
- merge_with_existing
- update_existing
- reject_duplicate
- reject_low_value
- mark_conflict

Reject low-value or duplicate memories. Prefer concise durable memory over transcript-like notes.
"""


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in classifier output.")
    return json.loads(text[start : end + 1])


def clamp_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def build_review_packet(conversation_id: str) -> str:
    recent_messages = database.recent_messages(conversation_id, limit=10)
    existing_memories = database.list_memories(limit=30)

    memory_preview = [
        {
            "id": memory["id"],
            "type": memory["type"],
            "content": memory["content"],
            "status": memory["status"],
        }
        for memory in existing_memories[:10]
    ]
    message_preview = [
        {
            "id": message["id"],
            "role": message["role"],
            "content": message["content"],
        }
        for message in recent_messages
        if message["role"] in {"user", "assistant"}
    ]

    return json.dumps(
        {
            "conversation_id": conversation_id,
            "recent_messages": message_preview,
            "similar_existing_memories": memory_preview,
            "response_schema": {
                "should_store": "boolean",
                "memory_type": "preference|identity|project|fact|procedure|correction|skill_candidate|none",
                "content": "concise durable memory text or empty string",
                "confidence": "0.0 to 1.0",
                "importance": "0.0 to 1.0",
                "rationale": "brief reason",
                "action": "save_new|merge_with_existing|update_existing|reject_duplicate|reject_low_value|mark_conflict|no_op",
                "related_memory_ids": ["optional existing memory ids"],
                "source_message_ids": ["message ids supporting this decision"],
            },
        },
        ensure_ascii=False,
    )


async def classify_memory_for_conversation(conversation_id: str, job_id: str, run_id: str) -> None:
    try:
        database.add_trace_event(
            run_id=run_id,
            step="memory.classifier",
            status="started",
            message="Reviewing latest conversation turn for durable memory.",
            metadata={"model": settings.vega_memory_model},
        )
        review_packet = build_review_packet(conversation_id)
        response = await ollama_client.chat(
            model=settings.vega_memory_model,
            temperature=0,
            think=False,
            messages=[
                ChatMessage(role="system", content=MEMORY_SYSTEM_PROMPT),
                ChatMessage(role="user", content=review_packet),
            ],
        )
        candidate = extract_json_object(response.content)
        should_store = bool(candidate.get("should_store"))
        action = str(candidate.get("action") or "no_op")
        content = str(candidate.get("content") or "").strip()

        if not should_store or not content or action.startswith("reject") or action == "no_op":
            database.add_trace_event(
                run_id=run_id,
                step="memory.classifier",
                status="completed",
                message="Classifier found no durable memory to store.",
                metadata={"action": action},
            )
            database.complete_memory_job(job_id, status="completed")
            return

        confidence = clamp_score(candidate.get("confidence"))
        importance = clamp_score(candidate.get("importance"))
        embedding_model = settings.vega_embedding_model
        candidate_embedding = None
        similar_memories = []
        if embedding_model:
            candidate_embedding = await ollama_client.embed(model=embedding_model, text=content)
            similar_memories = database.find_similar_memories(
                vector=candidate_embedding,
                embedding_model=embedding_model,
                limit=10,
            )
            if similar_memories and similar_memories[0]["similarity"] >= 0.92:
                database.add_trace_event(
                    run_id=run_id,
                    step="memory.dedupe",
                    status="completed",
                    message="Candidate rejected as a near-duplicate by embedding similarity.",
                    metadata={
                        "top_similarity": similar_memories[0]["similarity"],
                        "memory_id": similar_memories[0]["id"],
                    },
                )
                database.complete_memory_job(job_id, status="completed")
                return

        verifier_decision = await verify_memory_candidate(
            candidate=candidate,
            similar_memories=similar_memories,
            run_id=run_id,
        )
        verified_action = str(verifier_decision.get("action") or action)
        if verified_action.startswith("reject"):
            database.add_trace_event(
                run_id=run_id,
                step="memory.verifier",
                status="completed",
                message="Verifier rejected the memory candidate.",
                metadata=verifier_decision,
            )
            database.complete_memory_job(job_id, status="completed")
            return

        status = "active" if (confidence or 0) >= 0.78 and (importance or 0) >= 0.45 else "suggested"
        source_message_ids = [
            message_id
            for message_id in candidate.get("source_message_ids", [])
            if isinstance(message_id, str)
        ]

        memory = database.add_memory(
            memory_type=str(candidate.get("memory_type") or "fact"),
            content=content,
            status=status,
            confidence=confidence,
            importance=importance,
            rationale=str(candidate.get("rationale") or ""),
            source_conversation_id=conversation_id,
            source_message_ids=source_message_ids,
            metadata={
                "action": action,
                "verified_action": verified_action,
                "related_memory_ids": candidate.get("related_memory_ids", []),
                "classifier_model": settings.vega_memory_model,
                "verifier_model": settings.vega_memory_verifier_model,
                "embedding_model": embedding_model,
                "similar_memory_ids": [memory["id"] for memory in similar_memories],
                "verifier": verifier_decision,
            },
        )
        if candidate_embedding and embedding_model:
            database.add_memory_embedding(
                memory_id=memory["id"],
                embedding_model=embedding_model,
                vector=candidate_embedding,
            )
        database.add_trace_event(
            run_id=run_id,
            step="memory.write",
            status="completed",
            message="Stored memory candidate.",
            metadata={"memory_id": memory["id"], "status": status, "type": memory["type"]},
        )
        database.complete_memory_job(job_id, status="completed")
    except (OllamaError, ValueError, json.JSONDecodeError) as exc:
        database.add_trace_event(
            run_id=run_id,
            step="memory.error",
            status="failed",
            message="Memory classification failed.",
            metadata={"error": str(exc)},
        )
        database.complete_memory_job(job_id, status="failed", error=str(exc))


async def verify_memory_candidate(
    *,
    candidate: dict[str, Any],
    similar_memories: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    database.add_trace_event(
        run_id=run_id,
        step="memory.verifier",
        status="started",
        message="Verifying memory candidate against similar memories.",
        metadata={
            "model": settings.vega_memory_verifier_model,
            "similar_memory_count": len(similar_memories),
        },
    )
    packet = json.dumps(
        {
            "candidate": candidate,
            "similar_memories": [
                {
                    "id": memory["id"],
                    "type": memory["type"],
                    "content": memory["content"],
                    "status": memory["status"],
                    "similarity": memory.get("similarity"),
                }
                for memory in similar_memories[:10]
            ],
            "response_schema": {
                "action": "save_new|merge_with_existing|update_existing|reject_duplicate|reject_low_value|mark_conflict",
                "confidence": "0.0 to 1.0",
                "reason": "brief rationale",
                "target_memory_id": "existing memory id or null",
            },
        },
        ensure_ascii=False,
    )
    response = await ollama_client.chat(
        model=settings.vega_memory_verifier_model,
        temperature=0,
        think=False,
        messages=[
            ChatMessage(role="system", content=MEMORY_VERIFIER_PROMPT),
            ChatMessage(role="user", content=packet),
        ],
    )
    decision = extract_json_object(response.content)
    database.add_trace_event(
        run_id=run_id,
        step="memory.verifier",
        status="completed",
        message="Verifier returned a memory decision.",
        metadata=decision,
    )
    return decision
