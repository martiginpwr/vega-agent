import json
import re
from typing import Any

from backend.app.config import settings
from backend.app.db import database
from backend.app.ollama import OllamaError, ollama_client
from backend.app.schemas import ChatMessage

MEMORY_SYSTEM_PROMPT = """You are Vega's local memory classifier. Return exactly one JSON object.

Memory should be saved only for stable, reusable information:
- user preferences
- stable user identity or environment facts
- durable project constraints or decisions
- recurring procedures
- corrections to assistant behavior
- possible reusable skill/workflow candidates

Do not save one-off questions, temporary wording, generic facts, jokes, or content that is not useful later.
Do not copy instructions, schemas, examples, or placeholder text from the input.
Assistant messages are context only. They are not evidence for user preferences, user facts, or project decisions.
Only store a memory when user-authored source_messages explicitly support it.
The content field must be a complete, concise sentence that will be useful in a future conversation.
Use only message ids that appear in allowed_source_message_ids.

Output true case:
{"should_store":true,"memory_type":"preference","content":"The user prefers concise engineering answers with concrete verification steps.","confidence":0.95,"importance":0.8,"rationale":"Stable user preference explicitly stated.","action":"save_new","related_memory_ids":[],"source_message_ids":["use-a-real-id-from-allowed_source_message_ids"]}

Output false case:
{"should_store":false,"memory_type":"none","content":"","confidence":0.0,"importance":0.0,"rationale":"No durable reusable information.","action":"no_op","related_memory_ids":[],"source_message_ids":[]}
"""

MEMORY_VERIFIER_PROMPT = """You are Vega's local memory verifier. Return exactly one JSON object.

Choose exactly one action:
- save_new
- merge_with_existing
- update_existing
- reject_duplicate
- reject_low_value
- mark_conflict

Reject low-value or duplicate memories. Prefer concise durable memory over transcript-like notes.
Do not copy the candidate packet, instructions, schemas, or examples.
If there are no similar memories and the candidate is a clear durable user preference, choose save_new.

Output shape:
{"action":"save_new","confidence":0.95,"reason":"Clear durable memory and no duplicate was found.","target_memory_id":null}
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
    message_preview = [
        {
            "id": message["id"],
            "role": message["role"],
            "content": message["content"],
        }
        for message in recent_messages
        if message["role"] == "user"
    ]
    assistant_context = [
        {
            "id": message["id"],
            "role": message["role"],
            "content": message["content"],
        }
        for message in recent_messages
        if message["role"] == "assistant"
    ]

    return json.dumps(
        {
            "conversation_id": conversation_id,
            "source_messages": message_preview,
            "assistant_context_not_evidence": assistant_context,
            "allowed_source_message_ids": [message["id"] for message in message_preview],
        },
        ensure_ascii=False,
    )


def valid_source_message_ids(conversation_id: str, ids: Any) -> list[str]:
    if not isinstance(ids, list):
        return []
    valid_ids = {
        message["id"]
        for message in database.recent_messages(conversation_id, limit=10)
        if message["role"] == "user"
    }
    return [message_id for message_id in ids if isinstance(message_id, str) and message_id in valid_ids]


STOPWORDS = {
    "about",
    "assistant",
    "future",
    "memory",
    "prefer",
    "prefers",
    "preference",
    "question",
    "questions",
    "response",
    "responses",
    "their",
    "there",
    "user",
    "vega",
    "with",
}


def content_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", value.lower())
        if len(token) >= 4 and token not in STOPWORDS
    }


def validate_candidate_grounding(
    *,
    conversation_id: str,
    candidate: dict[str, Any],
    content: str,
) -> tuple[bool, str, list[str]]:
    source_message_ids = valid_source_message_ids(conversation_id, candidate.get("source_message_ids", []))
    messages = {
        message["id"]: message
        for message in database.recent_messages(conversation_id, limit=10)
        if message["role"] == "user"
    }
    if not source_message_ids:
        source_message_ids = list(messages.keys())
        source_id_reason = "Candidate omitted valid source ids; grounding was checked against recent user messages."
    else:
        source_id_reason = "Candidate is grounded in cited user-authored source messages."

    if not source_message_ids:
        return False, "No user-authored source messages were available.", []

    evidence = " ".join(messages[message_id]["content"] for message_id in source_message_ids if message_id in messages)
    evidence_tokens = content_tokens(evidence)
    memory_tokens = content_tokens(content)
    if not memory_tokens:
        return False, "Candidate content had no meaningful grounded terms.", source_message_ids

    overlap = memory_tokens & evidence_tokens
    support_ratio = len(overlap) / len(memory_tokens)
    if not overlap or support_ratio < 0.4:
        return (
            False,
            "Candidate content was not grounded in the cited user message text.",
            source_message_ids,
        )
    return True, source_id_reason, source_message_ids


def looks_like_prompt_echo(value: str) -> bool:
    lowered = value.lower()
    echo_markers = [
        "concise durable memory text",
        "optional existing memory ids",
        "message ids supporting this decision",
        "response_schema",
        "similar_memories",
        "reply exactly",
    ]
    return any(marker in lowered for marker in echo_markers)


def fallback_verifier_decision(candidate: dict[str, Any], similar_memories: list[dict[str, Any]]) -> dict[str, Any]:
    content = str(candidate.get("content") or "").strip()
    confidence = clamp_score(candidate.get("confidence")) or 0.0
    importance = clamp_score(candidate.get("importance")) or 0.0
    if not content or looks_like_prompt_echo(content):
        return {
            "action": "reject_low_value",
            "confidence": 0.0,
            "reason": "Candidate content looked like prompt echo or was empty.",
            "target_memory_id": None,
        }
    if similar_memories and (similar_memories[0].get("similarity") or 0) >= 0.88:
        return {
            "action": "reject_duplicate",
            "confidence": 0.9,
            "reason": "Candidate is too similar to an existing memory.",
            "target_memory_id": similar_memories[0]["id"],
        }
    if bool(candidate.get("should_store")) and confidence >= 0.7 and importance >= 0.4:
        return {
            "action": "save_new",
            "confidence": confidence,
            "reason": "Classifier produced a high-confidence durable candidate and no duplicate blocked it.",
            "target_memory_id": None,
        }
    return {
        "action": "reject_low_value",
        "confidence": confidence,
        "reason": "Candidate did not meet local verifier thresholds.",
        "target_memory_id": None,
    }


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
            response_format="json",
            messages=[
                ChatMessage(role="system", content=MEMORY_SYSTEM_PROMPT),
                ChatMessage(role="user", content=review_packet),
            ],
        )
        candidate = extract_json_object(response.content)
        should_store = bool(candidate.get("should_store"))
        action = str(candidate.get("action") or "no_op")
        content = str(candidate.get("content") or "").strip()
        database.add_trace_event(
            run_id=run_id,
            step="memory.classifier",
            status="completed",
            message="Classifier returned a memory candidate decision.",
            metadata={
                "should_store": should_store,
                "action": action,
                "memory_type": candidate.get("memory_type"),
                "content": content,
                "confidence": candidate.get("confidence"),
                "importance": candidate.get("importance"),
                "source_message_ids": candidate.get("source_message_ids", []),
            },
        )

        if looks_like_prompt_echo(content):
            database.add_trace_event(
                run_id=run_id,
                step="memory.classifier",
                status="completed",
                message="Classifier output looked like prompt or schema echo, so no memory was stored.",
                metadata={"content": content},
            )
            database.complete_memory_job(job_id, status="completed")
            return

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

        grounded, grounding_reason, source_message_ids = validate_candidate_grounding(
            conversation_id=conversation_id,
            candidate=candidate,
            content=content,
        )
        if not grounded:
            database.add_trace_event(
                run_id=run_id,
                step="memory.grounding",
                status="completed",
                message="Candidate rejected because it was not grounded in user-authored source messages.",
                metadata={
                    "reason": grounding_reason,
                    "candidate_content": content,
                    "source_message_ids": source_message_ids,
                },
            )
            database.complete_memory_job(job_id, status="completed")
            return
        database.add_trace_event(
            run_id=run_id,
            step="memory.grounding",
            status="completed",
            message="Candidate passed user-source grounding checks.",
            metadata={
                "reason": grounding_reason,
                "source_message_ids": source_message_ids,
            },
        )

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
        if "candidate" in verifier_decision or "similar_memories" in verifier_decision:
            verifier_decision = fallback_verifier_decision(candidate, similar_memories)
            database.add_trace_event(
                run_id=run_id,
                step="memory.verifier",
                status="completed",
                message="Verifier output looked like prompt echo, so local verifier fallback made the decision.",
                metadata=verifier_decision,
            )
            verified_action = str(verifier_decision.get("action") or "reject_low_value")

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
    except (OllamaError, ValueError, json.JSONDecodeError, Exception) as exc:
        database.add_trace_event(
            run_id=run_id,
            step="memory.error",
            status="failed",
            message="Memory classification failed.",
            metadata={"error": str(exc), "error_type": type(exc).__name__},
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
        },
        ensure_ascii=False,
    )
    response = await ollama_client.chat(
        model=settings.vega_memory_verifier_model,
        temperature=0,
        think=False,
        response_format="json",
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
