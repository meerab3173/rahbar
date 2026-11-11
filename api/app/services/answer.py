"""Turns a question into an AskResponse: the eight-step query pipeline (section 3.2).

    1 validate      AskRequest (3-500 chars) did this before we were called
    2 cache check   semantic cache, skipped for follow-ups and old-version questions
    3 embed         same model as the indexer (the pin is checked by the retriever)
    4 retrieve      top-k from Chroma, superseded documents filtered out unless asked for
    5 augment       services/prompt.py: instructions, numbered evidence, question
    6 generate      the provider, schema-constrained, low temperature
    7 validate      parse + Pydantic, retry once, then a canned fallback; citations are
                    re-checked against what was actually retrieved
    8 store         the turn, and a cache entry if it is safe to cache

Every step that can fail ends in a 200 with answered_from_context=false and a reason,
never in a 500.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..embedding import Embedder
from ..logjson import request_id_var
from ..providers.llm.base import GenerationError, LLMProvider, usage_var
from ..retrieval.base import Retriever
from ..schemas import Answer, AskRequest, AskResponse, Passage, RetrievedRef
from ..telemetry import (
    ABSTENTIONS,
    CITATIONS_DROPPED,
    FALLBACKS,
    GENERATION_SECONDS,
    RETRIEVAL_SECONDS,
    TOKENS_IN,
    TOKENS_OUT,
    TOP_SCORE,
    record_cache,
)
from .cache import SemanticCache, question_digest
from .guards import echoes_injection, ground_citations, is_followup, wants_old_version

log = logging.getLogger("rahbar.answer")

NOT_COVERED = ("The documents I have do not cover this question, so I can't answer it "
               "reliably. Please check with the relevant university office.")
FALLBACK = ("Sorry, I couldn't produce a reliable, cited answer to that just now. "
            "Please try again, or rephrase the question.")
UNAVAILABLE = ("The document search is temporarily unavailable. Please try again in a "
               "minute.")


@dataclass
class Pipeline:
    settings: Settings
    retriever: Retriever
    provider: LLMProvider
    embedder: Embedder | None = None
    cache: SemanticCache | None = None


def _abstain(text: str) -> Answer:
    return Answer(reasoning="", answer=text, citations=[], answered_from_context=False,
                  confidence="low")


def _refs(passages: list[Passage]) -> list[RetrievedRef]:
    return [RetrievedRef(chunk_id=p.chunk_id, source=p.source, section=p.section,
                         page=p.page, score=p.score) for p in passages]


def answer_question(request: AskRequest, pipe: Pipeline) -> AskResponse:
    settings, question = pipe.settings, request.question.strip()
    followup = is_followup(question, request.history)
    include_old = wants_old_version(question)
    # A follow-up is retrieved with its previous question as context.
    search_text = f"{request.history[-1]} {question}" if request.history else question
    usage_var.set({"tokens_in": 0, "tokens_out": 0})
    outcome: dict[str, Any] = {"followup": followup, "include_superseded": include_old,
                               "cached": False, "fallback": None, "abstain_reason": None,
                               "question_sha256": question_digest(question)}

    # Steps 2-4: embed once, check the cache, retrieve.
    passages: list[Passage] = []
    vector: list[float] | None = None
    generation = "unknown"
    cacheable_query = False
    started = time.perf_counter()
    try:
        generation = pipe.retriever.generation()
        if pipe.embedder is not None:
            vector = pipe.embedder.embed(search_text)
        cacheable_query = pipe.cache is not None and vector is not None \
            and not followup and not include_old
        if cacheable_query and pipe.cache is not None and vector is not None:
            hit, similarity = pipe.cache.lookup(generation, vector, question)
            record_cache(hit is not None)
            if hit is not None:
                response = AskResponse(**{**hit, "request_id": request_id_var.get(),
                                          "cached": True})
                log.info("answered", extra={"cached": True, "cache_similarity": round(similarity, 4),
                                            "top_k": settings.top_k,
                                            "chunk_ids": [r.chunk_id for r in response.retrieved],
                                            "scores": [r.score for r in response.retrieved],
                                            "answered_from_context": True})
                return response
        passages = pipe.retriever.retrieve(search_text, k=settings.top_k, vector=vector,
                                           include_superseded=include_old)
    except Exception as exc:
        RETRIEVAL_SECONDS.observe(time.perf_counter() - started)
        FALLBACKS.labels(reason="retrieval_error").inc()
        log.error("retrieval failed", extra={"error": f"{type(exc).__name__}: {exc}"})
        outcome["fallback"] = "retrieval_error"
        return _respond(_abstain(UNAVAILABLE), [], pipe, outcome)
    RETRIEVAL_SECONDS.observe(time.perf_counter() - started)
    top = max((p.score for p in passages), default=0.0)
    TOP_SCORE.observe(top)

    # Relevance floor: nothing close enough means the corpus does not cover it.
    if not passages or top < settings.relevance_floor:
        outcome["abstain_reason"] = "below_relevance_floor"
        return _respond(_abstain(NOT_COVERED), passages, pipe, outcome)

    # Steps 5-7: augment, generate, validate (retry and fallback live in the provider).
    started = time.perf_counter()
    try:
        answer = pipe.provider.generate(question, passages)
    except GenerationError as exc:
        outcome["fallback"] = exc.reason
        answer = _abstain(FALLBACK)
    except Exception as exc:  # a provider bug must still not become a 500
        outcome["fallback"] = f"provider_bug:{type(exc).__name__}"
        answer = _abstain(FALLBACK)
    GENERATION_SECONDS.observe(time.perf_counter() - started)
    if outcome["fallback"]:
        FALLBACKS.labels(reason=str(outcome["fallback"]).split(":")[0]).inc()
        return _respond(answer, passages, pipe, outcome)

    if echoes_injection(answer, passages):
        outcome["fallback"] = "injection_blocked"
        FALLBACKS.labels(reason="injection_blocked").inc()
        return _respond(_abstain(FALLBACK), passages, pipe, outcome)

    if not answer.answered_from_context:
        outcome["abstain_reason"] = "model_declined"
        answer = answer.model_copy(update={"citations": []})
        return _respond(answer, passages, pipe, outcome)

    citations, dropped = ground_citations(answer, passages)
    outcome["citations_dropped"] = dropped
    if dropped:
        CITATIONS_DROPPED.inc(dropped)
    if not citations:
        # An answer with no citation to a retrieved chunk is not grounded: decline.
        outcome["abstain_reason"] = "no_valid_citation"
        return _respond(_abstain(FALLBACK), passages, pipe, outcome)
    answer = answer.model_copy(update={"citations": citations})

    response = _respond(answer, passages, pipe, outcome)
    # Step 8b: cache only a clean, grounded, standalone answer.
    if cacheable_query and dropped == 0 and pipe.cache is not None and vector is not None:
        try:
            pipe.cache.store(generation, question, vector,
                             response.model_dump(exclude={"request_id", "cached"}))
        except Exception as exc:
            log.warning("cache store failed", extra={"error": str(exc)})
    return response


def _respond(answer: Answer, passages: list[Passage], pipe: Pipeline,
             outcome: dict[str, Any]) -> AskResponse:
    if not answer.answered_from_context:
        reason = outcome.get("abstain_reason") or outcome.get("fallback") or "unknown"
        ABSTENTIONS.labels(reason=str(reason).split(":")[0]).inc()
    usage = usage_var.get() or {"tokens_in": 0, "tokens_out": 0}
    if usage["tokens_in"] or usage["tokens_out"]:
        TOKENS_IN.observe(usage["tokens_in"])
        TOKENS_OUT.observe(usage["tokens_out"])
    rid = request_id_var.get()
    log.info("answered", extra={
        "retriever": pipe.retriever.name, "provider": pipe.provider.name,
        "top_k": pipe.settings.top_k,
        "chunk_ids": [p.chunk_id for p in passages],
        "scores": [round(p.score, 4) for p in passages],
        "answered_from_context": answer.answered_from_context,
        "abstention": not answer.answered_from_context,
        **outcome, **usage,
    })
    if pipe.cache is not None:   # step 8a: the turn, without the question text (ADR 0004)
        try:
            pipe.cache.store_turn({
                "request_id": rid, "ts": int(time.time()),
                "question_sha256": outcome.get("question_sha256"),
                "answered_from_context": answer.answered_from_context,
                "chunk_ids": [p.chunk_id for p in passages],
                "fallback": outcome.get("fallback"),
            })
        except Exception:
            pass
    return AskResponse(
        request_id=rid,
        answer=answer.answer,
        citations=answer.citations,
        answered_from_context=answer.answered_from_context,
        confidence=answer.confidence,
        retrieved=_refs(passages),
        provider=pipe.provider.name,
        cached=False,
    )
