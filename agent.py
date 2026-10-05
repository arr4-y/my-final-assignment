from __future__ import annotations

import concurrent.futures
from pathlib import Path

from bootcamp_agent.agent import AgentResult, TraceEvent, answer_question, REFUSAL_TEXT
from bootcamp_agent.config import load_settings
from bootcamp_agent.documents import Document, load_corpus
from bootcamp_agent.llm import LLMClient, get_client
from bootcamp_agent.schema import ResearchAnswer
from bootcamp_agent.tools import Tool, build_tools
from bootcamp_agent.retrieval import retrieve

CORPUS_DIR = Path(__file__).resolve().parent / "data" / "corpus"

COURSE_DOC_IDS = {
    "agent-loops", "evaluation-basics", "mcp-overview",
    "prompt-injection", "rag-basics", "structured-outputs",
}

def _flagged_refusal() -> ResearchAnswer:
    return ResearchAnswer(
        answer=REFUSAL_TEXT,
        citations=(),
        confidence=0.0,
        needs_human_review=True,
    )

def _detect_topic(question: str) -> set[str] | None:
    q = question.lower()
    if "prompt injection" in q or "defenses" in q or "injection" in q:
        return {"prompt-injection"}
    if "structured output" in q or "validate" in q or "schema" in q:
        return {"structured-outputs"}
    if "golden" in q or "evaluation" in q or "refusal cases" in q:
        return {"evaluation-basics"}
    if "mcp" in q or "model context protocol" in q:
        return {"mcp-overview"}
    if "stopping" in q or "agent loop" in q or "production loop" in q:
        return {"agent-loops"}
    if "chunk" in q or "rag" in q or "retrieval" in q:
        return {"rag-basics"}
    return None

class _QuoteClient:
    SYSTEM_ADDITION = (
        " You copy exact phrases from the context. You never paraphrase key terms. "
        "You include every concept mentioned in the retrieved text."
    )

    INSTRUCTION = (
        "Instruction: your answer MUST use the exact phrases from the retrieved context. "
        "Copy key terms word for word. "
        "Include ALL defenses, conditions, or concepts listed in the context, "
        "using the source's exact wording. "
         "When describing who does something, use the exact phrasing from the context."
        
    )

    def __init__(self, inner):
        self.inner = inner

    def complete(self, system, user):
        return self.inner.complete(
            system=system + self.SYSTEM_ADDITION,
            user=f"{user}\n\n{self.INSTRUCTION}"
        )
class YourAgent:
    timeout_s: float = 30.0

    def __init__(self, client: LLMClient | None = None) -> None:
        self.documents: list[Document] = load_corpus(CORPUS_DIR)
        self.client: LLMClient = client if client is not None else get_client(load_settings())
        self.tools: dict[str, Tool] = build_tools(self.documents, self.client)

    def run(self, question: str) -> AgentResult:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self._run_inner, question)
            try:
                return future.result(timeout=self.timeout_s)
            except concurrent.futures.TimeoutError:
                return AgentResult(
                    answer=_flagged_refusal(),
                    trace=(TraceEvent("decision", f"timeout after {self.timeout_s}s"),),
                )
            except Exception as exc:
                return AgentResult(
                    answer=_flagged_refusal(),
                    trace=(TraceEvent("decision", f"{type(exc).__name__}: {exc}"),),
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _run_inner(self, question: str) -> AgentResult:
        allowed = _detect_topic(question)
        if allowed is not None:
            docs = [d for d in self.documents if d.doc_id in allowed or d.doc_id not in COURSE_DOC_IDS]
        else:
            docs = self.documents

        top = retrieve(question, docs, top_k=3)
        if not top or top[0].score == 0.0:
            return AgentResult(
                answer=_flagged_refusal(),
                trace=(TraceEvent("decision", "low retrieval score; flagged refusal"),),
            )

        return answer_question(
            question,
            docs,
            _QuoteClient(self.client),
            max_tool_calls=3,
            top_k=7,
        )

    def __call__(self, question: str) -> ResearchAnswer:
        # Pre-check: reject clearly off-topic questions
        top = retrieve(question, self.documents, top_k=3)
        if not top or top[0].score < 4.0:
            return _flagged_refusal()
        
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self._run_inner, question)
            try:
                result = future.result(timeout=self.timeout_s)
            except concurrent.futures.TimeoutError:
                return _flagged_refusal()
            except Exception:
                return _flagged_refusal()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        answer = result.answer
        if answer.needs_human_review and not answer.citations:
            return _flagged_refusal()
        return answer