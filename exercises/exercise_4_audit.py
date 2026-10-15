"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Read `node_fetch_pr` below — it is the one fully-worked example. Pay attention
      to *why* each AuditEntry field has the value it does at that step.
    - For every other node, you decide what to log. Field reference:
      `common/schemas.py:AuditEntry`. Helper: `risk_level_for(confidence)`.
    - Implement the one-line body of `audit()`. Everything else (graph wiring,
      checkpointer setup, interrupt/resume loop) is already done for you.

Implemented: the audit() body, ONE AuditEntry per node for
analyze/route/commit/auto_approve/synthesize, TWO each for human_approval and
escalate (before + after the interrupt). On the escalate path, synthesize now
routes through human_approval so the reviewer confirms the refined review.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import time
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import checkpoint_serde, db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"

SYSTEM_PROMPT = """You are a senior software engineer reviewing a GitHub pull request.
Return a structured review:
- summary: one paragraph on what the PR does.
- risk_factors: concrete risks (security, data/schema migration, correctness, missing tests).
- comments: specific, actionable review comments (file, line from the diff hunk when known, severity).
- confidence: a calibrated probability (0.0-1.0) that your review is complete and correct,
  i.e. the PR can be handled without further human judgment.
- confidence_reasoning: why you picked that confidence value; name the rule you applied.

Confidence rules (apply the FIRST that matches, these are hard limits):
  1. Any security red flag -> confidence MUST be between 0.30 and 0.50.
     Red flags: weak/fast password hashing (MD5, SHA1, unsalted), secrets or tokens stored
     in plaintext, SQL built with string formatting/concatenation, hard-coded credentials
     or user ids, auth/session logic, plain-HTTP network calls carrying user data.
  2. A reasonable feature with an open design question a human must decide (schema/data
     migration, backwards compatibility, missing tests for new logic) -> 0.60 to 0.70.
  3. Trivial or mechanical change (typo, docs, formatting, dependency bump) with no open
     questions -> 0.85 to 0.95.

Escalation: if confidence < 0.58, populate escalation_questions with 2-4 specific,
context-rich questions for the human reviewer. Each question must reference the file and
function/section of the diff it is about and explain why the answer matters, e.g.
"auth.py `hash_password` uses MD5 - is this a placeholder, or should it move to bcrypt/argon2
before merge?". Otherwise leave escalation_questions empty."""

SYNTHESIZE_PROMPT = """You are a senior software engineer finalising a pull-request review.
You get the diff, your own initial structured analysis, and a human reviewer's answers to
the questions you escalated. Produce a refined structured review:
- Treat the reviewer's answers as authoritative context about intent and constraints.
- Keep every comment that is still valid, drop ones the answers resolve, and add comments for
  anything the answers reveal. Security problems in the code stay as blockers unless the
  reviewer explicitly says they are out of scope.
- confidence: how sure you now are that this review is complete and correct. With the
  questions answered it should normally be higher than before.
- confidence_reasoning: explain how the answers changed your assessment.
- escalation_questions: leave empty."""


def reviewer_id() -> str:
    """GitHub username of the human reviewer (falls back to the local OS user)."""
    return os.environ.get("GITHUB_USER") or getpass.getuser()


def elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


async def audit(state, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    `thread_id` and `pr_url` are taken from `state` so callers only build
    the entry itself.
    """
    await write_audit_event(thread_id=state["thread_id"], pr_url=state["pr_url"], entry=entry)


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n"
                f"Files: {', '.join(state['pr_files'])}\n\n"
                f"Diff:\n{state['pr_diff']}"
            )},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision="auto" if decision == "auto_approve" else (
            "escalate" if decision == "escalate" else "pending"),
        reason=f"Routed to {decision} (auto >= {AUTO_APPROVE_THRESHOLD:.0%}, "
               f"escalate < {ESCALATE_THRESHOLD:.0%})",
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    console.print("[cyan]→ human_approval[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    after_escalation = bool(state.get("escalation_answers"))

    # Note: on resume this node re-runs from the top, so this "pending" row is
    # written once when pausing and once more when resuming.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=("Final confirm of refined review" if after_escalation
                else "Waiting for reviewer: approve / reject / edit"),
        execution_time_ms=elapsed_ms(t0),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "after_escalation": after_escalation,
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "escalation_answers": state.get("escalation_answers") or {},
        "diff_preview": state["pr_diff"][:2000],
    })

    # t0 is reset by the re-run, so this measures the post-resume work only.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer_id(),
        decision=resp.get("choice") or "reject",
        reason=resp.get("feedback") or None,
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"human_choice": resp.get("choice"), "human_feedback": resp.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state) -> tuple[str, str, str | None]:
    """Post the review comment. Returns (final_action, body, comment_url)."""
    body = _render_comment_body(state)
    try:
        url = post_review_comment(state["pr_url"], body)
        console.print(f"  [green]✓[/green] posted comment to {url or state['pr_url']}")
        return "committed", body, url
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed", body, None


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    # Both HITL paths converge here through human_approval:
    #   1. route → human_approval → commit
    #   2. route → escalate → synthesize → human_approval → commit
    # Only post when the reviewer approved.
    a = state["analysis"]
    choice = state.get("human_choice")
    body, url = None, None
    if choice == "approve":
        action, body, url = _post(state)
        if state.get("escalation_answers") and action == "committed":
            action = "committed_after_escalation"
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={choice})")
        action = "edit_requested" if choice == "edit" else "rejected"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer_id(),
        decision=choice or "reject",
        reason=f"{action}: {url}" if url else action,
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"final_action": action, "posted_comment_body": body, "posted_comment_url": url}


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action, body, url = _post(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"auto_{action}: {url}" if url else f"auto_{action}",
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"final_action": f"auto_{action}", "posted_comment_body": body, "posted_comment_url": url}


async def node_escalate(state):
    console.print("[cyan]→ escalate[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    # Like human_approval, this row is written again when the node re-runs on resume.
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="escalate",
        reason=f"Asking reviewer {len(questions)} question(s): " + " | ".join(questions),
        execution_time_ms=elapsed_ms(t0),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer_id(),
        decision="escalate",
        reason=" | ".join(f"{q} → {ans}" for q, ans in answers.items()),
        execution_time_ms=elapsed_ms(t0),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    initial: PRAnalysis = state["analysis"]
    qa = "\n".join(f"Q: {q}\nA: {ans}" for q, ans in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": SYNTHESIZE_PROMPT},
            {"role": "user", "content": (
                f"Title: {state['pr_title']}\n\n"
                f"Diff:\n{state['pr_diff']}\n\n"
                f"Initial analysis (JSON):\n{initial.model_dump_json(indent=2)}\n\n"
                f"Reviewer Q&A:\n{qa}"
            )},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%} (was {initial.confidence:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        decision="pending",
        reason=refined.confidence_reasoning,
        execution_time_ms=elapsed_ms(t0),
    ))
    # human_approval runs next so the reviewer confirms the refined review.
    return {"analysis": refined}


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "human_approval")
    checkpointer.serde = checkpoint_serde()
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        title = "Refined review — final confirm" if payload.get("after_escalation") else "Approval request"
        console.print(Panel.fit(
            f"{payload['summary']}\n\n[dim]{payload['confidence_reasoning']}[/dim]",
            title=f"{title} · conf={payload['confidence']:.0%}", border_style="green",
        ))
        for c in payload.get("comments", []):
            console.print(f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    console.print(Panel.fit(
        payload["summary"] + "\n\n[bold]Risks:[/bold]\n" +
        "\n".join(f"• {r}" for r in payload.get("risk_factors", [])),
        title=f"Escalation · conf={payload['confidence']:.0%}", border_style="yellow",
    ))
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
