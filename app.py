"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer,
                                  then the refined review for a final confirm
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from audit.replay import fetch_threads
from common.db import db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()

RISK_BADGE = {"low": "🟢 low", "med": "🟡 med", "high": "🔴 high"}


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_session(thread_id: str) -> dict | None:
    """Rebuild UI state for a thread from the checkpointer (page reload / sidebar click)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snap = await app.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        return None
    return {
        "pr_url": snap.values.get("pr_url", ""),
        "payload": snap.interrupts[0].value if snap.interrupts else None,
        "final": snap.values if not snap.next else None,
    }


def open_thread(thread_id: str) -> None:
    session = asyncio.run(load_session(thread_id))
    if session is None:
        st.session_state.thread_id = None
        st.query_params.clear()
        return
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = session["pr_url"]
    st.session_state.interrupt_payload = session["payload"]
    st.session_state.final = session["final"]
    st.query_params["thread"] = thread_id


def apply_result(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")

# Restore the session from ?thread=<id> after a browser reload.
url_thread = st.query_params.get("thread")
if url_thread and st.session_state.thread_id != url_thread:
    open_thread(url_thread)


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    if st.button("➕ New review", width="stretch"):
        for k in ("thread_id", "interrupt_payload", "final"):
            st.session_state[k] = None
        st.session_state.pr_url = ""
        st.query_params.clear()
        st.rerun()

    try:
        threads = asyncio.run(fetch_threads(limit=20))
    except Exception as e:  # noqa: BLE001 — sidebar must never break the page
        threads = []
        st.caption(f"Could not read audit_events: {e}")

    if not threads:
        st.caption("No sessions yet — run a review.")
    else:
        rows = [{
            "PR": "#" + t["pr_url"].rstrip("/").rsplit("/", 1)[-1],
            "risk": RISK_BADGE.get(t["worst_risk"], t["worst_risk"]),
            "last event": str(t["last_event"])[:19].replace("T", " "),
            "thread": t["thread_id"][:8],
        } for t in threads]
        picked = st.dataframe(
            rows, hide_index=True, width="stretch",
            on_select="rerun", selection_mode="single-row", key="threads_table",
        )
        sel = picked.selection.rows
        if sel:
            chosen = threads[sel[0]]["thread_id"]
            if chosen != st.session_state.thread_id:
                open_thread(chosen)
                st.rerun()
        st.caption("Click a row to reopen that session.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_comments(comments: list[dict]) -> None:
    for c in comments:
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")


def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket (or final confirm after escalation): review + 3 buttons."""
    conf = payload["confidence"]
    if payload.get("after_escalation"):
        st.subheader(f"Refined review — final confirm · confidence {conf:.0%}")
        with st.expander("Your answers to the escalation questions", expanded=False):
            for q, a in payload.get("escalation_answers", {}).items():
                st.markdown(f"**{q}**  \n{a or '_(no answer)_'}")
    else:
        st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    render_comments(payload.get("comments", []))

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional — added to the posted comment)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve & post", type="primary", width="stretch"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", width="stretch"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Request edit", width="stretch"):
        if not feedback.strip():
            st.warning("Describe the edit you want in the feedback box first.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"⚠️ Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risk factors**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        st.markdown("**The agent needs answers before it can finish this review:**")
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(q, key=f"esc_q_{i}", height=80)
        if st.form_submit_button("Submit answers", type="primary"):
            if not all(a.strip() for a in answers.values()):
                st.warning("Please answer every question.")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    url = final.get("posted_comment_url")
    if action.startswith("auto") and action != "auto_commit_failed":
        st.success(f"🤖 Auto-approved — confidence {analysis.confidence:.0%}. "
                   "The agent posted the review itself; nothing for you to do.")
    elif action.startswith("committed"):
        st.success(f"✓ {action} — review comment posted.")
    elif action.endswith("commit_failed"):
        st.error("Posting the comment failed — check GITHUB_TOKEN and the terminal log.")
    elif action == "edit_requested":
        st.info("Edit requested — nothing posted; your feedback is recorded in the audit trail.")
    elif action == "rejected":
        st.warning("Rejected — no comment posted.")
    else:
        st.info(f"final_action = {action}")

    if url:
        st.link_button("View comment on GitHub", url)
    if analysis is not None:
        st.markdown(f"**Final confidence:** {analysis.confidence:.0%}")
        st.markdown(analysis.summary)
    if final.get("posted_comment_body"):
        with st.expander("Posted comment"):
            st.markdown(final["posted_comment_body"])


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.query_params["thread"] = st.session_state.thread_id

    try:
        with st.spinner("Fetching PR + asking the LLM..."):
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
        apply_result(result)
    except Exception as e:  # noqa: BLE001
        st.error(f"Review failed: {e}")

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        try:
            with st.spinner("Resuming..."):
                result = asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                ))
        except Exception as e:  # noqa: BLE001
            st.error(f"Resume failed: {e}")
        else:
            apply_result(result)
            st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    render_final(st.session_state.final)

if st.session_state.thread_id:
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
