"""Streamlit UI for backend-corpus IT operations QA."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.support_platform import CorpusRepository, MultiAgentSupportPlatform, SupportRequest

st.set_page_config(page_title="IT Operations Copilot", page_icon=":mag:", layout="wide")
st.markdown("""<style>
[data-testid="stAppViewContainer"] { background: #f3f6f5; }
[data-testid="stAppViewContainer"] .main { color: #173331; }
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] div[data-testid="stMarkdownContainer"] { color: #173331; }
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] { color: #4c6561; }
[data-testid="stAppViewContainer"] .answer-panel { background: #ffffff; color: #173331; padding: 1rem 1.2rem; border: 1px solid #c8d8d4; border-left: 5px solid #167c70; border-radius: 8px; }
[data-testid="stAppViewContainer"] .answer-panel strong { color: #0b4f49; }
[data-testid="stSidebar"] { background: #102a2a; }
[data-testid="stSidebar"] * { color: #edf8f5; }
.hero { padding: 2rem; background: #0d3b3b; color: #f4fbf9; border-radius: 10px; margin-bottom: 1.2rem; }
.hero h1 { margin: 0; font-size: 2.35rem; }
.hero p { color: #b9d5d0; margin: .5rem 0 0; }
</style>""", unsafe_allow_html=True)

SAMPLE_QUESTIONS = [
    "My SafeWord token is not working. How do I activate or recover it?",
    "How do I raise a CMP request for an application issue?",
    "Who is the POC and approver for this application access?",
    "How do I install the approved application?",
    "I have an application error. How do I resolve it?",
    "How do I request production application access and who approves it?",
    "The overnight batch failed. Can it be rerun and who approves the rerun?",
    "A TLS certificate is expiring. How do I renew it?",
    "My VPN is not connecting. Which support team should handle it?",
    "How do I check the status of an existing CMP request?",
    "How do I request access for a new joiner?",
]


def corpus_qa_tab() -> None:
    database_path = st.text_input("Backend corpus database", value=os.getenv("SUPPORT_PLATFORM_CORPUS_DATABASE", ":memory:"))
    llm_enabled = st.toggle("Use OpenAI for final answer", value=False)
    os.environ["SUPPORT_PLATFORM_LLM_ENABLED"] = "true" if llm_enabled else "false"
    if llm_enabled and not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not configured. Local grounded answer mode will be used.")

    if "corpus_repository" not in st.session_state or st.session_state.get("corpus_database") != database_path:
        repository = CorpusRepository(database_path)
        repository.seed()
        st.session_state.corpus_repository = repository
        st.session_state.corpus_database = database_path
    repository = st.session_state.corpus_repository
    orchestrator = MultiAgentSupportPlatform(corpus_repository=repository)

    st.subheader("Ask the operations corpus")
    st.caption("The corpus is maintained in the backend. You do not need to upload a document.")
    selected = st.selectbox("Sample questions", ["Choose a sample..."] + SAMPLE_QUESTIONS)
    default_question = "" if selected == "Choose a sample..." else selected
    question = st.text_area("Your question", value=default_question, height=90)
    if st.button("Ask operations copilot", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                st.session_state.last_corpus_answer = orchestrator.answer_question(question)
                st.session_state.last_corpus_question = question
            except Exception as error:
                st.session_state.last_corpus_answer = None
                st.error(f"The support agent could not answer this question: {error}")

    routed_answer = st.session_state.get("last_corpus_answer")
    if routed_answer is not None:
        st.subheader("Answer")
        st.markdown(
            f'<div class="answer-panel">{routed_answer.answer.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Tool selected: {routed_answer.tool}")
        st.subheader("Backend evidence")
        records = routed_answer.corpus_records
        if records:
            for record in records:
                with st.container(border=True):
                    st.write(f"**{record.record_id} | {record.title}**")
                    st.caption(f"Category: {record.category} | Source: {record.source}")
                    st.write(record.answer)
                    left, right = st.columns(2)
                    with left:
                        st.write("**Procedure**")
                        for step in record.procedure:
                            st.write(f"- {step}")
                    with right:
                        st.write(f"**POC:** {record.poc}")
                        st.write(f"**Approval:** {record.approval}")
        elif routed_answer.document_chunks:
            for chunk in routed_answer.document_chunks:
                with st.container(border=True):
                    st.caption(f"{chunk.document_name} | chunk {chunk.chunk_index}")
                    st.write(chunk.content)
        else:
            st.info("No matching backend record was found.")


def support_workflow_tab() -> None:
    st.subheader("Support workflow test")
    customer_id = st.text_input("Customer ID", value="CUST-DEMO", key="support_customer")
    summary = st.text_input("Summary", value="Account is locked", key="support_summary")
    details = st.text_area("Details", value="The customer cannot login after too many failed attempts.", key="support_details")
    if st.button("Run support workflow", type="primary"):
        outcome = MultiAgentSupportPlatform().process_request(SupportRequest(
            customer_id, summary, details,
            {"account_id": "ACCT-DEMO", "channel": "web", "identity_verified": True, "intent_confirmed": True},
        ))
        status_methods = {
            "resolved": st.success,
            "needs_information": st.warning,
            "needs_confirmation": st.warning,
            "escalated": st.error,
        }
        status_methods.get(outcome.status, st.info)(
            f"Status: {outcome.status.replace('_', ' ').title()} | Intent: {outcome.intent}"
        )
        st.subheader("Recommended actions")
        if outcome.recommended_actions:
            for action in outcome.recommended_actions:
                st.write(f"- {action}")
        else:
            st.caption("No automated actions returned.")
        if outcome.knowledge_artifacts:
            st.subheader("Knowledge used")
            for artifact in outcome.knowledge_artifacts:
                with st.container(border=True):
                    st.write(f"**{artifact.title}**")
                    st.caption(artifact.source)
                    st.write(artifact.summary)
        if outcome.follow_up:
            st.subheader("Customer follow-up")
            st.info(outcome.follow_up.prompt)
            st.caption("Missing: " + ", ".join(outcome.follow_up.missing_fields))
        if outcome.escalation:
            st.subheader("Human handoff")
            st.error(f"Route to: {outcome.escalation.team}")
            st.write(outcome.escalation.reason)
            st.write("Expertise: " + ", ".join(outcome.escalation.expertise_required))


st.markdown("""<div class="hero"><h1>IT operations copilot</h1><p>Ask how to resolve issues, raise CMP requests, find POCs, obtain approvals, or install approved software.</p></div>""", unsafe_allow_html=True)
with st.sidebar:
    st.header("Backend status")
    st.success("Operational corpus connected")
    st.caption("SQLite metadata corpus with POC, approval, procedure, and source fields.")
    if os.getenv("OPENAI_API_KEY"):
        st.success("OpenAI key detected")
    else:
        st.info("Local grounded mode")

qa_tab, support_tab = st.tabs(["Operations QA", "Support workflow"])
with qa_tab:
    corpus_qa_tab()
with support_tab:
    support_workflow_tab()
