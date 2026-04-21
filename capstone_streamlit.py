"""
=============================================================
  Study Buddy — Physics  |  Streamlit UI
  Student : Anshika Rai  |  Roll No: 2305766  |  Batch: 2027_Agentic AI
=============================================================
"""

import uuid
import streamlit as st
from agent import app, DOCUMENTS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy — Physics",
    page_icon="⚛️",
    layout="wide",
)

# ── Cache expensive resources ─────────────────────────────────────────────────
@st.cache_resource
def get_app():
    return app

agent_app = get_app()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph_messages" not in st.session_state:
    st.session_state.graph_messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚛️ Study Buddy")
    st.markdown("**Physics AI Assistant**")
    st.markdown("---")
    st.markdown("**👩‍🎓 Student:** Anshika Rai")
    st.markdown("**🎓 Roll No:** 2305766")
    st.markdown("**📚 Batch:** 2027_Agentic AI")
    st.markdown("---")
    st.markdown("### 📖 Topics Covered")
    for doc in DOCUMENTS:
        st.markdown(f"- {doc['topic']}")
    st.markdown("---")
    st.markdown("### 💡 Try asking:")
    st.markdown("- *Explain Newton's Second Law*")
    st.markdown("- *What is Ohm's Law?*")
    st.markdown("- *Calculate 9.8 * 5*")
    st.markdown("- *What is the speed of sound?*")
    st.markdown("---")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph_messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("⚛️ Study Buddy — Physics")
st.markdown(
    "I'm your AI-powered Physics tutor. Ask me anything from your B.Tech Physics syllabus!"
)
st.markdown("---")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📚 Sources: {', '.join(msg['sources'])}")

# Chat input
if prompt := st.chat_input("Ask a Physics question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            from langchain.schema import HumanMessage as HM
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            from agent import CapstoneState
            initial_state: CapstoneState = {
                "question": prompt,
                "messages": st.session_state.graph_messages,
                "route": "",
                "retrieved": "",
                "sources": [],
                "tool_result": "",
                "answer": "",
                "faithfulness": 1.0,
                "eval_retries": 0,
                "user_name": "",
            }
            result = agent_app.invoke(initial_state, config=config)

        answer = result.get("answer", "Sorry, I couldn't generate an answer.")
        sources = result.get("sources", [])
        faithfulness = result.get("faithfulness", 1.0)
        route = result.get("route", "")

        st.markdown(answer)
        if sources:
            st.caption(f"📚 Sources: {', '.join(sources)}")

        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"🔀 Route: `{route}`")
        with col2:
            st.caption(f"✅ Faithfulness: `{faithfulness:.2f}`")

    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.graph_messages = result.get("messages", [])
