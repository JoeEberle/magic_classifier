
# app_echo_chatbot.py
# Streamlit AI Chatbot 

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
import domain_topic_classifier as dtc
import magic_classifier as mg 

# -----------------------------
# Page & global config
# -----------------------------
st.set_page_config(page_title="Echo Chatbot", layout="wide")

LOGO_PATH_DEFAULT = "/users/josep/assets/chatbot_logo.png"  # change to your logo path or URL
CHAT_LOG_DIR = Path("./chat_logs")

# -----------------------------
# Auth (simple, replace later)
# -----------------------------
def verify_user(user_id: str, password: str) -> bool:
    """
    Replace this with your real auth (e.g., parquet, DB, OAuth).
    For now: accept demo/demo OR any non-empty pair (for prototyping).
    """
    if user_id == "demo" and password == "demo":
        return True
    return bool(user_id and password)

def login_ui() -> bool:
    st.sidebar.markdown("### 🔐 Sign In")
    user_id = st.sidebar.text_input("User ID", key="login_user_id")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    col_a, col_b = st.sidebar.columns([1, 1])
    with col_a:
        login = st.button("Sign In", type="primary", use_container_width=True)
    with col_b:
        st.button("Clear", on_click=lambda: [_ for _ in (st.session_state.pop("login_user_id", None),
                                                         st.session_state.pop("login_password", None))],
                  use_container_width=True)

    if login:
        if verify_user(user_id, password):
            st.session_state["auth_user"] = user_id
            st.toast("Signed in ✅", icon="✅")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials.")
            return False
    return "auth_user" in st.session_state

def logout():
    st.session_state.pop("auth_user", None)
    st.session_state.pop("messages", None)
    st.toast("Logged out", icon="👋")
    st.rerun()

# -----------------------------
# Chat model (stub for now)
# -----------------------------
def process_question(question: str, *, user_id: Optional[str] = None, controls: Optional[dict] = None) -> str:
    processed_answer = question
    domain_class,domain_score,domain_confidence,domain_evidence = dtc.domain_classifier(question, True, True)
    processed_answer += "\n\n" + f" domain class:{domain_class} score:{domain_score} " 
    magic_class,magic_score,magic_confidence,magic_evidence = mg.magic_classifier(question, True, True)
    processed_answer += "\n\n" + f" magic class:{magic_class} score:{magic_score} "     
    return processed_answer

# -----------------------------
# Utilities
# -----------------------------
def ensure_state():
    if "messages" not in st.session_state:
        # each message: {"role": "user"|"assistant", "content": str, "feedback": None|"up"|"down"}
        st.session_state["messages"]: List[Dict] = []

def save_conversation() -> Path:
    CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    user = st.session_state.get("auth_user", "anonymous")
    out = CHAT_LOG_DIR / f"chat_{user}_{ts}.json"
    payload = {
        "user": user,
        "timestamp": ts,
        "messages": st.session_state["messages"],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out

def set_feedback(idx: int, direction: str):
    msgs = st.session_state["messages"]
    if 0 <= idx < len(msgs):
        msgs[idx]["feedback"] = "up" if direction == "up" else "down"

# -----------------------------
# Sidebar
# -----------------------------
def render_sidebar():
    # Logo
    st.sidebar.markdown("## 🤖")
    logo_path = st.sidebar.text_input("Logo path or URL", value=LOGO_PATH_DEFAULT)
    if logo_path.strip():
        try:
            st.sidebar.image(logo_path, use_container_width=True)
        except Exception:
            st.sidebar.info("Could not load logo. Check path/URL.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Controls")
    st.sidebar.caption("These are placeholders for future options.")
    _model = st.sidebar.selectbox("Model (placeholder)", ["echo-stub"], index=0)
    _temperature = st.sidebar.slider("Creativity (placeholder)", 0.0, 1.0, 0.2, 0.05)

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    st.sidebar.caption("Tips: Type a question below. The assistant will echo it back for now.")

# -----------------------------
# Chat UI
# -----------------------------
def render_chat_ui():
    st.title("💬 Echo Chatbot")
    st.caption("Starter framework: login, sidebar, chat history, feedback buttons, and an extensible `process_question()`.")

    ensure_state()

    # Render chat history
    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"])
            # Feedback only for assistant messages
            if msg["role"] == "assistant":
                c1, c2, c3 = st.columns([0.12, 0.12, 0.76])
                with c1:
                    st.button("👍", key=f"up_{i}",
                              type="primary" if msg.get("feedback") == "up" else "secondary",
                              on_click=set_feedback, args=(i, "up"))
                with c2:
                    st.button("👎", key=f"dn_{i}",
                              type="primary" if msg.get("feedback") == "down" else "secondary",
                              on_click=set_feedback, args=(i, "down"))
                with c3:
                    if st.button("💾 Save Chat", key=f"save_{i}"):
                        path = save_conversation()
                        st.success(f"Saved to {path}")

    # Input box at the bottom
    user_input = st.chat_input("Ask a question…")
    if user_input:
        # 1) store user message
        st.session_state["messages"].append({"role": "user", "content": user_input, "feedback": None})
        # 2) process (echo)
        answer = process_question(user_input, user_id=st.session_state.get("auth_user"), controls={})
        # 3) store assistant message
        st.session_state["messages"].append({"role": "assistant", "content": answer, "feedback": None})
        st.rerun()

# -----------------------------
# App entry
# -----------------------------
def main():
    # If not signed in, show login in the sidebar and a welcome panel
    signed_in = login_ui()
    if not signed_in:
        st.title("💬 Echo Chatbot")
        st.info("Please sign in using the sidebar to start chatting.")
        return

    # If signed in: sidebar + chat
    render_sidebar()
    render_chat_ui()

if __name__ == "__main__":
    main()
