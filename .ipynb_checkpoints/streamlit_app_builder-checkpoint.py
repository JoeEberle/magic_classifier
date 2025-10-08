from pathlib import Path

def generate_data_editor_app_sidebar(app_name: str = "app_sidebar.py", app_path: str = "/users/josep/authentication/"):
    app_code = r'''
import os
import sys
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import domain_topic_classifier as dtc
import magic_classifier as mg 

# ---------- Page setup ----------
st.set_page_config(page_title="Data Editor (Sidebar Controls)", layout="wide")

# Title stays above the editor; all controls go to the sidebar.
st.title("📝 DataFrame Editor v1.2")

ALLOWED_EXTS = {".xlsx", ".xls", ".csv", ".parquet", ".pq", ".json", ".feather", ".pickle"} 

# ---------- Helpers ----------
def load_df(p: Path) -> pd.DataFrame:
    ext = p.suffix.lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if ext in {".csv", ".txt"}:
        return pd.read_csv(p)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if ext == ".json":
        # Try JSON Lines first, then array
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            return pd.read_json(p)
    raise ValueError(f"Unsupported file type: {ext}")

def save_df(df: pd.DataFrame, p: Path, fmt: str, sqlite_table: str | None = None):
    if fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "parquet":
        df.to_parquet(p, index=False)
    elif fmt == "excel":
        df.to_excel(p, index=False)
    elif fmt == "feather":
        df.to_feather(p, index=False)   
    elif fmt == "pickle":
        df.to_pickle(p, index=False)                
    elif fmt == "json":
        df.to_json(p, orient="records", indent=2, date_format="iso", force_ascii=False)
    elif fmt == "sqlite":
        if not sqlite_table:
            raise ValueError("Please provide a SQLite table name.")
        with sqlite3.connect(p) as conn:
            df.to_sql(sqlite_table, conn, if_exists="replace", index=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")

@st.cache_data(show_spinner=False)
def scan_files(root_dir: str) -> list[str]:
    """Return sorted list of relative file paths under root that match ALLOWED_EXTS."""
    root = Path(root_dir).expanduser().resolve()
    results: list[str] = []
    if not root.exists():
        return results
    for ext in ALLOWED_EXTS:
        for p in root.rglob(f"*{ext}"):
            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                rel = str(p)
            results.append(rel)
    return sorted(set(results), key=lambda s: s.lower())

def infer_fmt_from_ext(ext: str) -> str | None:
    ext = ext.lower()
    if ext in {".csv", ".txt"}:
        return "csv"
    if ext in {".parquet", ".pq"}:
        return "parquet"
    if ext in {".xlsx", ".xls"}:
        return "excel"
    if ext in {".feather", ".ft"}:
        return "feather"        
    if ext in {".pickle", ".pkl"}:
        return "pickle"            
    if ext == ".json":
        return "json"
    return None

def ext_for(fmt_label: str) -> str:
    if fmt_label.startswith("CSV"): return ".csv"
    if fmt_label.startswith("Parquet"): return ".parquet"
    if fmt_label.startswith("Feather"): return ".feather"        
    if fmt_label.startswith("Pickle"): return ".pickle"    
    if fmt_label.startswith("Excel"): return ".xlsx"
    if fmt_label.startswith("JSON"): return ".json"
    if fmt_label.startswith("SQLite"): return ".db"
    return ""

# ===========================
#        SIDEBAR UI
# ===========================
with st.sidebar:
    # --- Logo area (leave space even if not used) ---
    st.markdown("### 🌟 App Branding")
    logo_path = st.text_input("Logo file path or URL (optional)", value="", help="PNG/JPG/GIF; leave blank to skip.")
    if logo_path.strip():
        try:
            st.image(logo_path, use_container_width=True)
        except Exception:
            st.info("Could not load logo. Check the path/URL.")

    st.markdown("---")
    st.markdown("### 🔎 File Browser")

    default_root = str(Path.cwd())
    root_dir = st.text_input("Search root folder", value=default_root, help="Scans this folder and subfolders.")
    rescan = st.button("🔄 Rescan files", key="rescan")
    name_filter = st.text_input("Filename filter (contains)", value="")

    if rescan:
        scan_files.clear()  # clear cache

    file_list = scan_files(root_dir)
    if name_filter.strip():
        q = name_filter.strip().lower()
        file_list = [f for f in file_list if q in f.lower()]

    if not file_list:
        st.info("No data files found.\nSupported: .xlsx, .csv, .parquet, .pq, .json")
        st.stop()

    selected_rel = st.selectbox("Select a file (relative to root)", options=file_list, index=0, key="file_select")
    selected_path = Path(root_dir).expanduser().resolve() / selected_rel
    st.caption(f"Selected: **{selected_path}**")

# ---------- Load selected file ----------
try:
    df = load_df(selected_path)
    st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
except Exception as e:
    st.error(f"Failed to read {selected_path}: {e}")
    st.stop()

# ===========================
#       MAIN: EDITOR
# ===========================
edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    key="editor"
)

# ===========================
#   SIDEBAR: SAVE OPTIONS
# ===========================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💾 Save Options")

    # --- Overwrite same file ---
    orig_ext = selected_path.suffix.lower()
    same_fmt = infer_fmt_from_ext(orig_ext)

    with st.expander("Overwrite the ORIGINAL file", expanded=False):
        if st.button(f"💾 Overwrite original ({orig_ext or 'unknown'})", key="overwrite_btn"):
            try:
                if not same_fmt:
                    st.error(f"Unsupported original format: {orig_ext}")
                else:
                    save_df(edited_df, selected_path, same_fmt)
                    st.toast(f"Saved to {selected_path}", icon="✅")
                    st.success(f"Overwrote {selected_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    # --- Save As ---
    with st.expander("Save As… (choose folder/name/format)", expanded=True):
        default_dir = str(selected_path.parent)
        default_name = selected_path.stem

        dest_dir = st.text_input("Destination folder", value=default_dir, key="dest_dir")
        base_name = st.text_input("Filename (without extension)", value=default_name, key="base_name")

        fmt_label = st.selectbox(
            "Format",
            ["CSV (.csv)", "Parquet (.parquet)", "Excel (.xlsx)", "JSON (.json)", "Feather (.feather)",
            "Pickle (.pickle)", "SQLite (.db)"],
            index=1,
            key="fmt_label"
        )
        sqlite_table = st.text_input("SQLite table name (only for SQLite)", value="user_list", key="sqlite_tbl") \
                        if "SQLite" in fmt_label else None
        overwrite = st.checkbox("Overwrite if file exists", value=True, key="overwrite_chk")

        if st.button("💾 Save As", key="saveas_btn"):
            try:
                out_dir = Path(dest_dir).expanduser()
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{base_name}{ext_for(fmt_label)}"
                exists = out_path.exists()
                if exists and not overwrite and not fmt_label.startswith("SQLite"):
                    st.error(f"File exists: {out_path}. Uncheck 'Overwrite' or change the name.")
                else:
                    fmt = "sqlite" if "SQLite" in fmt_label else infer_fmt_from_ext(out_path.suffix)
                    if not fmt:
                        raise ValueError(f"Cannot infer format from extension: {out_path.suffix}")
                    save_df(edited_df, out_path, fmt, sqlite_table=sqlite_table)
                    st.toast(f"Saved to {out_path}", icon="✅")
                    st.success(f"Saved to {out_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    st.caption("Tip: Change the search root and click Rescan to browse other folders.")
'''
    out_path = Path(app_path).expanduser() / app_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(app_code, encoding="utf-8")
    print(f"Wrote sidebar Streamlit editor to {out_path}")
    return app_code


def generate_chatbot_app(app_name: str = "chatbot.py", app_path: str = "/users/josep/magic_classifier/"):
    app_code = r'''
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
'''
    out_path = Path(app_path).expanduser() / app_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(app_code, encoding="utf-8")
    print(f"Wrote sidebar Streamlit editor to {out_path}")
    return app_code



