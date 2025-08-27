import os
import time
import json
import math
import uuid
import queue
import hashlib
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lightweight embeddings (no torch)
from fastembed import TextEmbedding
import faiss

# -------- Optional NLI (auto-fallback if not installed) ----------
HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

APP_TITLE = "NTT 22nd Century Enterprise AI Knowledge & Decision Platform (MVP)"
APP_TAGLINE = "Free, lightweight, open-source RAG • Tool-calling (MCP substitute) • MLOps-lite • Streamlit"

# ---------------------------
# Utility: Determinism / Seeds
# ---------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
set_seed(42)

# ---------------------------
# Governance / Login (OAuth-lite)
# ---------------------------
def login_gate():
    """
    Super-light login gate to simulate OAuth (no external providers, suitable for Streamlit free tier).
    Use STREAMLIT_PASS in Secrets or leave empty to disable.
    """
    required_pass = st.secrets.get("STREAMLIT_PASS", "") if hasattr(st, "secrets") else ""
    if not required_pass:
        return True  # no password configured

    st.sidebar.subheader("🔐 Login")
    pwd = st.sidebar.text_input("Enter access pass", type="password")
    if st.sidebar.button("Unlock"):
        if pwd == required_pass:
            st.session_state["auth_ok"] = True
        else:
            st.error("Incorrect pass")
    return st.session_state.get("auth_ok", False)

# ---------------------------
# Data Redaction (PII-lite)
# ---------------------------
def redact_text(text: str) -> str:
    # very light redaction for emails & phones
    text = str(text)
    text = pd.Series([text]).str.replace(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", regex=True).iloc[0]
    text = pd.Series([text]).str.replace(r"\b(?:\+?\d[\d -]{8,}\d)\b", "[REDACTED_PHONE]", regex=True).iloc[0]
    return text

# ---------------------------
# Mini Tool Router (MCP substitute)
# ---------------------------
def tool_fetch_earthquakes(start_date: str, end_date: str, min_mag: float = 4.5, limit: int = 200) -> Dict[str, Any]:
    """
    USGS Free Earthquake API (lightweight big-data source).
    """
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": min_mag,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def tool_fetch_covid(country: str = "all") -> Dict[str, Any]:
    """
    disease.sh Free COVID-19 API (lightweight big-data source).
    country="all" for global, or country name (e.g., "India").
    """
    base = "https://disease.sh/v3/covid-19"
    url = f"{base}/all" if country.lower() == "all" else f"{base}/countries/{country}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

TOOL_REGISTRY = {
    "earthquakes": tool_fetch_earthquakes,
    "covid": tool_fetch_covid,
}

def call_tool(tool_name: str, **kwargs):
    t0 = time.time()
    out = TOOL_REGISTRY[tool_name](**kwargs)
    latency = (time.time() - t0) * 1000
    trace = {
        "tool": tool_name,
        "kwargs": kwargs,
        "latency_ms": round(latency, 2),
        "ts": datetime.utcnow().isoformat() + "Z"
    }
    return out, trace

# ---------------------------
# RAG Components (Lightweight)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    # Default small model is MiniLM L6 v2 (ONNX) pulled by fastembed; lightweight & accurate for RAG.
    return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize for cosine-like similarity using inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))
    return index

def embed_texts(texts: List[str]) -> np.ndarray:
    emb = get_embedder()
    # fastembed returns generator
    vectors = list(emb.embed(texts))
    return np.array(vectors, dtype=np.float32)

def make_docs_from_earthquakes(geojson: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = []
    for f in geojson.get("features", [])[:1000]:
        props = f.get("properties", {})
        place = props.get("place", "")
        mag = props.get("mag", None)
        time_ms = props.get("time", 0)
        url = props.get("url", "")
        iso = datetime.utcfromtimestamp(time_ms/1000).isoformat() + "Z" if time_ms else ""
        txt = f"Magnitude {mag} earthquake at {place}. Time (UTC): {iso}. Source: {url}"
        docs.append({
            "id": f.get("id", str(uuid.uuid4())),
            "text": txt,
            "meta": {"mag": mag, "place": place, "time_utc": iso, "url": url, "source": "USGS"}
        })
    return docs

def make_docs_from_covid(json_obj: Dict[str, Any], country: str) -> List[Dict[str, Any]]:
    keys = ["cases","todayCases","deaths","todayDeaths","recovered","todayRecovered","active","critical","tests","population","updated"]
    vals = {k: json_obj.get(k, None) for k in keys}
    updated_iso = datetime.utcfromtimestamp(vals["updated"]/1000).isoformat()+"Z" if vals.get("updated") else ""
    summary = (
        f"COVID-19 status for {country.title() if country!='all' else 'Global'}: "
        f"cases={vals['cases']}, todayCases={vals['todayCases']}, deaths={vals['deaths']}, "
        f"todayDeaths={vals['todayDeaths']}, recovered={vals['recovered']}, todayRecovered={vals['todayRecovered']}, "
        f"active={vals['active']}, critical={vals['critical']}, tests={vals['tests']}, population={vals['population']}. "
        f"Updated (UTC): {updated_iso}."
    )
    return [{
        "id": str(uuid.uuid4()),
        "text": summary,
        "meta": {"source":"disease.sh","country": country, **vals, "updated_iso": updated_iso}
    }]

def corpus_to_index(docs: List[Dict[str, Any]]) -> Tuple[faiss.IndexFlatIP, np.ndarray, List[Dict[str, Any]]]:
    texts = [redact_text(d["text"]) for d in docs]
    embeddings = embed_texts(texts)
    index = build_faiss_index(embeddings)
    return index, embeddings, docs

def retrieve(query: str, index, docs: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    qvec = embed_texts([query]).astype(np.float32)
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    out = []
    for score, i in zip(D[0], I[0]):
        if i == -1:
            continue
        d = docs[i]
        out.append({"score": float(score), **d})
    return out

# ---------------------------
# Synthesis (no heavy LLM)
# ---------------------------
def synthesize_answer(query: str, hits: List[Dict[str, Any]]) -> str:
    """
    Lightweight “answer” by selecting & compressing top facts.
    """
    if not hits:
        return "I couldn’t find relevant data."
    # Extractive summary: pick top sentences and format
    bullets = []
    for h in hits:
        txt = h["text"]
        bullets.append(f"- {txt}")
    # Reduce redundancy using TF-IDF cosine
    tfidf = TfidfVectorizer(stop_words="english").fit([h["text"] for h in hits])
    X = tfidf.transform([h["text"] for h in hits]).toarray()
    keep = []
    used = np.zeros(len(hits), dtype=bool)
    for i in range(len(hits)):
        if used[i]: 
            continue
        keep.append(i)
        sim = cosine_similarity([X[i]], X)[0]
        for j, s in enumerate(sim):
            if j!=i and s > 0.7:
                used[j] = True
    bullets = [f"- {hits[i]['text']}" for i in keep][:8]
    answer = f"**Query:** {query}\n\n**Answer (data-grounded):**\n" + "\n".join(bullets)
    return answer

# ---------------------------
# Judge: NLI (optional) or heuristic fallback
# ---------------------------
class NLIFallback:
    def score(self, premise: str, hypothesis: str) -> float:
        # cosine similarity heuristic as a proxy for support
        tf = TfidfVectorizer(stop_words="english").fit([premise, hypothesis])
        P, H = tf.transform([premise]).toarray(), tf.transform([hypothesis]).toarray()
        return float(cosine_similarity(P, H)[0][0])

class TinyMNLIJudge:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny-mnli")

    @torch.inference_mode()
    def score(self, premise: str, hypothesis: str) -> float:
        """
        Returns probability that premise ENTAILS hypothesis.
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=384)
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        # label mapping for MNLI: [contradiction, neutral, entailment]
        return float(probs[2].item())

def get_judge():
    if HAS_TRANSFORMERS:
        try:
            return TinyMNLIJudge()
        except Exception:
            return NLIFallback()
    else:
        return NLIFallback()

# ---------------------------
# Simple MLOps-lite: run log
# ---------------------------
def log_interaction(row: Dict[str, Any]):
    df_path = "runs_log.csv"
    df = pd.DataFrame([row])
    if os.path.exists(df_path):
        old = pd.read_csv(df_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(df_path, index=False)

# ---------------------------
# Streamlit UI
# ---------------------------
def sidebar_controls():
    st.sidebar.header("⚙️ Controls")
    demo = st.sidebar.radio(
        "Choose Demo Dataset",
        ["Earthquakes (USGS)", "COVID-19 (disease.sh)"],
        index=0
    )
    st.sidebar.write("---")
    st.sidebar.caption("RAG Settings")
    topk = st.sidebar.slider("Top-K passages", 1, 10, 5)
    min_mag = st.sidebar.slider("Min magnitude (Earthquakes)", 3.0, 7.5, 4.5, 0.1)
    country = st.sidebar.text_input("Country for COVID (e.g., India) – leave blank for Global", "")
    st.sidebar.write("---")
    st.sidebar.caption("Governance")
    allow_logging = st.sidebar.checkbox("Store run logs (CSV)", value=True)
    st.sidebar.write("---")
    st.sidebar.caption("Diagnostics")
    show_traces = st.sidebar.checkbox("Show tool traces", value=True)
    show_hits = st.sidebar.checkbox("Show retrieved passages", value=True)
    return demo, topk, min_mag, country.strip() or "all", allow_logging, show_traces, show_hits

def header():
    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)
    st.markdown(
        "> **Note**: This MVP intentionally uses lightweight, free, open-source components. "
        "It substitutes heavy stacks (React/FastAPI/OAuth) with Streamlit + a tool router + login gate to fit free-tier constraints."
    )

def run_demo():
    demo, topk, min_mag, country, allow_logging, show_traces, show_hits = sidebar_controls()
    st.write("### 🔎 Ask a question about the selected dataset")
    query = st.text_input("e.g., 'Where and when were the strongest quakes last week?' or 'What’s the latest COVID status in India?'")
    col1, col2, col3 = st.columns(3)
    with col1:
        q_days = st.number_input("Query window (days)", min_value=1, max_value=30, value=7, step=1)
    with col2:
        fetch_btn = st.button("Fetch & Build RAG")
    with col3:
        ask_btn = st.button("Retrieve & Answer")

    # State
    if "rag_index" not in st.session_state:
        st.session_state["rag_index"] = None
        st.session_state["rag_docs"] = []
        st.session_state["tool_traces"] = []
        st.session_state["judge_ready"] = False
        st.session_state["judge"] = None

    # Build corpus
    if fetch_btn:
        t0 = time.time()
        st.session_state["tool_traces"] = []
        if "Earthquakes" in demo:
            end = datetime.utcnow().date()
            start = end - timedelta(days=int(q_days))
            payload, trace = call_tool(
                "earthquakes",
                start_date=str(start),
                end_date=str(end),
                min_mag=float(min_mag),
                limit=2000
            )
            st.session_state["tool_traces"].append(trace)
            docs = make_docs_from_earthquakes(payload)
        else:
            payload, trace = call_tool("covid", country=country)
            st.session_state["tool_traces"].append(trace)
            docs = make_docs_from_covid(payload, country)

        if not docs:
            st.warning("No data returned for the selected parameters.")
        else:
            with st.spinner("Embedding & indexing (lightweight)…"):
                index, embeddings, docs_store = corpus_to_index(docs)
            st.session_state["rag_index"] = index
            st.session_state["rag_docs"] = docs_store
            st.success(f"RAG index built with {len(docs_store)} documents in {round((time.time()-t0)*1000, 1)} ms.")

    # Judge init (lazy)
    if st.session_state.get("judge") is None:
        st.session_state["judge"] = get_judge()
        st.session_state["judge_ready"] = True

    # Query
    if ask_btn:
        if not query.strip():
            st.error("Please enter a question.")
            return
        if st.session_state["rag_index"] is None:
            st.error("Please fetch data and build the RAG index first.")
            return

        t0 = time.time()
        hits = retrieve(query, st.session_state["rag_index"], st.session_state["rag_docs"], k=topk)
        answer = synthesize_answer(query, hits)
        latency_ms = round((time.time() - t0) * 1000, 2)

        # Judge score
        judge = st.session_state["judge"]
        judge_scores = []
        for h in hits:
            judge_scores.append(judge.score(h["text"], answer))
        avg_support = float(np.mean(judge_scores)) if judge_scores else 0.0

        st.markdown("### ✅ Answer")
        st.markdown(answer)
        st.caption(f"Latency: {latency_ms} ms • Judge support score: {avg_support:.3f} (1.0 = strong entailment/support)")

        if show_hits:
            st.write("#### 🔍 Retrieved Passages")
            for i, h in enumerate(hits, 1):
                st.markdown(f"**{i}.** score={h['score']:.3f} — {h['text']}")
                if "url" in h.get("meta", {}) and h["meta"]["url"]:
                    st.caption(h["meta"]["url"])

        if show_traces and st.session_state["tool_traces"]:
            st.write("#### 🧰 Tool Traces (MCP substitute)")
            st.json(st.session_state["tool_traces"])

        # Log run
        if allow_logging:
            row = {
                "id": str(uuid.uuid4()),
                "ts": datetime.utcnow().isoformat()+"Z",
                "demo": demo,
                "query": query,
                "topk": topk,
                "latency_ms": latency_ms,
                "avg_support": avg_support,
                "docs": len(st.session_state["rag_docs"])
            }
            log_interaction(row)

    # Eval tab
    with st.expander("📈 Evaluations & Logs"):
        st.write("Simple MLOps-lite view of stored interactions.")
        path = "runs_log.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.dataframe(df.tail(200), use_container_width=True)
        else:
            st.caption("No runs logged yet.")

def main():
    # Login / Governance
    ok = login_gate()
    if not ok:
        st.stop()

    header()
    run_demo()
    st.write("---")
    st.caption("Sources: USGS Earthquake API, disease.sh COVID-19 API • Embeddings: fastembed(ONNX) • Vector DB: FAISS • Optional NLI: prajjwal1/bert-tiny-mnli")

if __name__ == "__main__":
    main()
