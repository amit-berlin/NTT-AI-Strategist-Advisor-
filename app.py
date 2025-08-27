import os, time, uuid
from datetime import datetime, timedelta
import requests, numpy as np, pandas as pd, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding
import faiss

APP_TITLE = "NTT 22nd Century Enterprise AI Knowledge & Decision Platform (MVP)"
APP_TAGLINE = "Lightweight RAG • Tool-calling • Governance-lite • 22nd Century NTT"

# ---------------------------
# Utility: Redaction
# ---------------------------
def redact_text(text: str) -> str:
    text = str(text)
    return pd.Series([text]).str.replace(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[REDACTED_EMAIL]", regex=True
    ).str.replace(
        r"\b(?:\+?\d[\d -]{8,}\d)\b", "[REDACTED_PHONE]", regex=True
    ).iloc[0]

# ---------------------------
# Tool router (MCP substitute)
# ---------------------------
def tool_fetch_earthquakes(start_date, end_date, min_mag=4.5, limit=200):
    try:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        params = {"format": "geojson","starttime": start_date,"endtime": end_date,
                  "minmagnitude": min_mag,"limit": limit}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        st.warning("⚠️ Earthquake API busy → substituting with demo sample")
        return {"features":[{"id":"demo","properties":{
            "mag":5.0,"place":"Demo Place","time":int(time.time()*1000),
            "url":"https://example.com"}}]}

def tool_fetch_covid(country="all"):
    try:
        url = f"https://disease.sh/v3/covid-19/all" if country=="all" \
              else f"https://disease.sh/v3/covid-19/countries/{country}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        st.warning("⚠️ COVID API busy → substituting with demo sample")
        return {"cases":123456,"deaths":7890,"recovered":100000,"active":3456,
                "todayCases":123,"todayDeaths":4,"todayRecovered":111,
                "critical":12,"tests":55555,"population":999999,"updated":int(time.time()*1000)}

# ---------------------------
# Build documents
# ---------------------------
def make_docs_from_earthquakes(geojson):
    docs=[]
    for f in geojson.get("features",[]):
        p=f.get("properties",{})
        txt=f"Magnitude {p.get('mag')} earthquake at {p.get('place')} UTC {datetime.utcfromtimestamp(p.get('time',0)/1000).isoformat()}."
        docs.append({"id":f.get("id","demo"),"text":txt})
    return docs

def make_docs_from_covid(j,country):
    updated=datetime.utcfromtimestamp(j.get("updated",time.time())/1000).isoformat()
    txt=f"COVID in {country.title()}: cases={j.get('cases')}, deaths={j.get('deaths')}, recovered={j.get('recovered')}, active={j.get('active')}. Updated {updated}."
    return [{"id":str(uuid.uuid4()),"text":txt}]

# ---------------------------
# RAG
# ---------------------------
@st.cache_resource
def get_embedder():
    return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    emb=get_embedder()
    return np.array(list(emb.embed(texts)),dtype=np.float32)

def build_index(docs):
    texts=[redact_text(d["text"]) for d in docs]
    vecs=embed_texts(texts)
    faiss.normalize_L2(vecs)
    index=faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index,docs,vecs

def retrieve(query,index,docs,k=5):
    q=embed_texts([query]); faiss.normalize_L2(q)
    D,I=index.search(q,k)
    return [docs[i] for i in I[0] if i!=-1]

# ---------------------------
# Answer synthesis
# ---------------------------
def synthesize(query,hits):
    if not hits: return "No relevant info."
    return "**Query:** "+query+"\n\n**Answer:**\n"+"\n".join([f"- {h['text']}" for h in hits])

# ---------------------------
# Judge (safe fallback only)
# ---------------------------
class HeuristicJudge:
    def score(self,p,h): 
        tf=TfidfVectorizer().fit([p,h]);X=tf.transform([p,h]).toarray()
        return float(cosine_similarity([X[0]],[X[1]])[0][0])
def get_judge(): 
    st.info("⚠️ TinyMNLI Judge busy → using lightweight heuristic judge")
    return HeuristicJudge()

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.title(APP_TITLE); st.caption(APP_TAGLINE)
    demo=st.radio("Choose Demo",["Earthquakes","COVID"])
    query=st.text_input("Ask a question")
    if st.button("Run Demo"):
        if demo=="Earthquakes":
            data=tool_fetch_earthquakes(str(datetime.utcnow().date()-timedelta(days=7)),str(datetime.utcnow().date()))
            docs=make_docs_from_earthquakes(data)
        else:
            data=tool_fetch_covid("all")
            docs=make_docs_from_covid(data,"global")
        idx,docs,_=build_index(docs)
        hits=retrieve(query or "latest update",idx,docs,3)
        ans=synthesize(query or "latest update",hits)
        judge=get_judge()
        score=np.mean([judge.score(h["text"],ans) for h in hits]) if hits else 0
        st.markdown(ans)
        st.caption(f"Judge support score: {score:.2f}")

if __name__=="__main__": main()
