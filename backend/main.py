import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from google.cloud import storage
from google.cloud import firestore

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


# ----------------------------
# Config (env vars)
# ----------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "").strip()
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "europe-west2").strip()

# Required by spec: load precomputed index.json from Cloud Storage on startup
INDEX_GCS_URI = os.getenv("INDEX_GCS_URI", "gs://ai-event-ops-zack-484019/index.json").strip()

# Local fallback for development convenience (optional)
INDEX_LOCAL_PATH = os.getenv("INDEX_LOCAL_PATH", "").strip()

# Gemini model (keep cheap + fast for MVP)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash").strip()

# Retrieval settings
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_TOP_SCORE = float(os.getenv("MIN_TOP_SCORE", "0.72"))  # if below -> escalate
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))  # keep prompt small


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class Chunk:
    id: str
    doc_name: str
    source_type: str           # "pdf"|"text"
    page: Optional[int]        # for pdf
    section: Optional[str]     # for text headings
    chunk_index: int
    text: str
    embedding: List[float]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class Citation(BaseModel):
    doc: str
    page: Optional[int] = None
    section: Optional[str] = None
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: str  # "answer" | "escalate"
    reason: Optional[str] = None


# ----------------------------
# Globals (in-memory index)
# ----------------------------
app = FastAPI(title="AI Event Ops Agent API", version="0.1.0")

_chunks: List[Chunk] = []
_emb_matrix: Optional[np.ndarray] = None  # shape: (n_chunks, dim)
_emb_norms: Optional[np.ndarray] = None   # shape: (n_chunks,)
_llm: Optional[GenerativeModel] = None

_firestore_client: Optional[firestore.Client] = None


# ----------------------------
# Helpers: GCS + Index loading
# ----------------------------
def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"INDEX_GCS_URI must start with gs://, got: {gs_uri}")
    path = gs_uri[len("gs://"):]
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid gs:// uri (need bucket/object): {gs_uri}")
    return parts[0], parts[1]


def _load_index_json() -> Dict[str, Any]:
    # Primary: GCS (per spec)
    if INDEX_GCS_URI:
        bucket, obj = _parse_gs_uri(INDEX_GCS_URI)
        client = storage.Client()
        blob = client.bucket(bucket).blob(obj)
        data = blob.download_as_text(encoding="utf-8")
        return json.loads(data)

    # Fallback: local file (dev-only)
    if INDEX_LOCAL_PATH:
        with open(INDEX_LOCAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    raise RuntimeError("No index source configured. Set INDEX_GCS_URI (recommended) or INDEX_LOCAL_PATH.")


def _build_in_memory_index(index_obj: Dict[str, Any]) -> None:
    global _chunks, _emb_matrix, _emb_norms

    raw_chunks = index_obj.get("chunks", [])
    if not raw_chunks:
        raise RuntimeError("index.json has no chunks")

    _chunks = []
    embs = []

    for rc in raw_chunks:
        c = Chunk(
            id=rc["id"],
            doc_name=rc["doc_name"],
            source_type=rc.get("source_type", "text"),
            page=rc.get("page"),
            section=rc.get("section"),
            chunk_index=int(rc.get("chunk_index", 0)),
            text=rc["text"],
            embedding=rc["embedding"],
        )
        _chunks.append(c)
        embs.append(np.array(c.embedding, dtype=np.float32))

    _emb_matrix = np.vstack(embs)  # (n, dim)
    _emb_norms = np.linalg.norm(_emb_matrix, axis=1)
    if np.any(_emb_norms == 0):
        # avoid divide-by-zero (rare)
        _emb_norms = np.where(_emb_norms == 0, 1e-8, _emb_norms)


# ----------------------------
# Helpers: Retrieval
# ----------------------------
def _cosine_sim_to_all(query_vec: np.ndarray) -> np.ndarray:
    # cosine(query, doc) = dot / (||q|| * ||d||)
    global _emb_matrix, _emb_norms
    if _emb_matrix is None or _emb_norms is None:
        raise RuntimeError("Index not loaded")

    q = query_vec.astype(np.float32)
    qn = np.linalg.norm(q)
    if qn == 0:
        qn = 1e-8

    dots = _emb_matrix @ q
    sims = dots / (qn * _emb_norms)
    return sims


def _select_top_k(sims: np.ndarray, k: int) -> List[Tuple[float, Chunk]]:
    idx = np.argpartition(-sims, min(k, len(sims)) - 1)[:k]
    scored = sorted([(float(sims[i]), _chunks[i]) for i in idx], key=lambda x: x[0], reverse=True)
    return scored


def _build_context(top: List[Tuple[float, Chunk]]) -> Tuple[str, List[Citation], float]:
    """
    Build a context block with strict citations. We always return citations
    for the chunks provided to the model. The model must cite only these.
    """
    parts: List[str] = []
    cites: List[Citation] = []
    best = top[0][0] if top else 0.0
    total = 0

    for score, c in top:
        cite = Citation(doc=c.doc_name, page=c.page, section=c.section, chunk_id=c.id)
        cites.append(cite)

        header = f"[SOURCE: {c.id} | doc={c.doc_name}"
        if c.page is not None:
            header += f" | page={c.page}"
        if c.section:
            header += f" | section={c.section}"
        header += "]"

        chunk_text = c.text.strip()
        block = f"{header}\n{chunk_text}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)

    context = "\n".join(parts).strip()
    return context, cites, best


# ----------------------------
# Helpers: Vertex AI (LLM)
# ----------------------------
def _init_vertex() -> None:
    global _llm
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID env var is required")
    vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
    _llm = GenerativeModel(LLM_MODEL_NAME)


def _call_llm_strict(question: str, context: str, allowed_chunk_ids: List[str]) -> Dict[str, Any]:
    """
    Forces JSON output. If model output is not valid JSON or tries to cite unknown sources,
    we treat as escalate.
    """
    global _llm
    if _llm is None:
        raise RuntimeError("LLM not initialized")

    allowed = "\n".join([f"- {cid}" for cid in allowed_chunk_ids])

    system_rules = f"""
You are an event operations assistant. You MUST follow these rules:
1) Answer ONLY using the provided context sources.
2) If the answer is not clearly present in the context, you MUST escalate.
3) NEVER use outside knowledge.
4) You MUST return a single JSON object and nothing else.

CITATION RULE:
- You may cite ONLY these source ids:
{allowed}

JSON schema:
{{
  "confidence": "answer" | "escalate",
  "answer": "<short helpful answer or empty if escalating>",
  "citations": ["<source_id>", "..."], 
  "reason": "<required when confidence=escalate; explain what's missing>"
}}

If confidence="answer":
- "citations" must contain 1-3 source ids that directly support the answer.
If confidence="escalate":
- "answer" must be "" (empty string)
- "citations" must be []
"""

    prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}
"""

    resp = _llm.generate_content(
        [system_rules, prompt],
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=500,
            response_mime_type="application/json",
        ),
    )

    text = (resp.text or "").strip()

    # Parse JSON safely
    try:
        obj = json.loads(text)
    except Exception:
        return {"confidence": "escalate", "answer": "", "citations": [], "reason": "Model returned non-JSON output"}

    # Validate fields
    conf = obj.get("confidence")
    if conf not in ("answer", "escalate"):
        return {"confidence": "escalate", "answer": "", "citations": [], "reason": "Invalid confidence value"}

    citations = obj.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    # Validate citation ids
    allowed_set = set(allowed_chunk_ids)
    if any((cid not in allowed_set) for cid in citations):
        return {"confidence": "escalate", "answer": "", "citations": [], "reason": "Model cited unknown source id(s)"}

    if conf == "answer":
        ans = (obj.get("answer") or "").strip()
        if not ans:
            return {"confidence": "escalate", "answer": "", "citations": [], "reason": "Empty answer"}
        if len(citations) == 0:
            return {"confidence": "escalate", "answer": "", "citations": [], "reason": "Answer missing citations"}
        return {"confidence": "answer", "answer": ans, "citations": citations, "reason": None}

    # escalate
    reason = (obj.get("reason") or "").strip() or "Information not found in provided documents"
    return {"confidence": "escalate", "answer": "", "citations": [], "reason": reason}


# ----------------------------
# Helpers: Firestore escalation logging
# ----------------------------
def _init_firestore() -> None:
    global _firestore_client
    _firestore_client = firestore.Client()


def _log_escalation(session_id: Optional[str], question: str, reason: str, top_hits: List[Tuple[float, Chunk]]) -> None:
    global _firestore_client
    if _firestore_client is None:
        return

    doc = {
        "ts_unix": time.time(),
        "session_id": session_id or None,
        "question": question,
        "reason": reason,
        "top_hits": [
            {
                "score": float(score),
                "chunk_id": c.id,
                "doc_name": c.doc_name,
                "page": c.page,
                "section": c.section,
            }
            for score, c in top_hits
        ],
    }
    _firestore_client.collection("escalations").add(doc)


# ----------------------------
# FastAPI lifecycle
# ----------------------------
@app.on_event("startup")
def startup() -> None:
    # Load index from GCS
    index_obj = _load_index_json()
    _build_in_memory_index(index_obj)

    # Init Vertex + Firestore
    _init_vertex()
    _init_firestore()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "chunks_loaded": len(_chunks),
        "model": LLM_MODEL_NAME,
        "index_source": INDEX_GCS_URI if INDEX_GCS_URI else INDEX_LOCAL_PATH,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.message.strip()

    # Embed query using SAME embedding model as index
    # We use the embedding model via aiplatform SDK for simplicity:
    from vertexai.language_models import TextEmbeddingModel
    emb_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    q_vec = np.array(emb_model.get_embeddings([question])[0].values, dtype=np.float32)

    sims = _cosine_sim_to_all(q_vec)
    top = _select_top_k(sims, TOP_K)

    context, citations_all, best_score = _build_context(top)

    # Early escalation if retrieval is weak
    if not top or best_score < MIN_TOP_SCORE or not context:
        reason = f"Low retrieval confidence (best_score={best_score:.3f}) or insufficient context"
        _log_escalation(req.session_id, question, reason, top)
        return ChatResponse(
            answer="",
            citations=[],
            confidence="escalate",
            reason=reason,
        )

    allowed_chunk_ids = [c.chunk_id for c in citations_all]  # Citation.chunk_id
    llm_out = _call_llm_strict(question, context, allowed_chunk_ids)

    if llm_out["confidence"] == "escalate":
        _log_escalation(req.session_id, question, llm_out["reason"], top)
        return ChatResponse(
            answer="",
            citations=[],
            confidence="escalate",
            reason=llm_out["reason"],
        )

    # Map cited chunk ids back to citation objects
    cited = set(llm_out["citations"])
    cite_map = {c.chunk_id: c for c in citations_all}
    final_cites = [cite_map[cid] for cid in llm_out["citations"] if cid in cite_map]

    return ChatResponse(
        answer=llm_out["answer"],
        citations=final_cites,
        confidence="answer",
        reason=None,
    )
