# AI Event Ops Agent

An MVP AI assistant designed to answer exhibitor and speaker questions using event documentation.

Built using Google Vertex AI, FastAPI, and Flutter.

---

## 🚀 What it does

- Answers questions from **static event documents**
- Uses **RAG (Retrieval Augmented Generation)**
- Returns:
  - Answer
  - Source citations (doc + section)
  - Confidence flag (answer vs escalate)
- Logs unknown questions for review

---

## 🧠 Tech Stack

- Frontend: Flutter (single chat interface)
- Backend: FastAPI (Cloud Run)
- AI:
  - Vertex AI (Gemini)
  - text-embedding-004
- Storage:
  - Cloud Storage (RAG index)
  - Firestore (escalation logs)

---

## 🏗 Architecture

User → Flutter App → FastAPI (Cloud Run)  
→ Retrieve relevant document chunks  
→ Call Vertex AI (strict grounded response)  
→ Return answer + citations + confidence  
→ Log escalation (if needed)

---

## ⚙️ How it works

1. Documents are chunked and embedded
2. Stored as a lightweight index (`index.json`)
3. Query flow:
   - Embed user question
   - Retrieve top-k similar chunks
   - Apply similarity threshold
   - Call LLM with strict grounding rules
4. If confidence is low → escalate

---

## ✅ Features

- No hallucinations (strict doc-only answering)
- Safe fallback (escalation)
- Lightweight MVP architecture
- Fully deployable on GCP

---

## 📦 Current Limitations (Intentional)

- Single event only
- Static documents only
- No authentication
- No dashboard / analytics UI

---

## 🧪 Example Queries

**Answerable:**
- "When does build-up start?"
- "When does breakdown begin?"

**Escalated:**
- "What is the WiFi password?"
- "How do I book accommodation?"

---

## 📌 Why this project

This project demonstrates:
- End-to-end AI system design
- Practical RAG implementation
- Cloud deployment (Cloud Run)
- Safe AI patterns (no hallucination + escalation)

---

## 🔮 Next Steps

- Upload real event documentation
- Improve chunking for PDFs
- Add multi-event support
- Enhance retrieval accuracy

---

## 🧑‍💻 Author

Built by Zackary Puttock
