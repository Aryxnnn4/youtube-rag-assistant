# YouTube RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to ask
questions about YouTube videos and receive grounded answers based on the
video transcript.

The system uses semantic search with FAISS and MMR-based retrieval to
provide relevant and diverse context for answer generation.

---

## 🚀 Features

- 📺 Load and process YouTube video transcripts
- ✂️ Intelligent text chunking
- 🧠 Semantic search using FAISS vector store
- 🔀 MMR retrieval for better answer diversity
- 💾 Persistent embeddings (no re-indexing on restart)
- 🔍 Source attribution (video + chunk references)
- 🌐 Streamlit web UI for interactive querying

---

## 🧩 Architecture Overview
```
YouTube Video
↓
Transcript Extraction
↓
Text Chunking
↓
Embedding Generation
↓
FAISS Vector Store (Persistent)
↓
MMR-based Retrieval
↓
LLM (LLaMA 3.1 via HuggingFace)
↓
Answer + Sources (Streamlit UI)
```
---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **FAISS**
- **HuggingFace Inference API**
- **Streamlit**
- **YouTube Transcript API**

---

## ⚙️ Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/aryan-bhadana/youtube-rag-assistant.git
cd youtube-rag-assistant 
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Set environment variables
```bash
HUGGINGFACEHUB_API_TOKEN=your_api_key_here
```
4. Run the Application
```bash
streamlit run ui.py
```
## Enter:
- A YouTube video ID
- A natural language question

## The app will return:
- An answer grounded in the transcript
- The transcript chunks used as sources

## 🧠 Example Questions
- What challenges are discussed in this video?
- Is nuclear fusion mentioned?
- What problem does the speaker try to solve?

## 🔮 Future Improvements
- Multi-video ingestion
- Timestamp-level source attribution
- Conversation memory
- Evaluation metrics for retrieval quality

## 📌 Why This Project?
This project demonstrates how modern RAG systems are built in practice,
focusing on:
- Retrieval quality (MMR)
- Performance (persistent vector store)
- Transparency (source attribution)
- Usability (web UI)
