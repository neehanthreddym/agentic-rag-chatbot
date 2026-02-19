# Agentic RAG Chatbot

## Overview
This project is an **Agentic RAG Chatbot** that combines file-grounded Q&A with durable memory and intelligent query routing. It demonstrates an AI-first product feature capable of:

- **File-grounded Q&A (RAG)**: Answering questions based on uploaded documents with citations.
- **Durable Memory**: Storing high-signal facts about the user and the organization to personalize future interactions.
- **Intelligent Routing**: Dynamically deciding whether to search documents, lookup memory, or answer generally.

## Features

### 1. RAG Pipeline (Retrieval Augmented Generation)
- **Ingestion**: parses and chunks PDF documents.
- **Retrieval**: Semantic search using embeddings to find relevant context.
- **Grounded Answers**: Generates responses strictly based on the retrieved context with citations.

### 2. Persistent Memory
The bot extracts and stores key information to:
- `USER_MEMORY.md`: User-specific preferences and facts (e.g., "User is a Data Scientist").
- `COMPANY_MEMORY.md`: Organizational knowledge (e.g., "Project X deadline is Friday").

### 3. Agentic Routing
A routing layer determines the intent of the user's query:
- **Search**: For questions requiring external documentation.
- **Memory**: For questions about the user or past context.
- **General**: For chit-chat or general knowledge.

## Quick Start

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Google API Key or Groq API Key

### Installation

```bash
# clone the repo
git clone https://github.com/neehanthreddym/agentic-rag-chatbot.git
cd agentic-rag-chatbot

# create virtual environment & install dependencies
uv venv
uv pip install -r requirements.txt

# configure API keys
cp .env.example .env
# edit .env and add your GOOGLE_API_KEY
```

### Running the Application

**Streamlit Web Interface**
```bash
make chat
# Opens http://localhost:8501
```
Upload a PDF in the sidebar, then ask questions. The bot will use memory and documents to answer.

**CLI / Testing**
```bash
# Run the end-to-end sanity check
make sanity
```

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of the ingestion, retrieval, and memory modules.