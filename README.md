# RAG Knowledge Assistant

> **Intelligent question-answering system powered by Retrieval-Augmented Generation for custom knowledge bases**

## Description

A streamlined RAG system that provides accurate answers from your custom documents without hallucination. Built with Streamlit for easy web interaction, this tool lets you upload documents or load from URLs, then query your knowledge base using retrieval techniques combined with large language models.

**RAG:** Instead of relying on potentially inaccurate LLM responses, this system first searches your documents for relevant information, then generates contextually grounded answers based on that retrieved content.

## Features

- **📁 Dual Upload Options**: Upload documents via Streamlit interface or load from web URLs
- **🔍 Smart Document Processing**: Automatic text chunking and semantic search with FAISS
- **🧠 Multiple LLM Support**: Llama3 (Ollama) or Groq
- **🔗 Flexible Embeddings**: Hugging Face Sentence Transformers and Ollama embeddings
- **🚫 Zero Hallucination**: Responses grounded only in your provided documents
- **⚡ Local & Cloud Options**: Run locally with Ollama or use cloud APIs
- **🎯 Stuff Document Chain**: Efficient document concatenation for context building

## How It Works (Architecture/Flow)

```
1. 📥 Document Input
   ├── Upload files via Streamlit (Q&A_Chatbots/app.py)
   └── Load from URLs via web interface (groq/app.py)

2. ✂️ Document Processing
   ├── Split documents into optimized chunks
   ├── Apply text cleaning and preprocessing
   └── Use stuff document chain for context assembly

3. 🔢 Embedding & Storage
   ├── Generate vector embeddings (HF Transformers/Ollama)
   └── Store in FAISS vector database for fast retrieval

4. 🔍 Query & Retrieve
   ├── Convert user question to embeddings
   ├── Perform similarity search in vector store
   └── Retrieve most relevant document chunks

5. 🤖 Generate Response
   ├── Combine retrieved context using stuff chain
   └── Generate grounded answer with selected LLM
```

## Installation & Setup

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/rag-knowledge-assistant.git
cd rag-knowledge-assistant

# Install dependencies
pip install -r requirements.txt

```

### Environment Variables
Add to your `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

### Running the Applications

```bash
cd final_rag
streamlit run app.py
```

### Optional: Local LLM Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama3 model
ollama pull llama3
```

## Technologies Used

- **🐍 Python & Streamlit**: Web interface and core logic
- **🦙 LangChain**: RAG pipeline and stuff document chains
- **🤗 Hugging Face**: Sentence transformers for embeddings
- **🔍 FAISS**: Vector similarity search and storage
- **🦙 Ollama**: Local LLM runtime (Llama3)
- **⚡ Groq**: High-speed LLM inference
- **🤖 OpenAI**: GPT model integration
- **📄 Document Processing**: Multi-format support with intelligent chunking
