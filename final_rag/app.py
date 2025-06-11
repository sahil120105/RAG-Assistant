import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama  # Import Ollama for local LLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader  # Import WebBaseLoader
import time


from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
UPLOAD_DIR = "data"

# API Key
groq_api_key = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="RAG with Llama3 & Ollama", layout="centered")
st.title("RAG with Llama3 (Groq/Ollama)")

# --- LLM Selection ---
st.sidebar.header("LLM Configuration")
llm_choice = st.sidebar.radio("Choose your LLM:", ("Groq (Cloud)", "Ollama (Local)"))

llm = None
if llm_choice == "Groq (Cloud)":
    if not groq_api_key:
        st.sidebar.error("GROQ_API_KEY not found in environment variables!")
        st.stop()
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mistral-saba-24b")
    st.sidebar.success("Using Groq's mistral-saba-24b")
else: # Ollama (Local)
    ollama_model_name = st.sidebar.text_input("Ollama Model Name (e.g., llama3.2:1b)", "llama3.2:1b")
    try:
        llm = Ollama(model=ollama_model_name)
        st.sidebar.success(f"Using local Ollama model: {ollama_model_name}")
    except Exception as e:
        st.sidebar.error(f"Could not connect to Ollama. Is it running? Error: {e}")
        st.stop()

if llm is None:
    st.error("LLM not initialized. Please check your configuration.")
    st.stop()

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# --- Data Ingestion and Embedding ---
def vector_embeddings(loader_type, source_input=None):
    """
    Handles loading documents from either PDF directory or a URL,
    then chunks and creates FAISS embeddings.
    """
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b") # Keep Ollama for embeddings

        st.info(f"Loading documents from {loader_type}...")
        start_load_time = time.process_time()

        if loader_type == "PDF":
            st.session_state.loader = PyPDFDirectoryLoader(UPLOAD_DIR)
        elif loader_type == "Web":
            if not source_input:
                st.error("Please provide a URL for Web loading.")
                return
            st.session_state.loader = WebBaseLoader(source_input)

        try:
            st.session_state.docs = st.session_state.loader.load()
            st.success(f"Documents loaded in {time.process_time() - start_load_time:.2f} seconds.")
            
            st.info("Splitting documents into chunks...")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            # Limit the number of documents to avoid excessive memory usage for large files/webpages
            # For web, considering first 50 is a reasonable start. For PDFs, might be all or first N.
            if loader_type == "PDF":
                st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
            elif loader_type == "Web":
                st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
            st.success(f"Documents split into {len(st.session_state.final_docs)} chunks.")

            st.info("Creating FAISS vector store...")
            start_vector_time = time.process_time()
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
            st.success(f"Vector Store DB ready in {time.process_time() - start_vector_time:.2f} seconds!")
        except Exception as e:
            st.error(f"An error occurred during document loading or embedding: {e}")
            st.session_state.pop("vectors", None) # Clear partial state if error occurs


# Ensures the upload directory exists.
def ensure_upload_directory_exists():
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        st.success(f"Created directory for PDF uploads: {UPLOAD_DIR}")


# --- File Upload/URL Input Options ---
st.subheader("Choose your data source:")
data_source_choice = st.radio("Select input method:", ("Upload PDF", "Enter URL"))

if data_source_choice == "Upload PDF":
    st.write("Upload a PDF file you want to query from:")
    ensure_upload_directory_exists()

    uploaded_file = st.file_uploader("Choose a PDF file...", type="pdf") # Explicitly set type to pdf

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_path = os.path.join(UPLOAD_DIR, file_name)

        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File '{file_name}' successfully saved to '{UPLOAD_DIR}'!")

            st.subheader("File Details:")
            st.write(f"**File Name:** {file_name}")
            st.write(f"**File Type:** {uploaded_file.type}")
            st.write(f"**File Size:** {uploaded_file.size / (1024*1024):.2f} MB")

        except Exception as e:
            st.error(f"An error occurred while saving the file: {e}")

    if st.button("Process PDF Document"):
        if uploaded_file is not None:
            vector_embeddings("PDF")
        else:
            st.warning("Please upload a PDF file first.")

elif data_source_choice == "Enter URL":
    st.write("Enter the URL of the webpage you want to query from:")
    url_input = st.text_input("URL:", "https://docs.smith.langchain.com/")

    if st.button("Process Webpage"):
        if url_input:
            vector_embeddings("Web", url_input)
        else:
            st.warning("Please enter a URL.")

# --- Chat Interface ---
st.markdown("---")
st.subheader("Ask your question:")
prompt1 = st.text_input("Enter your question from the document/webpage")

if prompt1:
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.error("Please process a document or webpage first using the 'Process' button.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("Generating response..."):
            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt1})
            end = time.process_time()
            st.success(f"Response generated in {end - start:.2f} seconds.")

        st.write("### Answer:")
        st.write(response['answer'])

        with st.expander("Document Similarity Search (Context Used)"):
            if 'context' in response:
                for i, doc in enumerate(response["context"]):
                    st.write(f"**Source Document {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")
            else:
                st.write("No context found in the response.")