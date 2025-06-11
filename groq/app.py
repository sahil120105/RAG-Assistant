import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time


load_dotenv()

# Load Groq API Key
groq_api_key = os.environ["GROQ_API_KEY"]


if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")

    # Fetch and load the content of the webpage(s)
    st.session_state.docs = st.session_state.loader.load()

    # Split the loaded documents into manageable chunks.
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    
    # Create a FAISS vector store from the document chunks and their embeddings.
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


st.title("Chat with Groq")
llm = ChatGroq(groq_api_key=groq_api_key, model_name= "mistral-saba-24b" )

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>

    Question: {input}
    """
)

#The chain is reposible for filling the context with documents retrieved from the retriever
document_chain = create_stuff_documents_chain(llm,
                    prompt,
                    document_variable_name="context"    # Variable name to use for the formatted documents in the prompt. Defaults to “context”.
                    )  


#Creates a tool to fetch relevant document chunks from our FAISS vector store.
retriever = st.session_state.vectors.as_retriever()
# Orchestrates the retrieval of documents and feeding them to the LLM via the document chain.
retrieval_chain = create_retrieval_chain(retriever, document_chain)


prompt = st.text_input("Input your prompt here")


if(prompt):
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response Time: ", time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------------------")