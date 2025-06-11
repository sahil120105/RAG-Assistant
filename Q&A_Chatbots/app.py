import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time


from dotenv import load_dotenv

load_dotenv()
UPLOAD_DIR = "data"

#API Key
groq_api_key = os.getenv("GROQ_API_KEY")


st.title("Chatgroq With Llama3")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")


# Prompt 
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

# Create Embeddings
def vector_embeddings():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        st.session_state.loader = PyPDFDirectoryLoader("data")   # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()      # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:10])    # Change this according to the document length wanted
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)   # Create vector db


#Ensures the upload directory exists.
def ensure_upload_directory_exists():
    
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        st.success(f"Created directory: {UPLOAD_DIR}")
    

#Upload File -------------------------------------------------------------------------------------------------------------------

st.write("Upload a file you want to query from")
# Ensure the upload directory exists when the app starts
ensure_upload_directory_exists()

uploaded_file = st.file_uploader("Choose a file...", type=None) # type=None allows all file types



if uploaded_file is not None:
        # Get the original file name
        file_name = uploaded_file.name
        
        # Create the full path to save the file
        file_path = os.path.join(UPLOAD_DIR, file_name)

        try:
            # Write the file to the specified directory
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File '{file_name}' successfully saved!`")

            # Optional: Display file details
            st.subheader("File Details:")
            st.write(f"**File Name:** {file_name}")
            st.write(f"**File Type:** {uploaded_file.type}")
            st.write(f"**File Size:** {uploaded_file.size / (1024*1024):.2f} MB")

        except Exception as e:
            st.error(f"An error occurred while saving the file: {e}")

#-------------------------------------------------------------------------------------------------------------------


if st.button("Document Embedding"):
    vector_embeddings()
    st.success("Vector Store DB is ready!")


prompt1 = st.text_input("Enter your question from the document")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)



    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time: ", time.process_time()-start)
    st.write(response['answer'])
    print(response)

    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------------")







