import os
import torch
from dotenv import load_dotenv
from langchain_community.embeddings import  HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import streamlit as st

load_dotenv()

st.set_page_config(page_title="  Your PDF Chatbot " , layout="wide")
st.title("Upload and chat with your PDF")


st.sidebar.header("Upload your PDF")
uploaded_pdf=st.sidebar.file_uploader("Choose a PDF" , type="pdf")

if uploaded_pdf:

    pdfpath=os.path.join('data' ,uploaded_pdf.name)
    os.makedirs("data" , exist_ok=True)
    with open(pdfpath , 'wb' ) as f:
        f.write(uploaded_pdf.read())

    loader=PyPDFLoader(pdfpath)
    document=loader.load()

    textsplitter=RecursiveCharacterTextSplitter(chunk_size=800 , chunk_overlap=200)
    docs=textsplitter.split_documents(document)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embedding=embedding_model, persist_directory="chroma_db")
    vector_store.persist()
    
    model_name="google/flan-t5-base"
    generator=pipeline(
        "text2text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]

    st.header("Ask me Anything about your PDF")
    query=st.text_input("Your Question ")
    if query:
        results=vector_store.similarity_search(query , k=3)
        context=" ".join([doc.page_content for doc in results])
        unique_pages = sorted(set([doc.metadata.get('page', 'N/A') for doc in results]))
        sources = [f"Page {p}" for p in unique_pages]
        prompt=f"""you are an AI tutor , Use the following content to answer the question in detail
        context: {context}
        question: {query}
        Give a clear, step-by-step, beginner-friendly explanation.
        """
        result=generator(prompt , max_new_tokens=230 , temperature=0.9 , do_sample=True , repetition_penalty=2.0 )
        answer = result[0]['generated_text']
        answer_with_sources = answer + "\n\n**Sources:** " + ", ".join(sources)

        st.session_state.chat_history.append({"user": query, "bot": answer_with_sources})

        st.success(answer_with_sources)


    if st.session_state.chat_history:
        st.subheader("chat history")
        for chat in st.session_state.chat_history:
            st.chat_message("user").markdown(chat["user"])
            st.chat_message("assistant").markdown(chat["bot"])

else:
    st.warning("Please Upload a PDF to get started ")

    
