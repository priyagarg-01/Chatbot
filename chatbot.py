import os
import torch
from dotenv import load_dotenv
from langchain_community.embeddings import  HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
from sentence_transformers import CrossEncoder
import streamlit as st
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client=OpenAI(api_key=OPENAI_API_KEY)


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

    reranker = CrossEncoder(
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda" if torch.cuda.is_available() else 'cpu'
    )
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]

    st.header("Ask me Anything about your PDF")
    query=st.text_input("Your Question ")


    def get_answerwithchatCompletion(context , query):
        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system" , "content" : "You are an AI tutor who explains concepts simply and clearly."},
                {"role": "user" , "content" : f"Context: {context}\n\nQuestion: {query}\n\nExplain step by step like I'm a beginner."}
                ],
            temperature=0.8,
            max_tokens=400
        )

        return response.choices[0].message.content
        

    if query:
        results=vector_store.similarity_search(query , k=5)

        pairs = [[query, doc.page_content] for doc in results]
        scores=reranker.predict(pairs)
        scored_chunks=list(zip(results , scores))
        sorted_chunks=sorted(scored_chunks , key=lambda x : x[1] , reverse=True)
        top_chunks=sorted_chunks[:3]

        context = " ".join([doc.page_content for doc, score in top_chunks])
        unique_pages = sorted(set([doc.metadata.get('page', 'N/A') for doc in results]))
        sources = [f"Page {p}" for p in unique_pages]
        
    
        answer=get_answerwithchatCompletion(context , query)
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

    
