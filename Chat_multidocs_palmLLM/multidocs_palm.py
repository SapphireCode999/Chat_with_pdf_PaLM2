import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings



load_dotenv()

@st.cache_resource
def load_model():
    palm_llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1) 
    
    return palm_llm


# function for embedding the text vector store
@st.cache_resource
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def extract_docs(uploaded_files):
    texts = ""
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            texts += page.extract_text()
    
    return texts


           
def chunk_texts(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ' ', ''],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(texts)
    return text_chunks
    


def get_chain(vectorstore):
    palm_llm = load_model()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=palm_llm, 
        retriever=vectorstore.as_retriever(),
        memory=memory)
    return conversation_chain

