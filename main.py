import os
import json
from dotenv import load_dotenv


import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import groq


load_dotenv()


working_dir = os.path.dirname(os.path.abspath(__file__))




def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory, 
                      embedding_function=embeddings)
    return vectorstore


def chat_chain(vectorstore):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    retreiver = vectorstore.as_retriever()
    memory = ConversationBufferMemory(llm = llm ,
                                      output_key="answer",
                                      memory_key="chat_history", 
                                      return_messages=True)
    

    chain = ConversationalRetrievalChain.from_llm(llm=llm, 
                                                  retriever= retreiver,
                                                  chain_type="stuff",
                                                  memory=memory,
                                                  verbose=True,
                                                  return_source_documents=True)
    return chain


st.set_page_config(page_title="Chat with your documents", 
                   page_icon="📁", 
                   layout="wide")



st.title("📁 Chat with your documents")

GROQ_API_KEY = st.text_input("Please enter your GROQ API Key here: ", value="",type="password")
client = groq.Groq(api_key=GROQ_API_KEY)



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = chat_chain(st.session_state.vectorstore)



for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})























































