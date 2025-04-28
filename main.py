import os
import json

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


working_dir = os.path.dirname(os.path.abspath(__file__))

conffig_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = conffig_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY






















