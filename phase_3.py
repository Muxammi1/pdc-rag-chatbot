import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import warnings
import logging
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

# Load .env for API key
load_dotenv()

# LangChain & PDF tools
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Streamlit Page Config
st.set_page_config(page_title="Neura Thread", layout="centered")

# --- Viewport for Mobile Responsiveness ---
st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)
# --- Custom CSS for Dark UI and Animation ---
st.markdown("""
    <style>
    /* Global Background and Text */
    body {
        background-color: #FFFFFF;
        color: #111111;
        font-family: 'Segoe UI', sans-serif;
    }

    .main {
        background-color: #FFFFFF;
    }

    /* Typewriter Header */
    .typewriter-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 30px;
    }

    .typewriter-text {
        overflow: hidden;
        border-right: .15em solid #007ACC;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: .08em;
        animation: 
            typing 3s steps(40, end),
            blink-caret .75s step-end infinite;
        font-size: 22px;
        font-weight: bold;
        color: #111111;
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #007ACC; }
    }

    /* Title Styling with Subtle Pulse */
    .stApp h1 {
        animation: pulse 5s ease-in-out infinite;
        color: #111111;
        font-weight: 800;
    }

    @keyframes pulse {
        0% { color: #111111; }
        50% { color: #007ACC; }
        100% { color: #111111; }
    }

    /* Input Focus Styling */
    input:focus {
        border: 2px solid #007ACC !important;
        box-shadow: 0 0 8px #007ACC;
        transition: all 0.3s ease-in-out;
    }

    /* Chat Message Bubbles (Light Grey) */
    .stChatMessage {
        background-color: #F5F5F5 !important;
        border-radius: 8px;
        padding: 12px;
        color: #111111;
    }

    /* Responsive adjustments */
    @media (max-width: 600px) {
        .typewriter-text {
            font-size: 16px;
            letter-spacing: 0.05em;
        }

        .stApp h1 {
            font-size: 24px;
        }

        .main, .block-container {
            padding-left: 12px;
            padding-right: 12px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Animated Typewriter Header ---
st.markdown("""
    <div class="typewriter-container">
        <div class="typewriter-text">PDC RAG CHATBOT BY MUZAMMIL YASIR</div>
    </div>
""", unsafe_allow_html=True)

st.title("Ask Neura Thread")

# --- Chat History Setup ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# --- Load PDF and Create Vector Store ---
@st.cache_resource
def get_vectorstore():
    pdf_name = "./reflexion.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

# --- Chat Input ---
prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are very smart at everything, you always give the best, 
        the most accurate and most precise answers. Answer the following Question: {user_prompt}.
        Start the answer directly. No small talk please.
    """)

    model = "llama3-8b-8192"

    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model
    )

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document")

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain({"query": prompt})
        response = result["result"]

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Export Chat History with Timestamps ---
if st.session_state.get("messages"):
    def format_chat():
        chat_lines = []
        for msg in st.session_state.messages:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            role = msg['role'].capitalize()
            content = msg['content']
            chat_lines.append(f"[{timestamp}] {role}: {content}")
        return "\n\n".join(chat_lines)

    st.download_button(
        label="📥 Download Chat History with Timestamps",
        data=format_chat(),
        file_name="chat_history.txt",
        mime="text/plain"
    )
