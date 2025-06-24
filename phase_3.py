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
/* ---------- BASE COLORS ---------- */
body, .main, .stApp {
    background: #FFFFFF;  /* Pure White */
    color: #111111;       /* Deep Black */
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- HEADER ---------- */
.typewriter-container {
    display: flex;
    justify-content: center;
    margin: 40px 0 20px 0;
}

.typewriter-text {
    overflow: hidden;
    border-right: 2px solid #007ACC;
    white-space: nowrap;
    letter-spacing: 0.06em;
    animation: typing 2.5s steps(40, end), blink-caret 0.75s step-end infinite;
    font-size: 26px;
    font-weight: 800;
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

/* ---------- MAIN TITLE ---------- */
h1 {
    color: #111111;
    font-weight: 900;
    border-bottom: 3px solid #007ACC;
    display: inline-block;
    padding-bottom: 6px;
    margin-bottom: 30px;
}

/* ---------- CHAT BUBBLES (and CHILD TEXT!) ---------- */
.stChatMessage {
    background: #F2F4F5 !important;  /* Light Grey Bubble */
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

/* FORCE TEXT COLOR INSIDE BUBBLES */
.stChatMessage * {
    color: #111111 !important;   /* Bulletproof: force all children to black */
}

/* ---------- INPUT & PLACEHOLDER ---------- */
input, textarea {
    border-radius: 8px !important;
    color: #111111 !important;
    background: #FFFFFF !important;
}

input::placeholder, textarea::placeholder {
    color: #666666 !important;
}

input:focus, textarea:focus {
    border: 2px solid #007ACC !important;
    box-shadow: 0 0 6px #007ACC;
    outline: none !important;
    transition: 0.2s ease;
}

/* ---------- BUTTON ---------- */
button {
    color: #111111 !important;
    background: #E9ECEF !important;
    border-radius: 6px !important;
}

button:hover {
    background: #DDE1E4 !important;
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 600px) {
    .typewriter-text {
        font-size: 20px;
    }
    h1 {
        font-size: 24px;
    }
}
</style>
""", unsafe_allow_html=True)

# --- TYPEWRITER HEADER ---
st.markdown("""
<div class="typewriter-container">
    <div class="typewriter-text">PDC RAG CHATBOT BY MUZAMMIL YASIR</div>
</div>
""", unsafe_allow_html=True)

# --- TITLE ---
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
