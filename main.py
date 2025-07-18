import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv
import os
import tempfile
import re
from datetime import datetime

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenRouter LLM
llm = ChatOpenAI(
    model_name="openai/gpt-3.5-turbo",
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key
)

# Calculator tool function
def calculator(expression: str) -> str:
    try:
        if not re.match(r'^[\d\s+\-*/().]+$', expression):
            return "Invalid expression. Use numbers and operators (+, -, *, /, (, ))."
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Define tool list
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Evaluates mathematical expressions like '2 + 2' or '5 * 3'."
    )
]

# Setup memory and agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    handle_parsing_errors=True
)

# Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Custom CSS with Tailwind CDN
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .chat-container { max-height: 400px; overflow-y: auto; padding: 10px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: #f9fafb; }
        .chat-message-user { background-color: #3b82f6; color: white; padding: 10px; border-radius: 8px; margin: 5px 0; }
        .chat-message-assistant { background-color: #e5e7eb; color: black; padding: 10px; border-radius: 8px; margin: 5px 0; }
        .timestamp { font-size: 0.75rem; color: #6b7280; }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-6 rounded-lg shadow-lg mb-6"><h1 class="text-3xl font-bold">📄 PDF Chat Agent</h1><p class="text-sm mt-2">Upload a PDF and ask questions or use the calculator!</p></div>', unsafe_allow_html=True)

# Sidebar for PDF upload and controls
with st.sidebar:
    st.markdown('<h2 class="text-xl font-semibold text-gray-800">Controls</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", help="Upload a PDF to start chatting about its content.")
    if st.button(" Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

# PDF processing
if uploaded_file:
    with st.spinner("Processing PDF..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

            retriever = st.session_state.vectorstore.as_retriever()
            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory
            )
            st.success("PDF processed successfully!")
            os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f" Error processing PDF: {str(e)}")

# Chat input and display
st.markdown('<h3 class="text-lg font-medium text-gray-700 mb-4">Chat</h3>', unsafe_allow_html=True)
user_input = st.text_input("Ask about the PDF or use the calculator (e.g., 'What is 5 + 3?' or 'Summarize the PDF')", placeholder="Type your question here...")
if user_input:
    with st.spinner("Generating response..."):
        st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": datetime.now().strftime("%H:%M:%S")})
        try:
            if st.session_state.conversation and "calculate" not in user_input.lower():
                response = st.session_state.conversation({"question": user_input})
                answer = response["answer"]
            else:
                answer = agent.run(user_input)
            st.session_state.messages.append({"role": "assistant", "content": answer, "timestamp": datetime.now().strftime("%H:%M:%S")})
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}", "timestamp": datetime.now().strftime("%H:%M:%S")})

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message-user"><strong>You:</strong> {message["content"]}<br><span class="timestamp">{message["timestamp"]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message-assistant"><strong>Assistant:</strong> {message["content"]}<br><span class="timestamp">{message["timestamp"]}</span></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
