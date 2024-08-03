import os
import json
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import faiss
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("API token for Google is not set. Please check your environment variables.")

def extract_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

embedding_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

def embed_texts(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def similarity_search(query, index, text_chunks, top_k=5):
    query_embedding = embed_texts([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    distances, indices = index.search(query_embedding, top_k)
    top_results = [text_chunks[idx] for idx in indices[0]]
    return top_results

st.set_page_config(layout="wide")

if 'history' not in st.session_state:
    st.session_state['history'] = []
    st.session_state['context'] = ""
    st.session_state['file_loaded'] = False
    st.session_state['selected_celebrity'] = ""

celebrity_options = {
    "Elon Musk": {"image": "img/elon_musk.jpg", "file": "info/elon.txt"},
    "MS Dhoni": {"image": "img/ms_dhoni.jpg", "file": "info/dhoni.txt"},
    "Mark Zuckerberg": {"image": "img/mark_zuckerberg.jpg", "file": "info/zuckerberg.txt"},
    "Bill Gates": {"image": "img/bill_gates.jpg", "file": "info/gates.txt"}
}

selected_celebrity = st.sidebar.selectbox("Choose a celebrity", list(celebrity_options.keys()))

if selected_celebrity != st.session_state['selected_celebrity']:
    st.session_state['selected_celebrity'] = selected_celebrity
    st.session_state['history'] = []
    st.session_state['context'] = ""

celebrity_image = celebrity_options[selected_celebrity]["image"]
celebrity_file = celebrity_options[selected_celebrity]["file"]

st.sidebar.image(celebrity_image, width=150)

text_content = extract_text_from_file(celebrity_file)
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
text_chunks = text_splitter.split_text(text_content)

embedded_texts = embed_texts(text_chunks)
embedded_texts = embedded_texts / np.linalg.norm(embedded_texts, axis=1, keepdims=True)

dimension = embedded_texts.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embedded_texts)

# repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

st.session_state['file_loaded'] = True

history_folder = 'history'
os.makedirs(history_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
history_file = os.path.join(history_folder, f'history_{timestamp}.json')

history_data = {
    "file_name": celebrity_file,
    "load_time": timestamp,
    "chat_history": st.session_state['history']
}

with open(history_file, 'w') as f:
    json.dump(history_data, f, indent=4)

if st.session_state['file_loaded']:
    st.title(f"Chat with {selected_celebrity}")
    st.write("Ask anything!")

    query = st.text_input("Enter your question:")
    if query:
        st.session_state.history.append({"role": "user", "content": query})
        top_results = similarity_search(query, faiss_index, text_chunks)

        st.session_state.context += f"User: {query}\n"
        context = "\n\n".join(top_results)
        print(context)
        
        prompt_template = PromptTemplate.from_template(
            f"Answer the following question in {selected_celebrity}'s style: \n\n"
            "Chat History:\n{chat_history}\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)

        result = chain.invoke(input={"chat_history": st.session_state.context, "context": context, "question": query})
        st.session_state.context += f"Bot: {result['text']}\n"
        st.session_state.history.append({"role": "bot", "content": result['text']})

        chat_history = st.session_state.history[:]
        for chat in reversed(chat_history):
            if chat['role'] == 'user':
                st.markdown(f"<div style='text-align: right; background-color: #d3d3d3; padding: 8px; border-radius: 10px; margin: 5px; color: black;'>{chat['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; background-color: #add8e6; padding: 8px; border-radius: 10px; margin: 5px; color: black;'>{chat['content']}</div>", unsafe_allow_html=True)
                st.image(celebrity_image, width=50, caption=selected_celebrity)
