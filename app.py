import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# First Streamlit command
st.set_page_config(page_title="Insurance Policy Chatbot", page_icon="💬")

# Load the model, FAISS index, and chunks
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks
#def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)                                                  
    return model, index, chunks

def search_answer(query, model, index, chunks, top_k=5):
    """
    Searches for the most relevant text chunk based on user query.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    if indices[0][0] != -1:
        return chunks[indices[0][0]]
    else:
        return "Sorry, I couldn't find an answer. Let me connect you to a human agent."

# Load resources
model, index, chunks = load_resources()

# Streamlit UI
st.title("💬 Insurance Policy Information Chatbot")
st.write("Ask me anything about our insurance policies!")

# User input
user_query = st.text_input("Type your question here...")

if user_query:
    with st.spinner("Thinking... 🤔"):
        answer = search_answer(user_query, model, index, chunks)
    st.success(answer)
