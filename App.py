import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.docstore.document import Document

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Hostel Assistant",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------
# Sample Hostel Knowledge Base
# -------------------------------------------------
HOSTEL_INFO = """
Single Room: Rs. 5000/month, AC, WiFi, Meals
Double Room: Rs. 3500/month per person, WiFi
Triple Room: Rs. 2800/month per person
Dormitory: Rs. 2000/month

Entry Time: 6 AM – 10 PM
Visitors: 10 AM – 7 PM
No smoking or alcohol

Facilities:
WiFi, Laundry, Study Room, Gym, Hot Water

Meals:
Breakfast: 7:30–9:30
Lunch: 12:30–2:30
Dinner: 7:30–9:30
"""

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -------------------------------------------------
# Load Embeddings (cached)
# -------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# -------------------------------------------------
# Initialize Vector Store
# -------------------------------------------------
def init_vectorstore():
    if st.session_state.vectorstore is None:
        docs = [Document(page_content=HOSTEL_INFO)]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)
        embeddings = load_embeddings()
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

# -------------------------------------------------
# Simple QA (NO API TOKEN)
# -------------------------------------------------
def simple_answer(question):
    retriever = st.session_state.vectorstore.as_retriever(k=2)
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "Sorry, I don’t have that information."
    return docs[0].page_content[:500]

# -------------------------------------------------
# App UI
# -------------------------------------------------
st.title("🏠 Hostel Assistant")

tab1, tab2 = st.tabs(["💬 Chatbot", "📊 Statistics"])

# ---------------- CHATBOT TAB --------------------
with tab1:
    init_vectorstore()
    q = st.text_input("Ask about hostel rooms, rules, or facilities")

    if q:
        st.success(simple_answer(q))

# ---------------- STATISTICS TAB -----------------
with tab2:
    st.subheader("📊 Hostel Occupancy Statistics")

    data = {
        "Month": ["Jan", "Feb", "Mar", "Apr"],
        "Occupancy": [85, 88, 90, 92]
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots()
    ax.plot(df["Month"], df["Occupancy"], marker="o")
    ax.set_ylabel("Occupancy %")
    ax.set_title("Monthly Occupancy")

    st.pyplot(fig)
