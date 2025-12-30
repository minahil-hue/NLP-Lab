
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document
import os
from io import StringIO

# Set page config
st.set_page_config(page_title="Hostel Assistant", page_icon="🏠", layout="wide")

# Sample hostel data (knowledge base)
SAMPLE_HOSTEL_INFO = """
HOSTEL INFORMATION GUIDE

Room Types and Availability:
- Single Room: Rs. 5000/month, AC available, includes WiFi and meals
- Double Room: Rs. 3500/month per person, shared bathroom, WiFi included
- Triple Room: Rs. 2800/month per person, common bathroom, WiFi included
- Dormitory: Rs. 2000/month, shared space for 6-8 people, basic amenities

Hostel Rules:
1. Entry time: 6 AM to 10 PM. Late entry requires prior permission.
2. Visitors allowed only in common areas between 10 AM to 7 PM.
3. No smoking or alcohol consumption on premises.
4. Maintain cleanliness in rooms and common areas.
5. Noise levels must be kept low after 10 PM.

Facilities:
- 24/7 WiFi connectivity with high-speed internet
- Laundry service available twice a week
- Common kitchen for cooking (vegetarian only)
- Study room open 24/7
- Gym facilities available from 6 AM to 9 PM
- Common TV room with streaming services
- Hot water available 24/7

Booking Process:
1. Visit the hostel or contact us via phone/email
2. Fill out the registration form with ID proof
3. Pay security deposit (Rs. 5000, refundable)
4. Pay first month's rent in advance
5. Receive room keys and hostel ID card

Contact Information:
Phone: +91-9876543210
Email: info@hostelhome.com
Address: 123 University Road, Rawalpindi, Punjab

Meal Timings:
Breakfast: 7:30 AM - 9:30 AM
Lunch: 12:30 PM - 2:30 PM
Dinner: 7:30 PM - 9:30 PM

Security:
- CCTV surveillance in common areas
- Security guard on duty 24/7
- Biometric entry system
- Emergency contact numbers displayed throughout the hostel
"""

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

def initialize_vectorstore():
    """Initialize the vector store with sample hostel information"""
    if st.session_state.vectorstore is None:
        with st.spinner("Initializing knowledge base..."):
            # Create document from sample text
            documents = [Document(page_content=SAMPLE_HOSTEL_INFO)]
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            st.success("Knowledge base initialized!")

def get_chatbot_response(question, api_key):
    """Get response from RAG chatbot"""
    try:
        # Initialize LLM
        llm = OpenAI(temperature=0.7, openai_api_key=api_key)
        
        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=False
        )
        
        # Get response
        response = qa_chain.run(question)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def generate_sample_data():
    """Generate sample hostel statistics data"""
    data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Single_Rooms': [8, 9, 7, 10, 9, 8],
        'Double_Rooms': [15, 14, 16, 15, 17, 16],
        'Triple_Rooms': [10, 12, 11, 10, 13, 12],
        'Dormitory': [25, 28, 26, 30, 29, 27],
        'Occupancy_Rate': [85, 88, 82, 92, 90, 87]
    }
    return pd.DataFrame(data)

def display_statistics(df):
    """Display statistics and visualizations"""
    st.subheader("📈 Hostel Occupancy Statistics")
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_occupancy = df[['Single_Rooms', 'Double_Rooms', 'Triple_Rooms', 'Dormitory']].sum().sum()
        st.metric("Total Occupants", total_occupancy)
    
    with col2:
        avg_occupancy_rate = df['Occupancy_Rate'].mean()
        st.metric("Avg Occupancy Rate", f"{avg_occupancy_rate:.1f}%")
    
    with col3:
        max_month = df.loc[df['Occupancy_Rate'].idxmax(), 'Month']
        st.metric("Peak Month", max_month)
    
    # Visualizations
    st.subheader("Visualizations")
    
    tab1, tab2 = st.tabs(["Room Type Distribution", "Occupancy Trend"])
    
    with tab1:
        # Bar chart for room types
        fig, ax = plt.subplots(figsize=(10, 6))
        room_types = ['Single_Rooms', 'Double_Rooms', 'Triple_Rooms', 'Dormitory']
        avg_occupancy = df[room_types].mean()
        
        sns.barplot(x=avg_occupancy.index, y=avg_occupancy.values, palette="viridis", ax=ax)
        ax.set_xlabel("Room Type")
        ax.set_ylabel("Average Occupants")
        ax.set_title("Average Occupancy by Room Type")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab2:
        # Line chart for occupancy rate trend
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Month'], df['Occupancy_Rate'], marker='o', linewidth=2, color='#1f77b4')
        ax.set_xlabel("Month")
        ax.set_ylabel("Occupancy Rate (%)")
        ax.set_title("Monthly Occupancy Rate Trend")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Data table
    st.subheader("Detailed Data")
    st.dataframe(df, use_container_width=True)

# Main App
def main():
    st.title("🏠 Hostel Assistant Application")
    st.markdown("Your one-stop solution for hostel information and statistics")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This app provides hostel information through an AI chatbot and displays occupancy statistics.")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chatbot", "📊 Statistics"])
    
    with tab1:
        st.header("Hostel Information Chatbot")
        st.markdown("Ask me anything about hostel rooms, facilities, rules, or bookings!")
        
        # Initialize vectorstore
        initialize_vectorstore()
        
        # Chat interface
        if api_key:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # User input
            user_question = st.chat_input("Type your question here...")
            
            if user_question:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                # Get bot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_chatbot_response(user_question, api_key)
                        st.markdown(response)
                
                # Add assistant message to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to start chatting.")
    
    with tab2:
        st.header("Hostel Statistics Dashboard")
        
        # File upload option
        uploaded_file = st.file_uploader("Upload your hostel data (CSV/Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success("File uploaded successfully!")
                display_statistics(df)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.info("📁 No file uploaded. Displaying sample data.")
            df = generate_sample_data()
            display_statistics(df)

if __name__ == "__main__":
    main()
