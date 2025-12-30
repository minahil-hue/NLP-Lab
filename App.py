import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import os

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

Additional Services:
- Medical first aid available 24/7
- Parking facility for bikes and cars
- Water purifier on each floor
- Power backup generator
- Monthly maintenance and pest control

Payment Options:
- Cash payments accepted at reception
- Online transfer to hostel bank account
- Monthly, quarterly, and annual payment plans available
- 5% discount on annual advance payment

Check-in/Check-out:
- Check-in time: 12:00 PM onwards
- Check-out time: 11:00 AM
- Early check-in/late check-out subject to availability

Room Amenities:
- Bed with mattress and pillow
- Study table and chair
- Wardrobe for clothes
- Window with curtains
- Fan and light fixtures
- Individual locker for valuables
"""

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

@st.cache_resource
def load_embeddings():
    """Load embeddings model (cached)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def initialize_vectorstore():
    """Initialize the vector store with sample hostel information"""
    if st.session_state.vectorstore is None:
        with st.spinner("Initializing knowledge base... (This may take a minute on first run)"):
            try:
                # Load embeddings
                if st.session_state.embeddings is None:
                    st.session_state.embeddings = load_embeddings()
                
                # Create document from sample text
                documents = [Document(page_content=SAMPLE_HOSTEL_INFO)]
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create vector store
                vectorstore = FAISS.from_documents(chunks, st.session_state.embeddings)
                st.session_state.vectorstore = vectorstore
                st.success("✅ Knowledge base initialized!")
            except Exception as e:
                st.error(f"Error initializing knowledge base: {str(e)}")

def get_chatbot_response(question, hf_token):
    """Get response from RAG chatbot using Hugging Face model"""
    try:
        # Retrieve relevant documents
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Combine context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt template
        prompt_template = """You are a helpful hostel assistant. Use the following context to answer the question. 
If you cannot find the answer in the context, say "I don't have that information in my knowledge base."

Context:
{context}

Question: {question}

Answer: """
        
        prompt = prompt_template.format(context=context, question=question)
        
        # Initialize Hugging Face LLM
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            huggingfacehub_api_token=hf_token,
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        
        # Get response
        response = llm(prompt)
        return response.strip()
        
    except Exception as e:
        return f"Error: {str(e)}\n\nPlease check your Hugging Face token or try again."

def simple_qa_without_llm(question):
    """Simple keyword-based Q&A when no API token is provided"""
    question_lower = question.lower()
    
    # Retrieve relevant documents
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
    relevant_docs = retriever.get_relevant_documents(question)
    
    # Extract relevant context
    if relevant_docs:
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Return context with a note
        return f"**Based on hostel information:**\n\n{context[:800]}...\n\n*Note: For better responses, please provide a Hugging Face API token in the sidebar.*"
    else:
        return "I couldn't find relevant information. Please try rephrasing your question."

def generate_sample_data():
    """Generate sample hostel statistics data"""
    data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
        'Single_Rooms': [8, 9, 7, 10, 9, 8, 10, 9],
        'Double_Rooms': [15, 14, 16, 15, 17, 16, 18, 17],
        'Triple_Rooms': [10, 12, 11, 10, 13, 12, 14, 13],
        'Dormitory': [25, 28, 26, 30, 29, 27, 31, 30],
        'Occupancy_Rate': [85, 88, 82, 92, 90, 87, 94, 91],
        'Revenue_PKR': [450000, 480000, 440000, 510000, 495000, 470000, 530000, 505000]
    }
    return pd.DataFrame(data)

def display_statistics(df):
    """Display statistics and visualizations"""
    st.subheader("📈 Hostel Occupancy Statistics")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_occupancy = df[['Single_Rooms', 'Double_Rooms', 'Triple_Rooms', 'Dormitory']].sum().sum()
        st.metric("Total Occupants", total_occupancy)
    
    with col2:
        avg_occupancy_rate = df['Occupancy_Rate'].mean()
        st.metric("Avg Occupancy Rate", f"{avg_occupancy_rate:.1f}%")
    
    with col3:
        max_month = df.loc[df['Occupancy_Rate'].idxmax(), 'Month']
        st.metric("Peak Month", max_month)
    
    with col4:
        if 'Revenue_PKR' in df.columns:
            total_revenue = df['Revenue_PKR'].sum()
            st.metric("Total Revenue", f"Rs. {total_revenue:,.0f}")
    
    # Visualizations
    st.subheader("📊 Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Room Distribution", "Occupancy Trend", "Revenue Analysis"])
    
    with tab1:
        # Bar chart for room types
        fig, ax = plt.subplots(figsize=(10, 6))
        room_types = ['Single_Rooms', 'Double_Rooms', 'Triple_Rooms', 'Dormitory']
        avg_occupancy = df[room_types].mean()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        bars = ax.bar(range(len(avg_occupancy)), avg_occupancy.values, color=colors)
        ax.set_xlabel("Room Type", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Occupants", fontsize=12, fontweight='bold')
        ax.set_title("Average Occupancy by Room Type", fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(avg_occupancy)))
        ax.set_xticklabels(['Single', 'Double', 'Triple', 'Dormitory'], rotation=0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        # Line chart for occupancy rate trend
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Month'], df['Occupancy_Rate'], marker='o', linewidth=2.5, 
                color='#1f77b4', markersize=8, markerfacecolor='#ff7f0e')
        ax.fill_between(df['Month'], df['Occupancy_Rate'], alpha=0.3, color='#1f77b4')
        ax.set_xlabel("Month", fontsize=12, fontweight='bold')
        ax.set_ylabel("Occupancy Rate (%)", fontsize=12, fontweight='bold')
        ax.set_title("Monthly Occupancy Rate Trend", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(75, 100)
        
        # Add value labels
        for i, (month, rate) in enumerate(zip(df['Month'], df['Occupancy_Rate'])):
            ax.text(i, rate + 1, f'{rate}%', ha='center', fontsize=9)
        
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        if 'Revenue_PKR' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Revenue bar chart
            ax1.bar(df['Month'], df['Revenue_PKR'], color='#2ecc71', alpha=0.7)
            ax1.set_xlabel("Month", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Revenue (PKR)", fontsize=12, fontweight='bold')
            ax1.set_title("Monthly Revenue", fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3)
            
            # Room type contribution pie chart
            room_contribution = df[room_types].sum()
            colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            ax2.pie(room_contribution, labels=['Single', 'Double', 'Triple', 'Dormitory'], 
                   autopct='%1.1f%%', startangle=90, colors=colors_pie)
            ax2.set_title("Room Type Contribution", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Revenue data not available in the uploaded file.")
    
    # Summary Statistics Table
    st.subheader("📋 Summary Statistics")
    summary_stats = df.describe().round(2)
    st.dataframe(summary_stats, use_container_width=True)
    
    # Detailed Data Table
    st.subheader("📄 Detailed Data")
    st.dataframe(df, use_container_width=True)

# Main App
def main():
    st.title("🏠 Hostel Assistant Application")
    st.markdown("*Your one-stop solution for hostel information and statistics - Powered by AI*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.markdown("### 🤗 Hugging Face Settings")
        hf_token = st.text_input(
            "Hugging Face API Token (Optional)", 
            type="password", 
            help="Get your free token from https://huggingface.co/settings/tokens"
        )
        
        if not hf_token:
            st.info("💡 **No token?** The app will work with limited functionality using keyword matching.")
        
        st.markdown("---")
        
        st.markdown("### 📖 How to Get Token")
        with st.expander("Click for instructions"):
            st.markdown("""
            1. Go to [Hugging Face](https://huggingface.co/)
            2. Sign up for a free account
            3. Go to Settings → Access Tokens
            4. Create a new token (Read role is sufficient)
            5. Copy and paste it above
            
            **Model Used:** google/flan-t5-large (Free)
            """)
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This app uses free Hugging Face models for AI-powered responses and provides hostel statistics visualization.")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chatbot", "📊 Statistics"])
    
    with tab1:
        st.header("Hostel Information Chatbot")
        st.markdown("Ask me anything about hostel rooms, facilities, rules, bookings, or pricing!")
        
        # Initialize vectorstore
        initialize_vectorstore()
        
        if st.session_state.vectorstore is not None:
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
                        if hf_token:
                            response = get_chatbot_response(user_question, hf_token)
                        else:
                            response = simple_qa_without_llm(user_question)
                        
                        st.markdown(response)
                
                # Add assistant message to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.error("Failed to initialize knowledge base. Please refresh the page.")
    
    with tab2:
        st.header("Hostel Statistics Dashboard")
        
        # File upload option
        st.markdown("### 📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Upload hostel data (CSV/Excel)", 
            type=['csv', 'xlsx'],
            help="Upload a file with columns like Month, Single_Rooms, Double_Rooms, etc."
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success("✅ File uploaded successfully!")
                display_statistics(df)
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please ensure your file has the correct format.")
        else:
            st.info("📊 No file uploaded. Displaying sample data below.")
            df = generate_sample_data()
            display_statistics(df)

if __name__ == "__main__":
    main()
