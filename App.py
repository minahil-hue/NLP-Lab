import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Hostel Assistant", page_icon="🏠", layout="wide")

# Hugging Face Token (embedded)
HF_TOKEN = "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"

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

Hostel Capacity:
- Total Rooms: 50
- Single Rooms: 10
- Double Rooms: 20
- Triple Rooms: 15
- Dormitories: 5

Monthly Charges Breakdown:
- Room rent (includes electricity up to 100 units)
- WiFi charges included
- Meal charges: Breakfast Rs. 1500, Lunch Rs. 2000, Dinner Rs. 2000 per month
- Laundry: Rs. 500 per month (optional)

Special Features:
- Weekly housekeeping service
- Complimentary newspaper delivery
- Indoor games room with carrom, chess, and table tennis
- Terrace garden for relaxation
- Book exchange library
"""

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

@st.cache_resource
def load_embeddings():
    """Load embeddings model (cached)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_llm():
    """Load Hugging Face LLM (cached)"""
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

def initialize_rag_system():
    """Initialize the complete RAG system"""
    if st.session_state.vectorstore is None or st.session_state.qa_chain is None:
        with st.spinner("🔄 Initializing AI system... (First time may take 1-2 minutes)"):
            try:
                # Load embeddings
                embeddings = load_embeddings()
                
                # Create document from sample text
                documents = [Document(page_content=SAMPLE_HOSTEL_INFO)]
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=50,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create vector store
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.vectorstore = vectorstore
                
                # Load LLM
                llm = load_llm()
                
                # Create custom prompt template
                prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't have that information, don't try to make up an answer.
Keep your answer concise and relevant to the hostel information.

Context: {context}

Question: {question}

Helpful Answer:"""
                
                PROMPT = PromptTemplate(
                    template=prompt_template, 
                    input_variables=["context", "question"]
                )
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                st.session_state.qa_chain = qa_chain
                st.success("✅ AI system ready!")
                
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
                return False
    return True

def get_chatbot_response(question):
    """Get response from RAG chatbot"""
    try:
        response = st.session_state.qa_chain.run(question)
        return response.strip()
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."

def generate_sample_data():
    """Generate sample hostel statistics data"""
    data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
        'Single_Rooms': [8, 9, 7, 10, 9, 8, 10, 9, 8, 10],
        'Double_Rooms': [15, 14, 16, 15, 17, 16, 18, 17, 16, 19],
        'Triple_Rooms': [10, 12, 11, 10, 13, 12, 14, 13, 12, 14],
        'Dormitory': [25, 28, 26, 30, 29, 27, 31, 30, 28, 32],
        'Occupancy_Rate': [85, 88, 82, 92, 90, 87, 94, 91, 88, 96],
        'Revenue_PKR': [450000, 480000, 440000, 510000, 495000, 470000, 530000, 505000, 485000, 545000]
    }
    return pd.DataFrame(data)

def display_statistics(df):
    """Display statistics and visualizations"""
    st.subheader("📈 Hostel Occupancy Statistics")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_occupancy = df[['Single_Rooms', 'Double_Rooms', 'Triple_Rooms', 'Dormitory']].iloc[-1].sum()
        st.metric("Current Occupants", int(total_occupancy))
    
    with col2:
        avg_occupancy_rate = df['Occupancy_Rate'].mean()
        last_rate = df['Occupancy_Rate'].iloc[-1]
        prev_rate = df['Occupancy_Rate'].iloc[-2] if len(df) > 1 else last_rate
        delta = last_rate - prev_rate
        st.metric("Avg Occupancy Rate", f"{avg_occupancy_rate:.1f}%", f"{delta:+.1f}%")
    
    with col3:
        max_month = df.loc[df['Occupancy_Rate'].idxmax(), 'Month']
        max_rate = df['Occupancy_Rate'].max()
        st.metric("Peak Month", f"{max_month} ({max_rate}%)")
    
    with col4:
        if 'Revenue_PKR' in df.columns:
            total_revenue = df['Revenue_PKR'].sum()
            st.metric("Total Revenue", f"Rs. {total_revenue/1000000:.2f}M")
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Visual Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 Room Distribution", "📈 Occupancy Trend", "💰 Revenue", "📋 Data Table"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for average room occupancy
            fig, ax = plt.subplots(figsize=(8, 6))
            room_types = ['Single_Rooms', 'Double_Rooms', 'Triple_Rooms', 'Dormitory']
            avg_occupancy = df[room_types].mean()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            bars = ax.bar(range(len(avg_occupancy)), avg_occupancy.values, color=colors, edgecolor='black', linewidth=1.2)
            
            ax.set_xlabel("Room Type", fontsize=12, fontweight='bold')
            ax.set_ylabel("Average Occupants", fontsize=12, fontweight='bold')
            ax.set_title("Average Occupancy by Room Type", fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(range(len(avg_occupancy)))
            ax.set_xticklabels(['Single', 'Double', 'Triple', 'Dormitory'])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Pie chart for room type distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            room_totals = df[room_types].sum()
            
            ax.pie(room_totals, labels=['Single', 'Double', 'Triple', 'Dormitory'], 
                   autopct='%1.1f%%', startangle=90, colors=colors,
                   explode=(0.05, 0.05, 0.05, 0.05), shadow=True)
            ax.set_title("Total Room Type Distribution", fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with tab2:
        # Occupancy rate trend
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df['Month'], df['Occupancy_Rate'], marker='o', linewidth=3, 
                color='#2E86AB', markersize=10, markerfacecolor='#A23B72', 
                markeredgecolor='white', markeredgewidth=2, label='Occupancy Rate')
        ax.fill_between(df['Month'], df['Occupancy_Rate'], alpha=0.2, color='#2E86AB')
        
        ax.set_xlabel("Month", fontsize=13, fontweight='bold')
        ax.set_ylabel("Occupancy Rate (%)", fontsize=13, fontweight='bold')
        ax.set_title("Monthly Occupancy Rate Trend", fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=11)
        
        # Add value labels
        for i, (month, rate) in enumerate(zip(df['Month'], df['Occupancy_Rate'])):
            ax.text(i, rate + 1.5, f'{rate}%', ha='center', fontsize=9, fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Additional trend analysis
        st.markdown("### 📊 Trend Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Highest Rate:** {df['Occupancy_Rate'].max()}% in {df.loc[df['Occupancy_Rate'].idxmax(), 'Month']}")
        with col2:
            st.info(f"**Lowest Rate:** {df['Occupancy_Rate'].min()}% in {df.loc[df['Occupancy_Rate'].idxmin(), 'Month']}")
        with col3:
            st.info(f"**Average Rate:** {df['Occupancy_Rate'].mean():.1f}%")
    
    with tab3:
        if 'Revenue_PKR' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Revenue bar chart
            bars = ax1.bar(df['Month'], df['Revenue_PKR']/1000, color='#27AE60', alpha=0.8, edgecolor='black')
            ax1.set_xlabel("Month", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Revenue (K PKR)", fontsize=12, fontweight='bold')
            ax1.set_title("Monthly Revenue Performance", fontsize=14, fontweight='bold', pad=20)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}K',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Revenue vs Occupancy correlation
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(df['Month'], df['Revenue_PKR']/1000, marker='o', 
                            linewidth=2.5, color='#27AE60', label='Revenue', markersize=8)
            line2 = ax2_twin.plot(df['Month'], df['Occupancy_Rate'], marker='s', 
                                 linewidth=2.5, color='#E74C3C', label='Occupancy %', markersize=8)
            
            ax2.set_xlabel("Month", fontsize=12, fontweight='bold')
            ax2.set_ylabel("Revenue (K PKR)", fontsize=12, fontweight='bold', color='#27AE60')
            ax2_twin.set_ylabel("Occupancy Rate (%)", fontsize=12, fontweight='bold', color='#E74C3C')
            ax2.set_title("Revenue vs Occupancy Correlation", fontsize=14, fontweight='bold', pad=20)
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Revenue insights
            st.markdown("### 💡 Revenue Insights")
            col1, col2, col3 = st.columns(3)
            with col1:
                total_rev = df['Revenue_PKR'].sum()
                st.success(f"**Total Revenue:** Rs. {total_rev:,}")
            with col2:
                avg_rev = df['Revenue_PKR'].mean()
                st.success(f"**Average Monthly:** Rs. {avg_rev:,.0f}")
            with col3:
                growth = ((df['Revenue_PKR'].iloc[-1] - df['Revenue_PKR'].iloc[0]) / df['Revenue_PKR'].iloc[0]) * 100
                st.success(f"**Growth Rate:** {growth:+.1f}%")
        else:
            st.info("💡 Revenue data not available. Upload a file with 'Revenue_PKR' column to see revenue analysis.")
    
    with tab4:
        st.markdown("### 📋 Complete Data Table")
        
        # Summary statistics
        st.markdown("**Summary Statistics:**")
        summary = df.describe().round(2)
        st.dataframe(summary, use_container_width=True)
        
        st.markdown("---")
        
        # Full data
        st.markdown("**Detailed Monthly Data:**")
        st.dataframe(df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="hostel_statistics.csv",
            mime="text/csv"
        )

# Main App
def main():
    # Header
    st.title("🏠 Hostel Assistant Application")
    st.markdown("*AI-Powered Hostel Management & Information System*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/home.png", width=80)
        st.header("🎯 Navigation")
        
        st.markdown("### Features")
        st.markdown("""
        - 🤖 **AI Chatbot**: Get instant answers
        - 📊 **Statistics**: Visual analytics
        - 📈 **Trends**: Track occupancy
        - 💰 **Revenue**: Financial insights
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔧 System Info")
        st.info("""
        **AI Model:** FLAN-T5-Large  
        **Embeddings:** MiniLM-L6-v2  
        **Vector DB:** FAISS
        """)
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.caption("Powered by Hugging Face 🤗")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 AI Chatbot", "📊 Statistics Dashboard"])
    
    with tab1:
        st.header("🤖 Hostel Information Chatbot")
        st.markdown("Ask me anything about rooms, facilities, pricing, rules, or bookings!")
        
        # Initialize RAG system
        if initialize_rag_system():
            
            # Sample questions
            with st.expander("💡 Try these sample questions"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("What are the room types?", use_container_width=True):
                        st.session_state.sample_question = "What are the room types available?"
                    if st.button("What are the meal timings?", use_container_width=True):
                        st.session_state.sample_question = "What are the meal timings?"
                with col2:
                    if st.button("What facilities are available?", use_container_width=True):
                        st.session_state.sample_question = "What facilities are available?"
                    if st.button("How do I book a room?", use_container_width=True):
                        st.session_state.sample_question = "How do I book a room?"
            
            st.markdown("---")
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Handle sample question
            if 'sample_question' in st.session_state:
                user_question = st.session_state.sample_question
                del st.session_state.sample_question
                
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        response = get_chatbot_response(user_question)
                        st.markdown(response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
            
            # User input
            user_question = st.chat_input("Type your question here...")
            
            if user_question:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                # Get bot response
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        response = get_chatbot_response(user_question)
                        st.markdown(response)
                
                # Add assistant message
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with tab2:
        st.header("📊 Hostel Statistics Dashboard")
        
        # File upload
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your hostel data (CSV/Excel)", 
            type=['csv', 'xlsx'],
            help="File should contain columns: Month, Single_Rooms, Double_Rooms, Triple_Rooms, Dormitory, Occupancy_Rate"
        )
        
        st.markdown("---")
        
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
                st.info("💡 Please ensure your file has columns: Month, Single_Rooms, Double_Rooms, Triple_Rooms, Dormitory, Occupancy_Rate")
        else:
            st.info("📊 No file uploaded. Displaying sample hostel data.")
            df = generate_sample_data()
            display_statistics(df)

if __name__ == "__main__":
    main()
