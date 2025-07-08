# Imports
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.agents import create_csv_agent
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize ChromaDB
vectorstore = Chroma(collection_name="data", embedding_function=embeddings, persist_directory='./chromadb')

# Initialize LLM model
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-001", temperature=0.4)

# Setting prompt template for RAG
prompt_template = """
    You are an assistant for answering questions about my organization. Use **ONLY** the retrieved context below. 
    Respond in bullet points, format tables with Markdown, and say "I don't know" if the answer isn't in the context.
    
    **Question:** {question}  
    **Context:**  
    {context}  
    **Answer:**  
    - [Use bullet points (•) for answers.]  
    - [Tables must use Markdown formatting]  
    - [Include explanations and disclaimers if available in the context]  
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initializing conversational chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(top_k=3),
    chain_type_kwargs={"prompt": prompt},
)

# RAG Response generation
def get_rag_response(question):
    try:
        full_response = chain.invoke({"query": question})
        return {"output_text": full_response.get("result", "No response generated")}
    except Exception as e:
        return {"output_text": f"Error: {str(e)}"}

# CSV Agent Response generation
def get_csv_response(question, csv_path):
    try:
        # Financial data instruction for CSV agent
        instruction = """When answering questions about financial/gold loan data:
        1. Always use the FULL CSV data loaded as 'df'
        2. Focus on loan amounts, interest rates, gold weight, purity, and customer analysis
        3. Calculate totals, averages, trends, and financial metrics when asked
        4. Provdie answer in (INR) Indian Rupees
        5. Format numbers with proper currency symbols and percentages
        6. Use the python_repl_ast tool with format:
           Action: python_repl_ast
           Action Input: <your code>
        7. Provide final answer with clear financial analysis"""
        
        csv_agent = create_csv_agent(
            llm=llm,
            path=csv_path,
            verbose=True,
            prefix=instruction,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )
        
        response = csv_agent.invoke(question)
        return response.get("output", "Could not retrieve answer from CSV data")
    except Exception as e:
        return f"Error processing CSV request: {str(e)}"

# Custom retriever function to extract metadata (source URLs)
def custom_retriever(question):
    docs = vectorstore.similarity_search(question, k=3)
    urls = []
    for doc in docs:
        if doc.page_content:
            urls.append(doc.metadata['url'])
    return list(set(urls))  # Remove duplicates

# Clear chat history
def clear_chat_history():
    if st.session_state.mode == "rag":
        st.session_state.messages = [{"role": "assistant", "content": "🤖 Ready to help you explore Manappuram's services and information! What would you like to know?"}]
    else:
        st.session_state.messages = [{"role": "assistant", "content": "📊 Hello! I'm your Financial Data Assistant. Ask me anything about the uploaded CSV data!"}]



# Main App Function
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Manappuram Chatbot",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
<style>
    /* Sidebar background - cream color */
    .stSidebar {
        background-color: #FFFAF0 !important;
    }
    
    /* Sidebar text color - black */
    .stSidebar .stMarkdown {
        color: black !important;
    }
    
    /* All sidebar text elements - black */
    .stSidebar * {
        color: black !important;
    }
    
    /* Button colors - crimson */
    .stButton > button {
        background-color: #DC143C !important;
        color: white !important;
    }
    
    .stButton > button:hover {
        background-color: #B91C3C !important;
    }
</style>
""", unsafe_allow_html=True)
    
    # Initialize session state
    if "mode" not in st.session_state:
        st.session_state.mode = "rag"
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "🤖 Ready to help you explore Manappuram's services and information! What would you like to know?"}]
    if "csv_file" not in st.session_state:
        st.session_state.csv_file = None

    # Sidebar with Logo, Description and Help Section
    with st.sidebar:
        st.title("🏦 Manappuram Assistant")
        
        # Mode Selection
        st.markdown("### 🔄 Select Mode")
        mode = st.radio(
            "Choose your assistance type:",
            ["📄 Chatbot", "📊 CSV Analysis"],
            key="mode_selector"
        )
        
        # Update mode in session state
        if mode == "📄 Chatbot":
            if st.session_state.mode != "rag":
                st.session_state.mode = "rag"
                clear_chat_history()
        else:
            if st.session_state.mode != "csv":
                st.session_state.mode = "csv"
                clear_chat_history()

        st.markdown("---")

        # Mode-specific sections
        if st.session_state.mode == "rag":
            # RAG Mode Description
            st.markdown("""
            <div style="text-align: center; font-size: 14px; color: #555;">
                🔍 <strong>RAG-Powered Chatbot</strong><br>
                Built using ChromaDB & Gemini LLM<br>
                Answers based on Manappuram's official resources.
            </div>
            """, unsafe_allow_html=True)
            
            # How to Use Section for RAG
            with st.expander("🆘 How to Use - Chatbot"):
                st.markdown("""
                ### 📋 Quick Start Guide
                - Ask about organizational circulars or services
                - AI retrieves relevant information from documents
                - Source references included for transparency
                - Click 'Clear Chat' to start over
                """)
        
        else:
            # CSV Mode Description
            st.markdown("""
            <div style="text-align: center; font-size: 14px; color: #555;">
                📊 <strong>CSV Analysis Agent</strong><br>
                Built using LangChain & Gemini LLM<br>
                Analyze financial and gold loan data.
            </div>
            """, unsafe_allow_html=True)
            
            # File Upload Section
            st.markdown("### 📁 Upload CSV File")
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type=['csv'],
                help="Upload your financial data CSV file"
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                csv_path = f"temp_{uploaded_file.name}"
                with open(csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.csv_file = csv_path
                st.success("✅ CSV file uploaded successfully!")
            

            # How to Use Section for CSV
            with st.expander("🆘 How to Use - CSV Agent"):
                st.markdown("""
                ### 📊 Sample Questions
                - What's the total loan amount across all customers?
                - Show me customers with gold purity above 22 karat
                - What's the average interest rate by branch?
                - How many active loans do we have?
                - Which branch has the highest loan amounts?
                """)

    # Optional Footer/Disclaimer
    st.markdown("""
    <div style="position: fixed; bottom: 10px; text-align: center; font-size: 12px; color: #555;">
        © 2025 Manappuram Finance Ltd.<br>All responses are AI-generated and for informational purposes only.
    </div>
    """, unsafe_allow_html=True)

    # Clear Chat Button (Centered)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button('🧹 Clear Chat History', on_click=clear_chat_history, use_container_width=True)

    # Main Header
    if st.session_state.mode == "rag":
        st.header("📄 Chatbot Assistant")
        st.markdown("💬 Ask about Manappuram's services and policies")
    else:
        st.header("📊 CSV Agent")
        st.markdown("💬 Analyze your CSV data with AI")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input
    if st.session_state.mode == "rag":
        user_input = st.chat_input("💬 Ask me anything about Manappuram's services...")
    else:
        if st.session_state.csv_file:
            user_input = st.chat_input("📊 Ask me anything about your CSV data...")
        else:
            st.warning("⚠️ Please upload a CSV file or generate sample data to start analysis.")
            user_input = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing your request..."):
                if st.session_state.mode == "rag":
                    # RAG Response
                    response = get_rag_response(user_input)
                    full_response = response.get('output_text', '🤔 Hmm, I couldn\'t find a precise answer.')

                    # Retrieve metadata for RAG
                    metadata = custom_retriever(user_input)
                    if metadata:
                        metadata_str = "\n\n**📋 Sources:**\n" + "\n".join([f"• 🔗 {meta}" for meta in metadata])
                        combined_msg = full_response + metadata_str
                    else:
                        combined_msg = full_response

                else:
                    # CSV Response
                    if st.session_state.csv_file:
                        combined_msg = get_csv_response(user_input, st.session_state.csv_file)
                    else:
                        combined_msg = "⚠️ Please upload a CSV file first."

                # Save and display response
                st.session_state.messages.append({"role": "assistant", "content": combined_msg})
                st.write(combined_msg)

if __name__ == "__main__":
    main()