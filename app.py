# Imports
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Initialize Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize ChromaDB
vectorstore = Chroma(collection_name="data", embedding_function=embeddings, persist_directory='./chromadb')

# Initialize LLM model
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-001", temperature=0.4)

# Setting prompt template
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

# Response generation
def get_response(question):
    full_response = chain.invoke({"query": question})
    return {
        "output_text": full_response.get("result")
    }

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
    st.session_state.messages = [{"role": "assistant", "content": "🤖 Ready to help you explore Manappuram's services and information! What would you like to know?"}]

# Main App Function
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Manappuram RAG Chatbot",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar with Logo, Description and Help Section
    with st.sidebar:

        # Chatbot Description
        st.markdown("""
        <div style="text-align: center; font-size: 14px; color: #555;">
            🔍 <strong>RAG-Powered Chatbot</strong><br>
            Built using ChromaDB & Gemini LLM<br>
            Answers based on information scraped from Manappuram's official resources.
        </div>
        """, unsafe_allow_html=True)

        # Divider Line
        st.markdown("---")

        # How to Use Section
        with st.expander("🆘 How to Use"):
            st.markdown("""
            ### 📋 Quick Start Guide
            - Type your question about organizational circulars or services
            - Our AI will retrieve the most relevant information
            - Source references are included for transparency
            - Click 'Clear Chat' to start over
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

    # Initialize chat messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "🤖 Ready to help you explore Manappuram's services and information! What would you like to know?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input
    if prompt := st.chat_input("💬 Ask me anything about Manappuram's services and information..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching through documents..."):
                response = get_response(prompt)
                full_response = response.get('output_text', '🤔 Hmm, I couldn\'t find a precise answer.')

                # Retrieve metadata
                metadata = custom_retriever(prompt)
                if metadata:
                    metadata_str = "\n\n**📋 Source :**\n" + "\n".join([f"• 🔗 {meta}" for meta in metadata])
                    combined_msg = full_response + metadata_str
                else:
                    combined_msg = full_response

                # Save and display response
                message = {"role": "assistant", "content": combined_msg}
                st.session_state.messages.append(message)
                st.write(combined_msg)

if __name__ == "__main__":
    main()