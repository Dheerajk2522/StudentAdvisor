import streamlit as st
import time
import os
import asyncio
from pg2 import (
    search_database, process_query_with_openai
)

# Page configuration
st.set_page_config(
    page_title="Student Advisor Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS with improved colors
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f8f9fa;
        color:black;
    }
    
    /* Header styling */
    h1 {
        color: #4a2c8f !important;
        font-weight: 600 !important;
    }
    h3{
        color:black;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.8rem; 
        margin-bottom: 1.2rem; 
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .chat-message.user {
        background-color: #e8f0fe;
        border-left: 5px solid #4285f4;
    }
    .chat-message.bot {
        background-color: #f0f4f8;
        border-left: 5px solid #4a2c8f;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        margin-right: 1rem;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message .content {
        flex-grow: 1;
    }
    .chat-message .content p {
        margin: 0;
        line-height: 1.5;
        color:black;
    }
    
    /* Input area */
    .stTextInput > div > div > input {
        border: 2px solid #e6e6e6;
        border-radius: 10px;
        padding: 12px 15px;
        background-color: #f8f9fa;
        color: #333;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4a2c8f;
        box-shadow: 0 0 0 2px rgba(74, 44, 143, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4a2c8f;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #6940c9;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        color: #4a2c8f;
    }
    .streamlit-expanderContent {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Footer */
    footer {
        color: #6c757d;
    }
    
    /* Loading/thinking state */
    .thinking {
        color: #6c757d;
        font-style: italic;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    /* App logo/icon styling */
    .app-logo {
        font-size: 2.5rem;
        background-color: #4a2c8f;
        color: white;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
  
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to add message to chat history
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Header with logo
st.markdown("<div class='app-logo'>🎓</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Student Advisor Chatbot</h1>", unsafe_allow_html=True)

# Information section
with st.expander("ℹ️ About this app", expanded=False):
    st.markdown("""
    This Student Advisor Chatbot can help with:
    - Course recommendations for specific students
    - Prerequisites and study plans
    - Course scheduling and availability
    - Program requirements and specializations
    """)

# Display chat history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🎓"
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message['role']}">
            <div class="avatar">{avatar}</div>
            <div class="content">
                <p>{message['content']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# User input
with st.container():
    user_input = st.text_input(
        "Ask your question here:",
        key="user_input",
        placeholder="e.g., Which courses should John Smith take next semester?"
    )

    # Center the send button and make it a bit wider
    col1, col2, col3 = st.columns([5, 2, 5])
    with col2:
        submit_button = st.button("Send", use_container_width=True)

# Process the user query
if submit_button and user_input:
    # Add user message to chat history
    add_message("user", user_input)
    
    # Create a placeholder for the "thinking" message
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown('<div class="thinking">🔍 Searching database and generating response...</div>', unsafe_allow_html=True)
    
    try:
        # Search database
        start_time = time.time()
        results = search_database(user_input)
        search_time = time.time() - start_time
        
        if not results:
            bot_response = "I couldn't find any relevant information in the database. Try asking a different question related to student courses, programs, or enrollment."
        else:
            # Generate counts for search results
            source_counts = {}
            for r in results:
                source = r['source_table']
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Generate response
            start_time = time.time()
            bot_response = process_query_with_openai(user_input, results)
            response_time = time.time() - start_time
            
            # Add search metadata to the end of the response
            sources_str = ", ".join([f"{count} {source}" for source, count in source_counts.items()])
            search_info = f"\n\n*Found {len(results)} relevant pieces of information across {len(source_counts)} sources in {search_time:.2f} seconds.*"
            bot_response += search_info
    
    except Exception as e:
        bot_response = f"❌ Error: {str(e)}\n\nPlease try a different question or check your database connection."
    
    # Remove the "thinking" message
    thinking_placeholder.empty()
    
    # Add bot response to chat history
    add_message("bot", bot_response)
    
    # Force a rerun to update the UI with the new messages
    # st.rerun()
    asyncio.run(st.run())

# Add a footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6c757d;'>Student Advisor Chatbot</p>", unsafe_allow_html=True) 