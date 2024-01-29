import os
from openai import OpenAI
import streamlit as st
import time
from dotenv import load_dotenv


st.title("MARKETING-RESEARCH-CHATBOT")

# Set OpenAi Model
load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat-history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get chat input
if prompt := st.chat_input("How can I help"):
    # Display user message in chat container
    with st.chat_message("👤"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display chat-bot message in chat container
    with st.chat_message("🤖"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state["messages"],
            stream=True,
        ):
            full_response += (response.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})