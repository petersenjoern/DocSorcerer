"""Frontend for DocScorcerer backend."""

import streamlit as st
import requests


PAGE_TITLE: str = "Welcome to the DocScorcerer"
PAGE_ICON: str = "🤖"


def doc_scorcerer_ask_question(session: requests.Session, question: str) -> str:
    """Query backend API"""

    response = session.get(url=f"http://localhost:8000/ask?question={question}")
    response.raise_for_status()
    return response.text



# Page configuration
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.title(PAGE_TITLE)
st.header('Remember, I am just a Retrieval-Augmented Generation system with a vector database :sunglasses:')

session = requests.Session()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if question := st.chat_input("Type your question here, and let the magic happen!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        latest_state_message = st.session_state.messages[-1]["content"]
        full_response = doc_scorcerer_ask_question(session=session, question=latest_state_message)
        message_placeholder.markdown(full_response + "▌")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})