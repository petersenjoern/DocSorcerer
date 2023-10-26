"""Frontend for DocScorcerer backend."""

import random
from typing import Dict, Generator, Union

import requests
import streamlit as st

BASE_DOC_SORCERER_URL = "http://localhost:8000"
PAGE_TITLE: str = "Welcome to the DocScorcerer"
PAGE_ICON: str = "🤖"

WAITING_MESSAGES = [
    "As our AI genie rubs its virtual lamp for your response, here’s a quick joke: Why did the AI break up with the computer? It just couldn’t find the right byte!",
    "While our AI wordsmith crafts your response, here’s a little AI-themed fun fact: Did you know the first words spoken by a computer were ‘Hello, world!’?",
    "While our generative AI creates your answer, here’s a brain teaser: What do you call an AI that loves to dance? An al-gore-ithm!",
    "As our AI muse composes your response, here’s a quirky thought: If AI had a favorite song, it would be ‘Binary Solo’!",
    "While you wait for our AI magician to work its magic, here’s a playful tidbit: AI stands for ‘Artificially Ingenious’ in our book!",
    "While the AI chef prepares your response, here’s some AI food for thought: If AI made sandwiches, they’d be byte-sized!",
    "As our AI paints a canvas of words for you, here’s a creative idea: What if AI and humans formed a band? It would be called ‘The Algorithmics’!",
    "While the AI gears up for your response, here’s a virtual high-five for your patience. 🤖✋",
]


def _answer_question_streaming(
    session: requests.Session, question: str
) -> Generator[str, str, None]:
    """Query backend API for generating an answer to the question in a streamed fashion"""

    response = session.get(
        url=f"{BASE_DOC_SORCERER_URL}/answer-question?question={question}", stream=True
    )
    response.raise_for_status()
    for line in response.iter_content(chunk_size=1024):
        if line:
            yield line.decode("utf-8")


def _get_evidence_for_answer(
    session: requests.Session, question: str
) -> Dict[str, Union[str, float]]:
    """Query backend API to obtain evidence documents for answer"""

    response = session.get(
        url=f"{BASE_DOC_SORCERER_URL}/answer-question-evidence?question={question}"
    )
    response.raise_for_status()
    return response.json()


# Page configuration
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
st.title(PAGE_TITLE)
st.header(
    "Remember, I am just a Retrieval-Augmented Generation system with a vector database :sunglasses:"
)

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
        latest_state_message = st.session_state.messages[-1]["content"]

        evidences_for_answer = _get_evidence_for_answer(
            session=session, question=latest_state_message
        )
        st.json(evidences_for_answer, expanded=False)

        llm_answer_placeholder = st.empty()
        with st.spinner(random.choice(WAITING_MESSAGES)):
            full_response = []
            for response in _answer_question_streaming(
                session=session, question=latest_state_message
            ):
                full_response.append(response)
                result = "".join(full_response).strip()
                llm_answer_placeholder.markdown(result + "▌")
            llm_answer_placeholder.markdown(result)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result})
