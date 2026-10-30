import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

SYSTEM_PROMPT = (
    "You are a thoughtful and slightly witty human being. Respond to the following question "
    "as if you were talking to a friend. Keep your answer concise, creative, and easy to understand."
)

st.title("💬 Voice Chatbot")

model_option = st.radio("Select AI Model", ["Google Gemini", "OpenAI GPT"])


def text_to_speech(text):
    """Convert text to MP3 bytes with Google Text-to-Speech (works on every OS)."""
    try:
        buffer = BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buffer)
        return buffer.getvalue()
    except Exception as exc:  # network issues etc. should not break the chat
        st.warning(f"Could not generate audio: {exc}")
        return None


# API key: read from .env, otherwise ask in the UI.
api_ready = False
if model_option == "OpenAI GPT":
    api_key = os.getenv("OPENAI_API_KEY") or st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝")
    else:
        client = OpenAI(api_key=api_key)
        api_ready = True
else:
    api_key = os.getenv("GOOGLE_API_KEY") or st.text_input("Google API Key", type="password")
    if not api_key:
        st.info("Please add your Google API key to continue.", icon="🗝")
    else:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
        api_ready = True

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("audio"):
            st.audio(message["audio"], format="audio/mp3")

if api_ready:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if model_option == "OpenAI GPT":
                stream = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                    + [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            else:
                history = [
                    ("human" if m["role"] == "user" else "ai", m["content"])
                    for m in st.session_state.messages
                ]
                # Pass messages as variables so braces in user text are not
                # treated as template placeholders.
                prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMPT), ("placeholder", "{history}")]
                )
                chain = prompt_template | llm
                response = chain.invoke({"history": history}).text
                st.markdown(response)

            with st.spinner("Generating audio..."):
                audio_bytes = text_to_speech(response)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")

        st.session_state.messages.append(
            {"role": "assistant", "content": response, "audio": audio_bytes}
        )
