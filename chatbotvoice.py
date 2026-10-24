import os
from io import BytesIO

import streamlit as st
from gtts import gTTS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SYSTEM_PROMPT = (
    "You are a thoughtful and slightly witty human being. Respond to the following question as if you "
    "were talking to a friend. Keep your answer concise, creative, and easy to understand."
)

# Show title and description.
st.title("💬 Voice Chatbot")

# Add a radio button to select the AI model
model_option = st.radio("Select AI Model", ["OpenAI GPT", "Google Gemini"])

# Function to convert text to speech.
# gTTS returns real MP3 audio and works on hosted platforms such as Streamlit Community Cloud,
# where pyttsx3 has no speech engine available.
def text_to_speech(text):
    audio = BytesIO()
    gTTS(text=text, lang="en").write_to_fp(audio)
    return audio.getvalue()

# API key input based on selected model
if model_option == "OpenAI GPT":
    # Ask user for their OpenAI API key
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
        api_ready = False
    else:
        # Create an OpenAI client
        client = OpenAI(api_key=api_key)
        api_ready = True
else:
    # Ask user for their Google API key
    api_key = st.text_input("Google API Key", type="password")
    if not api_key:
        st.info("Please add your Google API key to continue.", icon="🗝️")
        api_ready = False
    else:
        # Create a Google Gemini client via LangChain
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key)
        api_ready = True

# Create a session state variable to store the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Add audio playback for assistant messages
        if message["role"] == "assistant" and "audio" in message:
            st.audio(message["audio"], format="audio/mp3")

# Create a chat input field
if api_ready:
    if prompt := st.chat_input("What is up?"):
        # Store and display the current prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if model_option == "OpenAI GPT":
            # Generate a response using the OpenAI API
            stream = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )

            # Stream the response
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
                
                # Convert the response to speech
                with st.spinner("Generating audio..."):
                    audio_bytes = text_to_speech(response)
                    st.audio(audio_bytes, format="audio/mp3")
            
            # Store both the text response and audio
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "audio": audio_bytes
            })
        else:
            # Build the conversation for LangChain. The current prompt is already the last
            # user message in session state, so it is not added a second time.
            history = [SystemMessage(content=SYSTEM_PROMPT)]
            for m in st.session_state.messages:
                if m["role"] == "user":
                    history.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    history.append(AIMessage(content=m["content"]))

            # Generate the response
            response = llm.invoke(history).text

            # Display the response
            with st.chat_message("assistant"):
                st.markdown(response)
                
                # Convert the response to speech
                with st.spinner("Generating audio..."):
                    audio_bytes = text_to_speech(response)
                    st.audio(audio_bytes, format="audio/mp3")
            
            # Store both the text response and audio
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "audio": audio_bytes
            })