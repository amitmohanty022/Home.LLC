import streamlit as st
from openai import OpenAI
import pyttsx3
import threading
import tempfile
import os
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Show title and description.
st.title("💬 Voice Chatbot")

# Add a radio button to select the AI model
model_option = st.radio("Select AI Model", ["OpenAI GPT", "Google Gemini"])

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        temp_filename = tmp_file.name
    
    engine.save_to_file(text, temp_filename)
    engine.runAndWait()
    
    # Read the file and return the bytes
    with open(temp_filename, 'rb') as f:
        audio_bytes = f.read()
    
    # Clean up the temporary file
    os.unlink(temp_filename)
    
    return audio_bytes

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
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
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
                model="gpt-4o-mini",
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
            # Prepare the chat history for LangChain
            chat_history = []
            for m in st.session_state.messages:
                if m["role"] == "user":
                    chat_history.append({"type": "human", "content": m["content"]})
                elif m["role"] == "assistant":
                    chat_history.append({"type": "ai", "content": m["content"]})
            
            # Create a prompt template with a thoughtful and witty instruction
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a thoughtful and slightly witty human being. Respond to the following question as if you were talking to a friend. Keep your answer concise, creative, and easy to understand."),
                *[(msg["type"], msg["content"]) for msg in chat_history],
                ("human", prompt)
            ])
            
            # Generate the response
            chain = prompt_template | llm
            response = chain.invoke({}).content
            
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