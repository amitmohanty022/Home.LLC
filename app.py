import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from gtts import gTTS
import tempfile
import base64

# Load environment variables
load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Create memory with a buffer size of 5
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_memory_size=5  # Limit memory to the last 5 exchanges
)

# Create prompt template
template = """
Respond to the following question as if you were a thoughtful and slightly witty human being. 
Keep your answer concise, creative, and easy to understand, as if you were talking to a friend.

Previous conversation:
{chat_history}

Question: {question}
"""

prompt = PromptTemplate(template=template, input_variables=["chat_history", "question"])

# Create an LLMChain for the chain
chain = LLMChain(llm=llm, prompt=prompt)

def text_to_speech(text):
    """Convert text to speech using gTTS and return audio file path."""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
        tts.save(f"{tmp_file.name}.mp3")
        return f"{tmp_file.name}.mp3"

def play_audio(file_path):
    """Play audio file in Streamlit."""
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

# Streamlit app layout
st.title("🎤 AI Voice Bot 🤖")  # Fun title with emojis
st.write("Ask me anything and I'll respond with a voice!")

# User input
user_input = st.text_input("Your question:", placeholder="Type your question here...")

if st.button("Submit"):
    if user_input:
        # Get response from Gemini
        response = chain.run(chat_history=memory.chat_memory, question=user_input)
        st.write("**Response:**", response)
        
        # Convert to speech and play
        audio_file_path = text_to_speech(response)
        play_audio(audio_file_path)
        
        # Update memory
        memory.save_context({"input": user_input}, {"output": response})
    else:
        st.warning("Please enter a question.")

# Clear conversation button
if st.button("Clear Conversation"):
    memory.clear()  # Clear the memory
    st.success("Conversation cleared!")

# Display conversation history
if memory.chat_memory:
    st.subheader("🗣️ Conversation History:")
    for entry in memory.chat_memory:
        st.write(f"**User:** {entry[0]}")
        st.write(f"**Bot:** {entry[1]}") 