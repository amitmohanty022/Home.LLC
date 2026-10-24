# Voice Chatbot (OpenAI or Gemini) with Text-to-Speech

A Streamlit chatbot that can talk to either OpenAI or Google Gemini and reads every answer aloud.

## 🔗 Live demo

Not hosted yet. Deploy it for free on Streamlit Community Cloud: click the badge (or go to https://share.streamlit.io → **Create app**) and use repository `amitmohanty022/Home.LLC`, branch `main`, main file `chatbotvoice.py`.

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=amitmohanty022/Home.LLC&branch=main&mainModule=chatbotvoice.py)

No secrets are needed: users type their own OpenAI or Google API key into the app.

## Features

- Switch between **OpenAI** and **Google Gemini** with a radio button
- API key entered in the app (never stored in the repo)
- Every reply is converted to MP3 speech with gTTS and can be replayed
- Full chat history kept for the session

## Tech stack

Python, Streamlit, OpenAI Python SDK, LangChain (`langchain-google-genai`), gTTS

## Run locally

```bash
git clone https://github.com/amitmohanty022/Home.LLC.git
cd Home.LLC
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Keys are typed into the app. Optional overrides via environment variables:

```env
OPENAI_MODEL=gpt-6-luna         # default
GEMINI_MODEL=gemini-3.8-flash   # default
```

Then start the app:

```bash
streamlit run chatbotvoice.py
```

## Author

**Amit Kumar Mohanty** · [GitHub](https://github.com/amitmohanty022) · [LinkedIn](https://www.linkedin.com/in/amitkrmohanty/)
