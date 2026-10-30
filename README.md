# Home.LLC – Voice Chatbot

A Streamlit chatbot that answers like a witty friend and reads every answer out loud. You can switch between **Google Gemini** and **OpenAI GPT** with one click.

## How to run (step by step)

You need **Python 3.10 or newer** ([download](https://www.python.org/downloads/)) and an internet connection (used for the AI and the voice).

**1. Download the project**

```bash
git clone https://github.com/amitmohanty022/Home.LLC.git
cd Home.LLC
```

**2. Create a virtual environment and activate it**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install the requirements**

```bash
pip install -r requirements.txt
```

**4. Add your API key (you only need one)**

- Gemini key (free): https://aistudio.google.com/apikey
- OpenAI key: https://platform.openai.com/account/api-keys
- Copy `.env.example` to a new file named `.env`
  - Windows: `copy .env.example .env`
  - macOS / Linux: `cp .env.example .env`
- Open `.env` and paste your key after `GOOGLE_API_KEY=` and/or `OPENAI_API_KEY=`

If you skip this step, the app asks for the key on the page instead.

**5. Start the app**

```bash
streamlit run chatbotvoice.py
```

Your browser opens at http://localhost:8501.

**6. Use it**

1. Pick **Google Gemini** or **OpenAI GPT**.
2. Type a message in the box at the bottom.
3. Read the reply, or press play on the audio player to hear it.

## Settings

| Variable in `.env` | Default | What it does |
|---|---|---|
| `GOOGLE_API_KEY` | (empty) | Gemini key |
| `OPENAI_API_KEY` | (empty) | OpenAI key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model |
| `OPENAI_MODEL` | `gpt-6-luna` | OpenAI model |

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate the virtual environment (step 2) first. |
| "Could not generate audio" | The voice uses Google Text-to-Speech online. Check your internet connection. |
| Model not found | Change `GEMINI_MODEL` or `OPENAI_MODEL` in `.env`. |
