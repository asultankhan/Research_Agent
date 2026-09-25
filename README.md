# AI Research Agent

Single CrewAI agent with Crossref and web search, using Groq and Streamlit.

## Run locally (Python 3.12)

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, enter your Groq API key, then run:

```bash
streamlit run app.py
```

## Deploy

Push `app.py`, `research_tools.py`, `requirements.txt`, `.gitignore`, and `README.md` to GitHub. On https://share.streamlit.io create an app from the repository, set entrypoint `app.py`, choose Python 3.12, and put `GROQ_API_KEY = "your-real-key"` in Advanced settings > Secrets. Never commit `.streamlit/secrets.toml`.

This prototype searches metadata and web snippets. It does not review full papers or create a systematic review. Verify every source and conclusion before academic use.
