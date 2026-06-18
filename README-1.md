# Business RAG Chatbot

A custom AI chatbot for local businesses that answers customer questions using
the business's own information (hours, services, pricing, policies) — instead
of generic, made-up answers. Built using a RAG (Retrieval-Augmented Generation)
pipeline so every answer is grounded in real business data.

## How it works

1. The business's info (FAQs, pricing, services) is split into small chunks.
2. Each chunk is converted into a vector embedding using the Gemini API and
   stored in a ChromaDB vector database.
3. When a customer asks a question, the bot embeds the question, finds the
   most relevant chunks from the business's data, and sends them to Gemini
   along with the question to generate an accurate, grounded answer.
4. If the answer isn't in the business's data, the bot says so instead of
   guessing — no hallucinated answers.

## Tech stack

- **Gemini API** (`gemini-embedding-001` for embeddings, `gemini-2.5-flash-lite`
  for generating answers) — free tier
- **ChromaDB** — lightweight vector database, no server setup needed
- **Flask** — serves the chatbot as a web API
- **Vanilla HTML/JS widget** — embeddable chat bubble for any website

## Project structure

```
app.py                 -> Flask backend (the chatbot API)
business_doc.txt       -> The business's info that the bot answers from
widget_snippet.html    -> Embeddable chat widget for the client's website
requirements.txt       -> Python dependencies
```

## Running it locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` to test the chat in your browser.

## Deploying

Deployed for free on [Render](https://render.com):
- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`

The knowledge base is rebuilt automatically from `business_doc.txt` on every
startup, so it survives free-tier server restarts without needing a persistent
database.

## Customizing for a new business

1. Replace the contents of `business_doc.txt` with the new business's actual
   info (hours, services, pricing, contact details).
2. Update `BUSINESS_NAME` in `app.py`.
3. Redeploy.

## Live demo

[Add your Render URL here once deployed]

## Author

Built by Rahat — AI automation tools for local businesses.
