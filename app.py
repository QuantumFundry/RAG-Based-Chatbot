"""
Flask wrapper for the RAG chatbot.
Single file, hardcoded config — same pattern as your other tools.

Run:  python app.py
Test: open http://localhost:5000 in a browser
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from google import genai
from google.genai import types
import chromadb

# ===================== CONFIG (edit per client) =====================
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash-lite"   # cheap + generous free tier; bump to gemini-2.5-flash if answers need more nuance
BUSINESS_NAME = "Glow Beauty Salon"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "business_kb"
BUSINESS_DOC_FILE = "business_doc.txt"  # plain text file with the client's info, checked into your repo
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
PORT = 5000

client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

app = Flask(__name__)
CORS(app)  # lets the widget call this API from the client's website domain


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_kb_from_file():
    """Rebuilds the knowledge base from business_doc.txt on every startup.
    This matters because free hosting (like Render) wipes local disk storage
    on restart -- so chroma_db disappears, but business_doc.txt (in your repo)
    doesn't. This keeps the bot always working after a restart, with no
    manual re-ingestion needed."""
    if collection.count() > 0:
        return  # already built this run
    with open(BUSINESS_DOC_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    for i, chunk in enumerate(chunk_text(text)):
        embedding = client.models.embed_content(
            model=EMBED_MODEL, contents=chunk,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        ).embeddings[0].values
        collection.add(ids=[f"doc_{i}"], embeddings=[embedding], documents=[chunk])
    print(f"Knowledge base built: {collection.count()} chunks loaded from {BUSINESS_DOC_FILE}")


build_kb_from_file()


def retrieve_and_answer(query, top_k=4):
    q_embed = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    ).embeddings[0].values

    results = collection.query(query_embeddings=[q_embed], n_results=top_k)
    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(docs)

    prompt = f"""You are a helpful customer support assistant for {BUSINESS_NAME}.
Answer ONLY using the context below. If the answer isn't there, say you don't have
that info and suggest contacting the business directly. Keep it short and friendly.

Context:
{context}

Question: {query}
Answer:"""

    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    try:
        answer = retrieve_and_answer(message)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


TEST_PAGE = """
<!doctype html>
<title>{{ business }} - Chat Test</title>
<style>
body{font-family:sans-serif;max-width:480px;margin:40px auto;}
#log{border:1px solid #ddd;padding:12px;height:300px;overflow-y:auto;border-radius:8px;margin-bottom:10px;}
.msg{margin:6px 0;} .user{color:#1a73e8;} .bot{color:#222;}
input{width:75%;padding:8px;} button{padding:8px 14px;}
</style>
<h3>{{ business }} — Test Chat</h3>
<div id="log"></div>
<input id="inp" placeholder="Ask something..." onkeydown="if(event.key==='Enter')send()">
<button onclick="send()">Send</button>
<script>
async function send(){
  const inp = document.getElementById('inp');
  const log = document.getElementById('log');
  const msg = inp.value.trim();
  if(!msg) return;
  log.innerHTML += `<div class="msg user"><b>You:</b> ${msg}</div>`;
  inp.value = '';
  const res = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message: msg})});
  const data = await res.json();
  log.innerHTML += `<div class="msg bot"><b>Bot:</b> ${data.answer || data.error}</div>`;
  log.scrollTop = log.scrollHeight;
}
</script>
"""


@app.route("/")
def index():
    return render_template_string(TEST_PAGE, business=BUSINESS_NAME)


if __name__ == "__main__":
    print(f"Running on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
