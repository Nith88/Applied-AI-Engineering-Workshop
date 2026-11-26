import os
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY in .env")

# 1. Load selectable text from a PDF.
reader = PdfReader("data/AI_Ethics_Policy.pdf")
raw_text = "\\n".join(page.extract_text() or "" for page in reader.pages)
if not raw_text.strip():
    raise ValueError("No text extracted; scanned PDFs need OCR.")

# 2. Split into 300-character chunks with 50-character overlap.
def split_text(text, chunk_size=300, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        part = text[start:start + chunk_size].strip()
        if part:
            chunks.append(part)
        start += chunk_size - overlap
    return chunks

chunks = split_text(raw_text, 300, 50)
print("Chunks:", len(chunks))

# 3. Embed and store chunks.
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./rag_chroma")
collection = client.get_or_create_collection("policy_chunks")
collection.upsert(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embedder.encode(chunks).tolist()
)

# 4. Retrieve, 5. augment, 6. generate.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite", google_api_key=api_key, temperature=0
)

def answer_question(question, k=3):
    found = collection.query(
        query_embeddings=embedder.encode([question]).tolist(),
        n_results=k
    )
    evidence = found["documents"][0]
    context = "\\n\\n".join(evidence)
    prompt = f"""Answer using ONLY the context below.
If the answer is absent, say you cannot find it in the document.
Do not invent policy details.

CONTEXT:
{context}

QUESTION: {question}"""
    answer = llm.invoke(prompt).content
    return answer, evidence

while True:
    q = input("Ask about the PDF (or exit): ")
    if q.lower() in {"exit", "quit"}:
        break
    answer, evidence = answer_question(q)
    print("\\nANSWER:", answer)
    print("\\nRETRIEVED EVIDENCE:")
    for item in evidence:
        print("-", item)
