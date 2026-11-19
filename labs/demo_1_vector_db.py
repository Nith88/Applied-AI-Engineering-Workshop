from sentence_transformers import SentenceTransformer, util
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "A king rules a kingdom.", "A queen governs a country.",
    "An apple is a sweet fruit.", "Machine learning helps computers learn from data."
]
vectors = model.encode(sentences)
print("Vector dimensions:", len(vectors[0]))
print("King / queen:", round(util.cos_sim(vectors[0], vectors[1]).item(), 3))
print("King / apple:", round(util.cos_sim(vectors[0], vectors[2]).item(), 3))

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("teaching_knowledge")
documents = [
    "Machine learning enables systems to learn patterns from data.",
    "A database stores and organizes information for retrieval.",
    "Embeddings represent text as numerical vectors.",
    "ChromaDB supports similarity search over vector embeddings.",
    "RAG retrieves relevant passages before generating an answer.",
    "Python is popular for AI application development.",
    "Neural networks use connected processing units.",
    "A vector database indexes embeddings for nearest-neighbour search.",
    "A lesson plan defines objectives, activities, and assessment.",
    "Cybersecurity protects systems, networks, and information."
]
collection.upsert(
    ids=[f"doc-{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=model.encode(documents).tolist()
)
question = "How can software learn from examples?"
found = collection.query(
    query_embeddings=model.encode([question]).tolist(), n_results=3
)
for text, distance in zip(found["documents"][0], found["distances"][0]):
    print(f"- {text} (distance={distance:.3f})")
