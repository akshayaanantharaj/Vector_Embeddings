
import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Document folder path
document_folder = "dataset/documents"

documents = []
ids = []
metadatas = []

# Load text documents
for filename in os.listdir(document_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(document_folder, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(text)
        ids.append(filename)

        metadatas.append({
            "title": filename,
            "category": "Artificial Intelligence",
            "source": file_path
        })

# Generate embeddings
embeddings = model.encode(documents)

# Save embeddings
np.save("dataset/embeddings.npy", embeddings)

# Create ChromaDB database
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="ai_documents"
)

# Store documents and embeddings
collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Documents successfully stored in ChromaDB!")
print("Total documents:", len(documents))
print("Embedding shape:", embeddings.shape)


import pandas as pd

# Load sample queries
queries_df = pd.read_csv("dataset/queries.csv")

results = []

# Perform similarity search
for query in queries_df["query"]:

    query_embedding = model.encode([query])

    search_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    for i in range(len(search_results["ids"][0])):

        results.append({
            "query": query,
            "rank": i + 1,
            "document_id": search_results["ids"][0][i],
            "document": search_results["documents"][0][i],
            "distance": search_results["distances"][0][i]
        })

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "dataset/retrieved_results.csv",
    index=False
)

print("\nSimilarity Search Completed!")
print("Total retrieved results:", len(results_df))
print(results_df.head())
