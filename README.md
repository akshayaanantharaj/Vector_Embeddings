# Week 11 – Vector Embeddings with ChromaDB

## Project Title
Creating and Storing Vector Embeddings using ChromaDB

## Objective
- Generate vector embeddings using Sentence Transformers.
- Store embeddings in ChromaDB.
- Perform semantic similarity searches.
- Retrieve the Top-5 relevant documents.
- Prepare a knowledge base for RAG applications.

## Domain
Artificial Intelligence

## Technologies Used
- Python
- Sentence Transformers
- ChromaDB
- NumPy
- Pandas
- Jupyter Notebook

## Embedding Model
all-MiniLM-L6-v2

## Project Structure
```text
Vector_Embeddings/
├── dataset/
│   ├── documents/
│   ├── queries.csv
│   ├── retrieved_results.csv
│   └── embeddings.npy
├── chroma_db/
├── notebook/
│   └── Week11_Vector_Embeddings.ipynb
├── vector_store.py
├── README.md
└── requirements.txt
```

## Implementation
1. Created 10 AI-related text documents.
2. Generated vector embeddings.
3. Created a ChromaDB collection.
4. Stored documents with metadata.
5. Performed similarity searches using 10 queries.
6. Retrieved Top-5 relevant documents.
7. Saved retrieval results in CSV.
8. Compared semantic search with keyword-based search.

## Conclusion
Successfully generated and stored vector embeddings using Sentence Transformers and ChromaDB. Semantic search was performed to retrieve relevant documents. This project prepares a knowledge base for future RAG applications.