import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers.utils import logging

# Suppress verbose logging
logging.set_verbosity_error()

# Set working directory
os.chdir('C:/Users/Skyline/Intelligent-Complaint-Analysis-for-Financial-Services')
print(f'Current working directory: {os.getcwd()}')

# Load cleaned dataset
try:
    df = pd.read_csv('data/complaints_cleaned.csv')
    print('Cleaned dataset loaded successfully.')
except FileNotFoundError:
    print('Error: complaints_cleaned.csv not found.')
    raise

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=['\n\n', '\n', '.', ' ', '']
)

# Chunk narratives
chunks = []
metadata = []
for idx, row in df.iterrows():
    narrative = row['Consumer complaint narrative']
    if pd.notnull(narrative):
        split_texts = text_splitter.split_text(narrative)
        for i, chunk in enumerate(split_texts):
            chunks.append(chunk)
            metadata.append({
                'complaint_id': str(row['Complaint ID']),
                'product': row['Product'],
                'chunk_index': i
            })

print(f'Total chunks created: {len(chunks)}')

# Initialize embedding model with explicit cache directory
cache_dir = 'C:/Users/Skyline/Intelligent-Complaint-Analysis-for-Financial-Services/model_cache'
os.makedirs(cache_dir, exist_ok=True)
try:
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
    print('Embedding model loaded successfully.')
except Exception as e:
    print(f'Error loading model: {e}')
    raise

# Generate embeddings with smaller batch size
try:
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=16)
except MemoryError:
    print('MemoryError: Reducing batch size to 8 and retrying.')
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=8)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

# Save vector store and metadata
os.makedirs('vector_store', exist_ok=True)
faiss.write_index(index, 'vector_store/complaint_embeddings.faiss')
with open('vector_store/chunk_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print('Vector store and metadata saved to vector_store/')