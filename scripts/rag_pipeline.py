import pandas as pd
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from transformers.utils import logging

# Suppress verbose logging
logging.set_verbosity_error()

# Set working directory
os.chdir('C:/Users/Skyline/Intelligent-Complaint-Analysis-for-Financial-Services')
print(f'Current working directory: {os.getcwd()}')

# Load vector store and metadata
try:
    index = faiss.read_index('vector_store/complaint_embeddings.faiss')
    with open('vector_store/chunk_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print('Vector store and metadata loaded successfully.')
except FileNotFoundError:
    print('Error: Vector store or metadata not found.')
    raise

# Load chunks from metadata
chunks = []
for meta in metadata:
    try:
        chunks.append(meta.get('text', ''))
    except KeyError:
        print('Warning: Missing text in metadata. Using empty string.')
        chunks.append('')
print(f'Loaded {len(chunks)} chunks.')

# Initialize embedding model
cache_dir = 'C:/Users/Skyline/Intelligent-Complaint-Analysis-for-Financial-Services/model_cache'
os.makedirs(cache_dir, exist_ok=True)
try:
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
    print('Embedding model loaded successfully.')
except Exception as e:
    print(f'Error loading embedding model: {e}')
    raise

# Initialize LLM
try:
    llm_pipeline = pipeline(
        'text-generation',
        model='distilgpt2',
        max_length=150,
        truncation=True,
        device=-1,  # CPU
        num_return_sequences=1
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    print('LLM loaded successfully.')
except Exception as e:
    print(f'Error loading LLM: {e}')
    raise

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=['context', 'question'],
    template="""
You are a financial analyst assistant for CrediTrust. Answer questions about customer complaints using only the provided context. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:
"""
)

# RAG pipeline function
def rag_pipeline(question, top_k=5):
    try:
        # Embed question
        question_embedding = model.encode([question], show_progress_bar=False)[0]
        
        # Perform similarity search
        distances, indices = index.search(np.array([question_embedding], dtype=np.float32), top_k)
        
        # Retrieve chunks and metadata
        retrieved_chunks = []
        for idx in indices[0]:
            if idx < len(chunks):
                chunk = chunks[idx]
                meta = metadata[idx]
                retrieved_chunks.append(f"Complaint ID: {meta['complaint_id']}, Product: {meta['product']}, Text: {chunk[:100]}...")
            else:
                retrieved_chunks.append("Invalid chunk index.")
        
        # Combine context
        context = "\n\n".join(retrieved_chunks)
        
        # Generate prompt
        prompt = prompt_template.format(context=context, question=question)
        
        # Generate response
        response = llm(prompt)
        
        # Extract answer
        answer_start = response.find("Answer:") + len("Answer:") if "Answer:" in response else 0
        answer = response[answer_start:].strip()
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks[:2]  # Top 2 for evaluation
        }
    except Exception as e:
        return {'question': question, 'answer': f'Error: {str(e)}', 'retrieved_chunks': []}

# Test questions
questions = [
    "What are common issues with credit card complaints?",
    "How do consumers describe problems with money transfers?",
    "What complaints involve unauthorized transactions?",
    "Are there issues with customer service in checking accounts?",
    "What are typical payday loan complaints?"
]

# Run and evaluate
results = []
for question in questions:
    result = rag_pipeline(question)
    results.append(result)

# Save evaluation results
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/evaluation_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print('Evaluation results saved to notebooks/evaluation_results.pkl')