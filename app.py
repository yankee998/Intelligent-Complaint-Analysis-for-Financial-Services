import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import gradio as gr
from transformers.utils import logging

# Suppress verbose logging
logging.set_verbosity_error()

# Set working directory
os.chdir('C:/Users/Skyline/Intelligent-Complaint-Analysis-for-Financial-Services')
print(f'Current working directory: {os.getcwd()}')

# Load vector store and metadata
index = faiss.read_index('vector_store/complaint_embeddings.faiss')
with open('vector_store/chunk_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
chunks = [meta.get('text', '') for meta in metadata]

# Initialize embedding model
cache_dir = 'model_cache'
os.makedirs(cache_dir, exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)

# Initialize LLM
llm_pipeline = pipeline(
    'text-generation',
    model='distilgpt2',
    max_length=300,
    truncation=True,
    device=-1
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=['context', 'question'],
    template="""
You are a financial analyst assistant for CrediTrust. Provide a concise answer to the question about consumer complaints based solely on the provided context. If the context lacks sufficient information, say 'I don’t have enough information to answer.'

Context: {context}

Question: {question}

Answer:
"""
)

# State management
history = []

# RAG pipeline function with history
def rag_pipeline(question, show_history=True):
    global history
    # Embed question
    question_embedding = model.encode([question])[0]
    
    # Perform similarity search
    distances, indices = index.search(np.array([question_embedding], dtype=np.float32), 5)
    
    # Retrieve chunks
    retrieved_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):
            chunk = chunks[idx]
            meta = metadata[idx]
            retrieved_chunks.append(f"Complaint ID: {meta['complaint_id']}, Product: {meta['product']}, Text: {chunk[:100]}...")
    
    # Limit to top 2 chunks
    context = "\n\n".join(retrieved_chunks[:2])
    
    # Generate response
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)
    
    # Extract answer
    answer_start = response.find("Answer:") + len("Answer:") if "Answer:" in response else 0
    answer = response[answer_start:].strip() if answer_start > 0 else response.strip()
    
    # Update history
    history.append(f"Q: {question}\nA: {answer}\nSources: {retrieved_chunks[:2]}\n---")
    
    return answer, "\n".join(retrieved_chunks[:2]), "\n".join(history[-5:]) if show_history else ""

# Feedback function
def submit_feedback(feedback, question, answer):
    with open("feedback.txt", "a") as f:
        f.write(f"Q: {question}\nA: {answer}\nFeedback: {feedback}\n---\n")
    return "Thank you for your feedback!"

# Export history
def export_history():
    return "\n".join(history) if history else "No history available."

# Gradio interface
with gr.Blocks(title="CrediTrust Complaint Analyst", theme="soft") as demo:
    gr.Markdown("""
    # CrediTrust Complaint Analyst
    A professional tool to analyze consumer financial complaints. Ask questions, view sources, and provide feedback.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(label="Your Question", placeholder="e.g., What are common credit card issues?")
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear", variant="secondary")
            theme_dropdown = gr.Dropdown(["soft", "huggingface", "default"], value="soft", label="Select Theme")
            history_toggle = gr.Checkbox(label="Show History", value=True)
        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="Answer", interactive=False)
            source_output = gr.Textbox(label="Source Chunks", interactive=False)
            history_output = gr.Textbox(label="Conversation History", interactive=False, lines=5)
    
    with gr.Row():
        feedback_input = gr.Textbox(label="Provide Feedback", placeholder="e.g., Answer was helpful/unhelpful")
        feedback_btn = gr.Button("Submit Feedback", variant="secondary")
        export_btn = gr.Button("Export History", variant="secondary")
        export_output = gr.Textbox(label="Exported History", interactive=False, lines=5)
    
    # Event handlers
    submit_btn.click(
        fn=rag_pipeline,
        inputs=[question_input, history_toggle],
        outputs=[answer_output, source_output, history_output]
    ).then(
        fn=lambda: gr.update(value=""),  # Clear question input
        inputs=None,
        outputs=[question_input]
    )
    
    clear_btn.click(
        fn=lambda: (gr.update(value=""), gr.update(value=""), gr.update(value=""), gr.update(value="")),
        outputs=[question_input, answer_output, source_output, history_output]
    )
    
    theme_dropdown.change(
        fn=lambda theme: gr.Blocks(theme=theme).update(theme=theme),
        inputs=theme_dropdown,
        outputs=demo
    )
    
    feedback_btn.click(
        fn=submit_feedback,
        inputs=[feedback_input, question_input, answer_output],
        outputs=[export_output]
    )
    
    export_btn.click(
        fn=export_history,
        inputs=None,
        outputs=export_output
    )

demo.launch()