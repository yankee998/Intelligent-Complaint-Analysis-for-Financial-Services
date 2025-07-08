# 🎉 Intelligent Complaint Analysis for Financial Services

Welcome to the **Intelligent Complaint Analysis for Financial Services** project! 🚀 This repository implements a cutting-edge **Retrieval-Augmented Generation (RAG)** pipeline to analyze consumer complaints from the [Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/). Leveraging Python 3.13.3, this project evolves from Exploratory Data Analysis (EDA) to a fully functional chatbot, empowering financial institutions to enhance customer service and compliance.

🌟 **Explore the journey**: Click sections below to dive into details, run code snippets, and visualize results!

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Task 1: EDA and Preprocessing](#task-1-eda-and-preprocessing)
- [Task 2: Text Chunking, Embedding, and Indexing](#task-2-text-chunking-embedding-and-indexing)
- [Task 3: Building the Chatbot](#task-3-building-the-chatbot)
- [Task 4: Evaluation and Deployment](#task-4-evaluation-and-deployment)
- [Troubleshooting](#troubleshooting)
- [Submission Instructions](#submission-instructions)
- [Future Work](#future-work)
- [Contributors](#contributors)

## 🌍 Project Overview
This project transforms `complaints.csv` into a semantic search and response system. It comprises four tasks:
- **Task 1**: Perform EDA and preprocess the dataset.
- **Task 2**: Chunk narratives, generate embeddings, and index them in a FAISS vector store.
- **Task 3**: Build a chatbot with retrieval and generation capabilities.
- **Task 4**: Evaluate performance and deploy the solution.

Built with Python 3.13.3 in a `venv` on Windows using VS Code, it uses libraries like `pandas`, `langchain`, `sentence-transformers`, `faiss-cpu`, and `gradio` for an interactive interface.

## 📂 Repository Structure
```
Intelligent-Complaint-Analysis-for-Financial-Services/
├── data/
│   ├── complaints_cleaned.csv       # Preprocessed dataset (500 rows)
│   └── word_count_distribution.png  # EDA visualization
|── captured_Screenshots/
|   |── screenshot_1.pdf
├── notebooks/
│   ├── eda_preprocessing.ipynb      # Task 1: EDA and preprocessing
│   └── eda_report.md               # Report for all tasks
├── scripts/
│   ├── chunk_embed_index.py         # Task 2: Chunking, embedding, indexing
│   └── chatbot.py                  # Task 3: Chatbot implementation
├── vector_store/
│   ├── complaint_embeddings.faiss   # FAISS vector store
│   └── chunk_metadata.pkl          # Metadata for chunks
├── app.py       # Task 4: Deployable app
├── .github/workflows/
│   └── ci.yml                      # CI pipeline for syntax checks
├── .gitignore                      # Git ignore file
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## 🛠️ Setup Instructions
Get started on your Windows machine with these steps:

1. **Clone the Repository**:
   ```bash
   cd C:\Users\Skyline
   git clone https://github.com/yankee998/Intelligent-Complaint-Analysis-for-Financial-Services.git
   cd Intelligent-Complaint-Analysis-for-Financial-Services
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include:
   - `pandas==2.2.3`
   - `matplotlib==3.9.2`
   - `seaborn==0.13.2`
   - `jupyter==1.0.0`
   - `langchain==0.2.16`
   - `sentence-transformers==3.0.1`
   - `faiss-cpu==1.8.0`
   - `hf_xet==0.2.0`
   - `gradio==4.36.0`
   - `nltk==3.8.1`

4. **Download `complaints.csv`**:
   - Place it in `data/` from [CFPB](https://www.consumerfinance.gov/data-research/consumer-complaints/).

## 📊 Task 1: EDA and Preprocessing
### Objective
Analyze and preprocess `complaints.csv` for the RAG pipeline.

### Steps
1. **Run Notebook**:
   ```bash
   jupyter notebook
   ```
   - Open `notebooks/eda_preprocessing.ipynb`.
   - Execute cells to:
     - Generate `data/word_count_distribution.png` (histogram).
     - Filter products (e.g., Credit Card, Consumer Loan).
     - Clean narratives (lowercase, remove special characters).
     - Sample 500 rows to `data/complaints_cleaned.csv`.

2. **Outputs**:
   - `data/complaints_cleaned.csv`
   - `data/word_count_distribution.png`
   - Console stats (word count, narrative counts).

3. **Report**: See `notebooks/eda_report.md`.

## 🔗 Task 2: Text Chunking, Embedding, and Indexing
### Objective
Prepare a vector store for semantic search.

### Steps
1. **Run Script**:
   ```bash
   python scripts/chunk_embed_index.py
   ```
   - **Chunking**: `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`).
   - **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
   - **Indexing**: FAISS `IndexFlatL2` with metadata.

2. **Outputs**:
   - `vector_store/complaint_embeddings.faiss`
   - `vector_store/chunk_metadata.pkl`
   - Console: ~1409 chunks, model loaded, store saved.

3. **Report**: See `notebooks/eda_report.md`.

## 💬 Task 3: Building the Chatbot
### Objective
Create a chatbot using the vector store for retrieval and generation.

### Steps
1. **Run Script**:
   ```bash
   python scripts/chatbot.py
   ```
   - **Retrieval**: Uses LangChain’s `FAISS` retriever with cosine similarity.
   - **Generation**: Integrates a lightweight LLM (e.g., `HuggingFaceHub` with `distilbert`).
   - **Interaction**: Command-line interface for testing.

2. **Outputs**:
   - Console: Interactive prompt (e.g., "Ask about a complaint:").
   - Example response: "Based on complaint ID 12345, issue resolved within 7 days."

3. **Enhancements**:
   - Add context-aware responses.
   - Log queries in `logs/chatbot.log`.

### Example Usage
```
> Ask about a complaint: Credit card fraud
> Response: Complaint ID 12345 reports credit card fraud, resolved in 7 days. Contact support for details.
```

## 🚀 Task 4: Evaluation and Deployment
### Objective
Evaluate the chatbot and deploy it as a web app.

### Steps
1. **Evaluate**:
   - Run `python app/complaint_chatbot.py --eval`.
   - Metrics: Precision, recall, and user satisfaction (manual feedback).
   - Output: `evaluation_results.txt` with scores.

2. **Deploy**:
   - Launch web app:
     ```bash
     python app/complaint_chatbot.py
     ```
   - Opens a Gradio interface at `http://localhost:7860`.
   - Features: Text input, response display, and feedback button.

3. **Outputs**:
   - `evaluation_results.txt`
   - Web app with real-time interaction.

### Example Web Interaction
- Input: "Tell me about loan issues."
- Output: "Complaint ID 67890 highlights loan processing delays, resolved in 14 days."

## 🔧 Troubleshooting
### Task 1 Issues
- **KeyError**: Check column names:
  ```bash
  python -c "import pandas as pd; df = pd.read_csv('data/complaints.csv'); print(df.columns)"
  ```
  Update `notebooks/eda_preprocessing.ipynb`.

### Task 2 Issues
- **MemoryError**: Close apps, clear cache:
  ```bash
  rmdir /S C:\Users\Skyline\.cache\huggingface
  ```
  Adjust batch size in `scripts/chunk_embed_index.py`.

### Task 3 Issues
- **LLM Error**: Ensure `hf_xet` is installed and internet is stable.

### Task 4 Issues
- **Gradio Not Loading**: Verify port 7860 is free:
  ```bash
  netstat -aon | findstr :7860
  ```
  Change port in `app/complaint_chatbot.py` if needed.

### Git Push Issues
- **Non-fast-forward**: 
  ```bash
  git pull origin main --allow-unrelated-histories
  git push origin main
  ```

## 📤 Submission Instructions
1. **Verify Files**:
   ```bash
   dir notebooks data scripts vector_store app .github\workflows .gitignore requirements.txt
   ```
   Ensure all files from the structure are present.

2. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Tasks 1-4: EDA, RAG pipeline, chatbot, and deployment"
   git push origin main
   ```

3. **Manual Upload**:
   - Go to https://github.com/yankee998/Intelligent-Complaint-Analysis-for-Financial-Services.
   - Upload all files and commit to `main`.

4. **Submit**: Share the link on the challenge platform.

## 🌱 Future Work
- Optimize LLM for faster responses.
- Add multi-language support.
- Integrate real-time complaint feeds.

## 👥 Contributors
- **Yared Genanaw**: Lead Developer  
- [Contribute!](mailto:yared@example.com)  

---

**📅 Last Updated**: 07:40 PM EAT, July 08, 2025  
**License**: MIT  
**Repository**: https://github.com/yankee998/Intelligent-Complaint-Analysis-for-Financial-Services