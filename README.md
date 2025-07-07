# Intelligent Complaint Analysis for Financial Services

Welcome to the **Intelligent Complaint Analysis for Financial Services** project! This repository implements a Retrieval-Augmented Generation (RAG) pipeline to analyze consumer complaints from the [Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/). The project includes Exploratory Data Analysis (EDA), preprocessing, text chunking, embedding, and vector store indexing to support a chatbot for financial complaint analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Task 1: EDA and Preprocessing](#task-1-eda-and-preprocessing)
- [Task 2: Text Chunking, Embedding, and Indexing](#task-2-text-chunking-embedding-and-indexing)
- [Troubleshooting](#troubleshooting)
- [Submission Instructions](#submission-instructions)
- [Future Work](#future-work)

## Project Overview
This project processes consumer complaint data (`complaints.csv`) to enable semantic search for a chatbot. It consists of two tasks:
- **Task 1**: Perform EDA and preprocess the dataset, filtering for relevant products and cleaning narratives.
- **Task 2**: Chunk narratives, generate embeddings, and index them in a FAISS vector store.

The project uses Python 3.13.3, a virtual environment (`venv`), and libraries like `pandas`, `langchain`, `sentence-transformers`, and `faiss-cpu`. All code is developed on Windows using VS Code.

## Repository Structure
```
Intelligent-Complaint-Analysis-for-Financial-Services/
├── data/
│   ├── complaints_cleaned.csv       # Preprocessed dataset (500 rows)
│   └── word_count_distribution.png  # EDA visualization
├── notebooks/
│   ├── eda_preprocessing.ipynb      # Task 1: EDA and preprocessing
│   └── eda_report.md               # Report for Tasks 1 and 2
├── scripts/
│   └── chunk_embed_index.py         # Task 2: Chunking, embedding, indexing
├── vector_store/
│   ├── complaint_embeddings.faiss   # FAISS vector store
│   └── chunk_metadata.pkl          # Metadata for chunks
├── .github/workflows/
│   └── ci.yml                      # CI pipeline for syntax checks
├── .gitignore                      # Git ignore file
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Setup Instructions
Follow these steps to set up the project on a Windows machine.

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

4. **Download complaints.csv**:
   - Place `complaints.csv` in `data/` from [CFPB](https://www.consumerfinance.gov/data-research/consumer-complaints/).

## Task 1: EDA and Preprocessing
### Objective
Analyze and preprocess `complaints.csv` to prepare narratives for the RAG pipeline.

### Steps
1. **Run Notebook**:
   ```bash
   jupyter notebook
   ```
   - Open `notebooks/eda_preprocessing.ipynb`.
   - Run all cells to:
     - Generate `data/word_count_distribution.png` (word count histogram).
     - Filter products (e.g., Credit Card, Consumer Loan).
     - Clean narratives (lowercase, remove special characters, boilerplate).
     - Sample 500 rows to create `data/complaints_cleaned.csv`.

2. **Outputs**:
   - `data/complaints_cleaned.csv`: Preprocessed dataset.
   - `data/word_count_distribution.png`: Visualization.
   - Console: Word count stats, narrative counts.

3. **Report**:
   - See `notebooks/eda_report.md` for details.

### Example Output
```
Dataset columns: ['Complaint ID', 'Product', 'Consumer complaint narrative', ...]
Word Count Summary Statistics:
count    500.000000
mean     120.450000
...
Total complaints after sampling (500 rows): 500
```

## Task 2: Text Chunking, Embedding, and Indexing
### Objective
Convert cleaned narratives into a vector store for semantic search.

### Steps
1. **Run Script**:
   ```bash
   python scripts/chunk_embed_index.py
   ```
   - **Chunking**: Uses LangChain’s `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`).
   - **Embedding**: Uses `sentence-transformers/all-MiniLM-L6-v2` for 384-dimensional embeddings.
   - **Indexing**: Stores embeddings in FAISS (`IndexFlatL2`) with metadata (`Complaint ID`, `Product`, `chunk_index`).

2. **Outputs**:
   - `vector_store/complaint_embeddings.faiss`: Vector store.
   - `vector_store/chunk_metadata.pkl`: Metadata.
   - Console: ~1409 chunks created, model loaded, store saved.

3. **Report**:
   - See `notebooks/eda_report.md` for chunking and embedding details.

### Example Output
```
Current working directory: C:\Users\Skyline\Intelligent-Complaint-Analysis-for-Financial-Services
Cleaned dataset loaded successfully.
Total chunks created: 1409
Embedding model loaded successfully.
Vector store and metadata saved to vector_store/
```

## Troubleshooting
### Task 1 Issues
- **KeyError**: Verify column names in `data/complaints.csv`:
  ```bash
  python -c "import pandas as pd; df = pd.read_csv('data/complaints.csv'); print(df.columns)"
  ```
  Update `notebooks/eda_preprocessing.ipynb` (e.g., `Consumer_complaint_narrative`).

### Task 2 Issues
- **MemoryError**:
  - Close other applications or restart your computer.
  - Clear Hugging Face cache:
    ```bash
    rmdir /S C:\Users\Skyline\.cache\huggingface
    ```
  - Reduce batch size in `scripts/chunk_embed_index.py` (e.g., `batch_size=4`).
  - Reduce dataset size in `notebooks/eda_preprocessing.ipynb` (e.g., `n=200`).

- **Model Download Error**:
  - Ensure stable internet.
  - Install `hf_xet`:
    ```bash
    pip install hf_xet
    ```

### Git Push Issues
- **Non-fast-forward/Unrelated Histories**:
  ```bash
  git pull origin main --allow-unrelated-histories
  git add .
  git commit -m "Merge with remote"
  git push origin main
  ```
- **Early EOF**:
  ```bash
  git config --global http.postBuffer 1048576000
  git push --force origin main
  ```

## Submission Instructions
1. **Verify Files**:
   ```bash
   dir notebooks data scripts vector_store .github\workflows .gitignore requirements.txt
   ```
   Ensure:
   - `notebooks/eda_preprocessing.ipynb`
   - `notebooks/eda_report.md`
   - `data/complaints_cleaned.csv`
   - `data/word_count_distribution.png`
   - `scripts/chunk_embed_index.py`
   - `vector_store/complaint_embeddings.faiss`
   - `vector_store/chunk_metadata.pkl`
   - `.github/workflows/ci.yml`
   - `.gitignore`
   - `requirements.txt`

2. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Tasks 1 and 2: EDA, chunking, embedding, and vector store"
   git push origin main
   ```

3. **Manual Upload (Recommended)**:
   - Go to https://github.com/yankee998/Intelligent-Complaint-Analysis-for-Financial-Services.
   - Create directories (`notebooks/`, `data/`, `scripts/`, `vector_store/`, `.github/workflows/`).
   - Upload all files listed above.
   - Commit to `main` with message: "Tasks 1 and 2 submission: EDA and RAG pipeline".

4. **Submit**:
   - Share the repository link on the challenge platform.

## Future Work
- Optimize chunking parameters for specific complaint types.
- Implement Task 3: Build the chatbot with retrieval and generation.
- Enhance error handling and scalability.

---

**Author**: Yared Genanaw  
**Repository**: https://github.com/yankee998/Intelligent-Complaint-Analysis-for-Financial-Services  
**License**: MIT