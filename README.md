# Clinical Agentic RAG System for Multi-Modal Medical Reasoning

## Project Overview

This project implements an Agentic Retrieval-Augmented Generation (RAG) system capable of reasoning across both structured and unstructured clinical data sources.

The system integrates:

- Structured patient records stored in CSV format
- Unstructured clinical guidelines stored in PDF format

Unlike traditional RAG pipelines that rely only on semantic retrieval, this system uses an Agentic Architecture powered by Llama-3.1-70B. The agent autonomously selects tools, retrieves relevant information, and performs multi-step reasoning to generate clinically grounded responses.

This enables deterministic database querying, semantic retrieval, and cross-modality reasoning in a unified pipeline.

---

## Technical Stack

**LLM (Reasoning Engine):**
meta/llama-3.1-70b-instruct (via NVIDIA NIM)

**Embedding Model:**
nvidia/nv-embedqa-e5-v5

**Framework:**
LangChain (Agent orchestration and tool calling)

**Vector Database:**
FAISS (CPU-optimized local vector store)

**Structured Data Engine:**
Pandas DataFrame Agent

**User Interface:**
Streamlit

**Language:**
Python 3.11

**Environment:**
Conda

---

## System Architecture

The pipeline consists of four major layers:

### 1. Ingestion Layer

Unstructured Data (PDF):

- Loaded using PyPDFLoader
- Split using RecursiveCharacterTextSplitter
- Chunk size: 800
- Overlap: 100
- Preserves clinical semantic context

Structured Data (CSV):

- Loaded into a persistent Pandas DataFrame
- Enables deterministic filtering and aggregation

---

### 2. Indexing Layer

- PDF chunks are converted into embeddings using nv-embedqa-e5-v5
- Embeddings stored in FAISS vector database
- Enables fast semantic similarity retrieval

---

### 3. Agentic Tool Layer

Two specialized tools enable retrieval from different modalities:

**search_pdf**

- Performs semantic similarity search
- Retrieves clinical protocol information
- Uses FAISS vector store

**search_csv**

- Executes dynamic Python queries
- Filters and analyzes patient records
- Powered by LangChain Pandas agent

---

### 4. Agentic Reasoning Layer

The Llama-3.1-70B agent performs:

- Query understanding
- Tool selection
- Multi-step reasoning
- Context fusion
- Final response synthesis

This enables true multi-modal reasoning.

---

## End-to-End Workflow

### Phase 1: Ingestion and Vectorization

PDF Pipeline:

- Parse clinical guideline PDF
- Split into semantic chunks
- Convert chunks into embeddings
- Store embeddings in FAISS index

CSV Pipeline:

- Load patient records into Pandas DataFrame
- Enable structured querying

---

### Phase 2: Agentic Reasoning Loop

Example Query:

Does Patient ID 102's A1C level match the guideline recommendations?

Execution Steps:

1. Agent analyzes query intent
2. Calls search_csv tool to retrieve patient A1C value
3. Calls search_pdf tool to retrieve clinical A1C guideline
4. Fuses both retrieved contexts
5. Generates final clinically grounded answer

---

## Workflow Diagram

```
User Query
    |
    v
Llama 3.1 Agent (Reasoning Brain)
    |
    |------------------------|
    v                        v
search_csv Tool        search_pdf Tool
    |                        |
    v                        v
Pandas DataFrame       FAISS Vector Store
    |                        |
    |------------------------|
             |
             v
Context Fusion and Reasoning
             |
             v
Final Clinical Response
```

---

## Key Capabilities

Deterministic Structured Queries:

Example:
Find all patients with Type 2 Diabetes over age 60.

Semantic Protocol Retrieval:

Example:
What is the recommended dosage of Metformin?

Cross-Modality Clinical Reasoning:

Example:
Does Patient 104 meet guideline A1C recommendations?

---

## User Interface

Built using Streamlit.

Features:

- ChatGPT-style interface
- Interactive query input
- Real-time agent responses
- Session state chat history

---

## Project Structure

```
clinical-rag/
│
├── gui.py
├── rag_pipeline.py
├── search_tools.py
│
├── data/
│   ├── mtsamples.csv
│   └── clinical_guidelines.pdf
│
├── vectorstore/
│   └── faiss_index
│
├── .env
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Step 1: Create Conda Environment

```
conda create -n clinical_rag python=3.11
conda activate clinical_rag
```

---

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

---

### Step 3: Configure API Key

Create a .env file:

```
NVIDIA_API_KEY=your_api_key_here
```

---

### Step 4: Run Application

```
streamlit run gui.py
```

---

## Example Queries

You can test queries such as:

- What is the recommended A1C level?
- Show patient ID 102 records.
- Does patient 104 meet guideline recommendations?
- List diabetic patients above age 60.

---

## Implementation Details

- FAISS used for fast local vector search
- LangChain agent enables autonomous tool usage
- Pandas agent enables structured reasoning
- Secure credential handling using .env
- Modular architecture for extensibility

---

## Key Innovation

This project implements an Agentic RAG architecture capable of:

- Multi-modal reasoning
- Autonomous tool selection
- Structured and unstructured data fusion
- Clinical-grade reasoning workflows

---

## Author

Vaishnavi Srivastava

Machine Learning Engineer | AI Systems | Data Science
