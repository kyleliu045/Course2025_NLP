# Retrieval-Augmented Generation (RAG) Assignment

This repository presents a basic implementation of a Retrieval-Augmented Generation (RAG) system via the `nlp_rag_example.ipynb` Jupyter Notebook, executed on Google Colab with GPU support.

The primary goal of this work is to replicate the reference implementation. This `README.md` serves as a concise project report, detailing the usage of the provided sample code and explaining the resulting baseline performance.

---

## Table of Contents

- Project Overview
- Dataset
- Dependencies
- Setup Instructions
- Code Structure, Explaination, and Outputs

---

## Project Overview

Retrieval-Augmented Generation (RAG) is a hybrid approach combining information retrieval and text generation to answer questions based on a large corpus of documents. This project implements a RAG system with:

- **Retriever**: A dense retriever using `sentence-transformers/all-MiniLM-L6-v2` to retrieve relevant documents.
- **Generator**: A large language model `TinyLlama/TinyLlama-1.1B-Chat-v1.0` to generate answers based on retrieved documents.

The notebook covers loading the dataset, encoding documents, retrieving top documents for each question, generating answers, and evaluating performance using metrics like Hit Rate, Term Match Recall, and BERTScore.

---

## Dataset

The project uses two datasets:

- `documents.json`: Contains 1,908 medical-related documents, each with a unique `doc_id` and `text`.
- `questions.json`: Contains 325 questions, each with a unique `question_id` and `question`.

Each question is linked to relevant documents from `documents.json`, which serve as the context for generating answers.

---

## Dependencies

The project relies on the following Python libraries:

- `transformers`: For loading and using pre-trained models like TinyLlama.
- `sentence-transformers`: For document and query embeddings.
- `accelerate`: For efficient model loading and GPU utilization.
- `torch`: For tensor operations and GPU support.
- `json`: For handling JSON files.
- `tqdm`: For progress bars during processing.

Install and import these dependencies in the notebook with:
```bash
!pip install -q transformers sentence-transformers accelerate
```
```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
```

---

## Setup Instructions

To run this project:

#### 1. Set up Google Colab with GPU:

  - Open the notebook in Google Colab.
  - Enable GPU: Go to Runtime > Change runtime type > Select GPU (e.g., Tesla T4).

#### 2. Install Dependencies:
  - Run the first cell to install required libraries.

#### 3. Upload Files:
  - Upload `documents.json`, `questions.json`, `eval_bertscore.py`, `eval_retrieval.so` and `evaluate.py` to the Colab environment.

#### 4. Run the Notebook:
  - Execute cells sequentially to process the data, run the RAG pipeline, and evaluate results.

--- 

## Code Structure, Explanation, and Outputs

The Jupyter Notebook is organized into the following sections:

### 1. Environment Setup & Load Input Files
  - Installs libraries and loads `documents.json` and `questions.json`.
```python
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

documents = load_json('documents.json')
questions = load_json('questions.json')
```

### 2. Define Dense Retriever (Sentence Embedding Model)
  - Loads `sentence-transformers/all-MiniLM-L6-v2` for encoding documents and queries.
  - **Output**: Model loaded successfully.
```python
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```
![image](https://github.com/user-attachments/assets/0d3a59f5-8c1d-4ad9-93f1-2a135986734d)

### 3. Save Document Embeddings
  - Encodes all documents and saves embeddings for retrieval.
  - **Output**: Embeddings saved to `doc_embeddings.pt`.
```python
def preprocess(text):
    emb = embedding_model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    return emb

print("Encoding documents...")
doc_embeddings = []
for doc in tqdm(documents):
    emb = preprocess(doc["text"])
    doc_embeddings.append({
        "doc_id": doc["doc_id"],
        "embedding": emb
    })

from google.colab import files
embedding_data = {
    doc["doc_id"]: doc["embedding"].cpu() for doc in doc_embeddings
}

torch.save(embedding_data, "doc_embeddings.pt")
```
![image](https://github.com/user-attachments/assets/98eeb898-7928-4b9b-ae2a-0e2a9fb824a5)

### 4. Define Generator Model (Large Language Model)
  - Loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for answer generation.
  - **Output**: Model loaded successfully.
```python
hf_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(hf_model)
generate_model = AutoModelForCausalLM.from_pretrained(hf_model, device_map="auto", torch_dtype=torch.float16)

def ask_llm(context, question):
    prompt = f"""<|system|>
    You are a helpful medical assistant.
    <|user|>
    Answer the following question using the provided context.

    Context:
    {context}

    Question:
    {question}
    <|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(generate_model.device)
    outputs = generate_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in response:
        return response.split("<|assistant|>")[-1].strip()
    else:
        return response.strip()
```
![image](https://github.com/user-attachments/assets/a3be2530-b40e-4dfb-929c-11e5b1ce0730)

### 5. RAG Pipeline
  - Retrieves top-5 documents per question using cosine similarity, then uses top-3 as context for the LLM to generate answers.
  - **Output**: Answers saved to pred.json.
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
doc_embeddings = torch.load("doc_embeddings.pt")
doc_lookup = {doc["doc_id"]: doc["text"] for doc in documents}

def cosine_similarity(tensor1, tensor2):
    tensor1 = F.normalize(tensor1.unsqueeze(0))
    tensor2 = F.normalize(tensor2.unsqueeze(0))
    return torch.mm(tensor1, tensor2.T).item()

print("Retrieving top 5 documents for each question...")
results = []

for q in tqdm(questions):
    q_emb = preprocess(q["question"])
    scored_docs = []

    for doc_id, doc_emb in doc_embeddings.items():
        score = cosine_similarity(q_emb, doc_emb.to(device))
        scored_docs.append((doc_id, score))

    top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:5]
    retrieved_ids = [doc_id for doc_id, _ in top_docs]

    combined_context = "\n\n".join([doc_lookup[doc_id] for doc_id in retrieved_ids[:3] if doc_id in doc_lookup])

    resp = ask_llm(combined_context, q["question"])

    results.append({
        "question_id": q["question_id"],
        "retrieved_docs": retrieved_ids,
        "answer": resp
    })

with open('pred.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

### 6. Evaluation Setup

The script (`evaluate.py`) serves as the entry point for evaluating retrieval and generation results based on a given prediction file `pred.json`.

#### Required files:

  - `pred.json`: Input prediction file containing retrieved documents and generated answers.
  - `evaluate.py`: The main script coordinating the evaluation pipeline.
  - `eval_bertscore.py`: Script to evaluate answer quality using BERTScore.
  - `eval_retrieval.so`: Compiled shared object (.so) file for evaluating retrieval accuracy.

**Output**: Results saved to result.json.

#### Environment Setup:

```bash
! pip install cython==3.1.0
! pip install bert_score==0.3.13
```

### 7. Evaluation (Complete Evaluation)

  - Evaluates retrieval and generation using Hit Rate, Term Match Recall, and BERTScore.
  - **Output**: Average retrieval_hits: 84.23%, Average term_match_recall: 57.54%, Average bert_score: 50.22%.
```bash
!python evaluate.py --topk 3 --use_bertscore
```
![image](https://github.com/user-attachments/assets/19b5c7e6-23fe-47f6-a80f-4ff210b8b05a)

#### Explaination of the baseline results

The baseline RAG system, utilizing `sentence-transformers/all-MiniLM-L6-v2` for retrieval and `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for generation, yielded the following results: Average retrieval_hits @3: 84.23%, Average term_match_recall: 57.54%, and Average bert_score: 50.22%. The retrieval_hits @3 of 84.23% indicates that the retriever successfully identifies at least one relevant document within the top-3 results for most questions, demonstrating effective document retrieval. However, the term_match_recall of 57.54% is relatively low, suggesting that the generated answers often fail to include key terms or phrases from the reference answers, likely due to the limited expressive capacity of the TinyLlama model. The bert_score of 50.22% reflects moderate semantic similarity between generated and ground truth answers, but there is significant room for improvement compared to the strong baseline (e.g., Term Match Recall: 97.33%), particularly in capturing precise terminology and enhancing answer quality.

