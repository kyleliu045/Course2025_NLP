# RAG System Improvement Report

This repository documents an enhanced implementation of a Retrieval-Augmented Generation (RAG) system, built upon the baseline provided in the `nlp_rag_example.ipynb` Jupyter Notebook. 

The primary objective is to improve the RAG pipeline's retrieval accuracy and answer quality, evaluated using metrics such as Retrieval Hits, Term Match Recall, and BERTScore. This README serves as a detailed report of the testing process, modifications, and final results.

---
## Project Overview

Retrieval-Augmented Generation (RAG) integrates information retrieval with text generation to provide answers based on a document corpus. This project enhances the baseline RAG system by:

- Upgrading the retriever (sentence embedding) and generator components.
- Refining the prompt to improve answer relevance and accuracy.

The baseline system uses `sentence-transformers/all-MiniLM-L6-v2` for retrieval and `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for generation, achieving a Term Match Recall of 57.54%. Our improvements aim to approach or exceed the strong baseline's Term Match Recall of 97.33%.

---
## Hardware and Environment Setup

This project was executed locally using Windows Subsystem for Linux (WSL) instead of Google Colab, due to the following limitations of Colab's free tier:

- Limited GPU resource availability.
- Automatic disconnection after prolonged inactivity.

### Hardware Specifications
- **CPU**: Intel Core i5-14400F
- **GPU**: NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)
- **GPU memory**: Total 31.9 GB (16 GB VRAM + shared GPU memory 15.9 GB)
- **OS**: Windows 11 (via WSL2 with Ubuntu 22.04).

### Environment Setup
A virtual environment was set up with Python 3.11 to manage dependencies and ensure reproducibility. To set up the environment:

#### 1. Set Up Python Virtual Environment:
- Create a virtual environment: `conda create --name env_rag python=3.11`
- Activate the environment: `conda activate env_rag`

#### 2. Install Dependencies:
- For `rag_improved.py`:
```bash
pip install -q transformers sentence-transformers accelerate
```
- For `evaluate.py`:
```bash
pip install cython==3.1.0
pip install bert_score==0.3.13
```

#### 3. Run the Project:
- Upload `documents.json`, `questions.json`, `eval_bertscore.py`, `eval_retrieval.so`, and `evaluate.py` to the WSL environment.
- Execute the Python scripts (`rag_improveed.py`、`evaluate.py`) sequentially.

---
## Model Information

This project utilizes multiple models to implement the Retrieval-Augmented Generation (RAG) system's retrieval and generation capabilities. The following details provide an overview of each model:

### Sentence Embedding Models:
- **Baseline**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
   - Type: Transformer-based sentence embedding model
   - Version: v2
   - Parameters: ~22M
   - Characteristics: Lightweight model optimized for fast retrieval, though limited in semantic capture.
- **Optimized**: [`intfloat/e5-large`](https://huggingface.co/intfloat/e5-large)
   - Type: Transformer-based embedding model
   - Version: Latest release (2023)
   - Parameters: ~335M
   - Characteristics: Enhanced semantic understanding and retrieval accuracy, particularly suited for diverse medical queries.

### Generator Models:
- **Baseline**: [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
   - Type: Causal language model
   - Version: v1.0
   - Parameters: ~1.1B
   - Characteristics: Compact model with limited generation capacity, resulting in lower baseline performance.
- **Tested Varients**:
   - [`MiniMaxAI/SynLogic-7B`](https://huggingface.co/MiniMaxAI/SynLogic-7B): 7B parameter model optimized for dialogue.
   - [`Qwen/Qwen2.5-7B`](https://huggingface.co/Qwen/Qwen2.5-7B): 7B parameter model supporting multilingual.
   - [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct): ~3.8B parameter instruction-tuned model with high generation quality.

---
## Testing Process and Results

Multiple tests were conducted to optimize the RAG system by adjusting the generator model and prompt. The table below summarizes each test's configuration and outcomes.

| Test | Generator Model             | Prompt Modifications                             | Results (Average)                          |
|------|-----------------------------|--------------------------------------------------|--------------------------------------------|
| 1    | `"MiniMaxAI/SynLogic-7B"`   | Basic prompt emphasizing concise, accurate answers based on context. | - Retrieval Hits: 84.23% <br>- Term Match Recall: 75.03% <br>- BERTScore: 45.91% |
| 2    | `"Qwen/Qwen2.5-7B"`         | Same as Test 1.                                  | - Retrieval Hits: 84.23% <br>- Term Match Recall: 72.36% <br>- BERTScore: 44.14% |
| 3    | `"Qwen/Qwen2.5-7B"`         | Reverted to baseline prompt for comparison.      | - Retrieval Hits: 84.23% <br>- Term Match Recall: 43.33% <br>- BERTScore: 28.18% |
| 4    | `"microsoft/Phi-4-mini-instruct"` | Enhanced prompt focusing on medical accuracy. | - Retrieval Hits: 84.23% <br>- Term Match Recall: 86.72% <br>- BERTScore: 29.49% |
| 5    | `"microsoft/Phi-4-mini-instruct"` | Same as Test 4, with embedding model `"intfloat/e5-large"`. | - Retrieval Hits: 93.45% <br>- Term Match Recall: 96.21% <br>- BERTScore: 29.68% |

### Analysis:

- **Tests 1-4**: With `all-MiniLM-L6-v2`, Retrieval Hits remained at 84.23%. 
- **Test 1 & 2**: Switching to larger models (7B) and simply modifying the prompt improved Term Match Recall over the baseline (57.54%).
- **Test 3**: Reverting to the baseline prompt significantly degraded performance, underscoring the value of prompt engineering.
- **Test 4**: The enhanced prompt with "Phi-4-mini-instruct" boosted Term Match Recall further.
- **Test 5**: Upgrading the retriever to "intfloat/e5-large" markedly improved Retrieval Hits and Term Match Recall, nearing the strong baseline.

---
## Final Configuration and Results
The optimal configuration is:
- **Embedding Model**: `"intfloat/e5-large"`
- **Generator Model**: `"microsoft/Phi-4-mini-instruct"`
- **Prompt**:
  ```plain
  <|system|>
  You are an expert medical assistant with access to a comprehensive database of medical documents. 
  Your role is to provide precise, concise, and medically accurate answers based solely on the provided context. 
  Use medical terminology appropriately. If a medical term is used in the context without definition, 
  you may provide a brief explanation based on standard medical knowledge, 
  but only if you are confident in its accuracy. Respond in a professional and informative tone.
  <|user|>
  Using only the information in the context below, provide a concise and accurate answer to the following question. 
  Focus on the most relevant parts of the context that directly address the question. 
  If the context does not contain the necessary information, state that clearly.

  Context: {context}

  Question: {question}
  <|assistant|>
  ```

#### Final Results:
- **Average Retrieval Hits**: 93.45%
- **Average Term Match Recall**: 96.21%
- **Average BERTScore**: 29.68%
- **Execution Time**: ~15 minutes for 325 questions

Compared to the baseline (Retrieval Hits: 84.23%, Term Match Recall: 57.54%, BERTScore: 50.22%), this configuration achieves substantial gains, approaching the strong baseline's Term Match Recall of 97.33%.

Terminal Screenshot:
![image](https://github.com/user-attachments/assets/6f0be18f-be0a-44df-b196-e06a37f682f4)

---

## Conclusion

Through iterative testing, the RAG system was significantly enhanced by optimizing the generator model, embedding model, and prompt. The final configuration delivers high retrieval accuracy and term recall, making it suitable for medical question-answering tasks. Future improvements could focus on boosting BERTScore through larger models or further prompt refinements.

