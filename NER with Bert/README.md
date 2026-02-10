# Named Entity Recognition (NER) with BERT on WikiANN Dataset
This repository contains a Jupyter Notebook `NTNUCSIE_NLP_BERT_NER.ipynb` that demonstrates how to fine-tune a BERT-based model `DistilBERT` for Named Entity Recognition (NER) on the WikiANN dataset using Google Colab with GPU acceleration. 

The project focuses on identifying and classifying named entities (e.g., Person, Organization, Location) in text using the BIO (Beginning, Inside, Outside) tagging scheme.

---
## Table of Contents

- Project Overview
- Dataset
- Dependencies
- Setup Instructions
- Code Structure, Explanation, and Outputs
- Usage

---
## Project Overview

Named Entity Recognition (NER) is a natural language processing (NLP) task that involves identifying and categorizing named entities (e.g., persons, organizations, locations) in text.

This project fine-tunes the distilbert-base-uncased model from the Hugging Face Transformers library on the English subset of the WikiANN dataset to perform NER.

The notebook includes data preprocessing, model training, and evaluation steps, leveraging GPU acceleration in Google Colab for efficient computation.

The project achieves high performance, with evaluation metrics (precision, recall, F1-score, and accuracy) computed on the test set after training.

---
## Dataset

The [WikiANN dataset](https://huggingface.co/datasets/unimelb-nlp/wikiann) (also known as PAN-X) is a multilingual NER dataset derived from Wikipedia articles. This project uses the English subset, which contains:

- 20,000 training samples
- 10,000 validation samples
- 10,000 test samples

 Each sample includes:

 - `tokens`: A list of words in a sentence.
 - `ner_tags`: Labels in BIO format (e.g., O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC).
 - `langs`: Language identifiers (all `en` for English in this case).
 - `spans`: Human-readable entity spans (e.g., PER: Anders Lindström).

The NER tags follow the IOB2 format:

- `O`: Token is not part of a named entity.
- `B-TYPE`: Beginning of a named entity of type TYPE (e.g., B-PER for a person).
- `I-TYPE`: Inside a named entity of type TYPE (e.g., I-PER).

---
## Dependencies

The project relies on the following Python libraries:

- `datasets`: For loading and processing the WikiANN dataset.
- `tokenizers`: For tokenization utilities.
- `transformers`: For the DistilBERT model and training utilities.
- `seqeval`: For computing NER-specific evaluation metrics (precision, recall, F1-score, accuracy).
- `numpy`: For numerical operations during evaluation.

These dependencies are installed in the notebook via pip:
```bash
!pip install -U datasets
!pip install tokenizers -q
!pip install transformers -q
!pip install seqeval==0.0.3 -q
```

---
## Setup Instructions

To run this project, follow these steps:

1. Set up Google Colab with GPU:
- Open the notebook in Google Colab.
- Enable GPU acceleration: Go to Runtime > Change runtime type > Select GPU (e.g., Tesla T4).
2. Install Dependencies:
- Run the first cell to install the required libraries (datasets, tokenizers, transformers, seqeval).
3. Verify GPU Availability:
- Run the `!nvidia-smi` command to confirm that a GPU is available.
4. Run the Notebook:
- Execute the cells sequentially to load the dataset, preprocess the data, fine-tune the model, and evaluate the results.

---
## Code Structure, Explanation, and Outputs

The Jupyter Notebook is organized into the following sections:

1. Dependency Installation
- Installs necessary Python libraries using pip.
- Checks GPU availability with `nvidia-smi`.
2. Dataset Loading
- Loads the English subset of the WikiANN dataset using `datasets.load_dataset("wikiann", "en")`.

![image](https://github.com/user-attachments/assets/10bc472e-ae05-4c05-9943-a9755c3c6def)

- Displays dataset statistics (20,000 train, 10,000 validation, 10,000 test samples).

![image](https://github.com/user-attachments/assets/d421c072-0217-4ea4-9fb3-b8fcaf346f5b)

- Shows label names ('O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC') and sample data.

![image](https://github.com/user-attachments/assets/26c5e14d-e867-4cec-9fea-8bb94760b7c4)

3. Data Preprocessing
- Tokenization:
    - Uses `AutoTokenizer` from `transformers` to tokenize the dataset with `distilbert-base-uncased`.  
![image](https://github.com/user-attachments/assets/e6b21c74-a32f-4ac7-b70d-116e98787bf2)

    - Tokens are padded to a fixed length (`padding="max_length"`) and truncated if necessary (`truncation=True`).
![image](https://github.com/user-attachments/assets/0d3c4aeb-3c07-4316-a8f5-013dd7dcb4e3)

- Label Alignment:
    - Defines a custom function (`tokenize_adjust_labels`) to align NER tags with tokenized inputs.
    - Handles subword tokenization by assigning the same label to subword tokens and setting special tokens to -100 (ignored in loss computation).
    - Removes unnecessary columns (`tokens`, `ner_tags`, `langs`, `spans`) from the tokenized dataset.

![image](https://github.com/user-attachments/assets/2c3a09ea-3c0f-4320-9e91-180fd8b256e6)

![image](https://github.com/user-attachments/assets/c684009c-bc45-4509-bd55-fa2c211bf1ae)

4. Fine-Tuing
- Model Initialization:
    - Loads `AutoModelForTokenClassification` with `distilbert-base-uncased` and configures it for 7 labels (corresponding to the WikiANN label set).
- Data Collator:
    - Uses `DataCollatorForTokenClassification` to dynamically pad inputs and labels during training.
- Training Arguments:
    - Configures training parameters (e.g., batch size = 64, epochs = 2, logging steps).
    - Sets output directory to "results" and evaluation strategy to "epoch".
- Trainers:
    - Initializes a `Trainer` object with the model, dataset, data collator, and custom metrics (precision, recall, F1, accuracy).
    - Fine-tunes the model using the `trainer.train()` method.
![image](https://github.com/user-attachments/assets/0552c239-9759-4bdb-8b17-2bc3d73d704e)

5. Evaluation
- Evaluates the model on the test set using `trainer.predict()`.
- Computes metrics (precision, recall, F1-score, accuracy) by comparing predicted labels with true labels, ignoring special tokens (-100).
- Output: **Precision: 0.8128, Recall: 0.8325, F1: 0.8225, Accuracy: 0.9226**
---
## Key Functions

- `tokenize_adjust_labels`: Aligns NER labels with tokenized inputs, handling subword tokens and special tokens.
- `compute_metrics`: Calculates NER-specific metrics (precision, recall, F1, accuracy) using `seqeval`.

--- 
## Usage

To use the fine-tuned model for inference:

1. Load the trained model from the `results` directory.
2. Tokenize new text input using the same `distilbert-base-uncased` tokenizer.
3. Pass the tokenized input through the model to obtain predictions.
4. Convert predicted label IDs back to label names using `label_names`.


