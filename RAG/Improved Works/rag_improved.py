# Environment setup (in Terminal)
# pip install -q transformers sentence-transformers accelerate

# Import necessary libraries
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Json loading utility
# This function loads a JSON file and returns its content as a Python dictionary.
# It is useful for reading configuration files or datasets in JSON format.
import json
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

documents = load_json('documents.json')  # Load documents from a JSON file
questions = load_json('questions.json')  # Load questions from a JSON file

# Define Sentance Embedding Model
embedding_model = SentenceTransformer("intfloat/e5-large")
def preprocess(text):
    """ Preprocess the text by encoding it into embeddings using the SentenceTransformer model.
    Args:
        text (str): The input text to be encoded.
    Returns:
        torch.Tensor: The normalized embedding of the input text.
    """
    emb = embedding_model.encode(text)
    return torch.tensor(emb)

# Save embeddings for each document
print("Creating embeddings for documents...")
document_embeddings = []
for doc in tqdm(documents):
    emb = preprocess(doc['text'])
    document_embeddings.append({
        'doc_id': doc['doc_id'],
        'embedding': emb
    })
print("Embeddings created for all documents.")

# Download Embedding
embedding_data = {
    doc["doc_id"]: doc["embedding"].cpu() for doc in document_embeddings
}

torch.save(embedding_data, 'doc_embeddings.pt')

# Define Generator Model (Large Language Model)
# Load a Hugging Face causal language model for generating responses based on retrieved context.
model_name = "microsoft/Phi-4-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
generator_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto', torch_dtype=torch.float16)

def generate_response(context, question):
    """
    Generate a response based on the context and question using the language model.
    """
    prompt = f"""<|system|>
    You are an expert medical assistant with access to a comprehensive database of medical documents. 
    Your role is to provide precise, concise, and medically accurate answers based solely on the provided context. 
    Use medical terminology appropriately. If a medical term is used in the context without definition, 
    you may provide a brief explanation based on standard medical knowledge, 
    but only if you are confident in its accuracy. Respond in a professional and informative tone, 
    as if you are communicating with a patient or a healthcare professional.
    <|user|>
    Using only the information in the context below, provide a concise and accurate answer to the following question. 
    Focus on the most relevant parts of the context that directly address the question. 
    If the context does not contain the necessary information, state that clearly.

    Context:
    {context}

    Question:
    {question}

    <|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(generator_model.device)
    outputs = generator_model.generate(
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
    
# Main RAG Pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
document_embeddings = torch.load("doc_embeddings.pt")
doc_lookup = {doc["doc_id"]: doc["text"] for doc in documents}

def cosine_similarity(tensor1, tensor2):
    tensor1 = F.normalize(tensor1.unsqueeze(0))
    tensor2 = F.normalize(tensor2.unsqueeze(0))
    return torch.mm(tensor1, tensor2.T).item()

print("Retrieving top 5 documents for each question...")
results = []

for question in tqdm(questions):
    question_embedding = preprocess(question['question'])
    
    # Calculate cosine similarity between the question embedding and document embeddings
    similarities = []
    for doc_id, doc_embedding in document_embeddings.items():
        sim = cosine_similarity(question_embedding.to(device), doc_embedding.to(device))
        similarities.append((doc_id, sim))
    
    # Sort documents by similarity score and retrieve top 5
    top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    
    # Generate response using the retrieved documents
    context = "\n\n".join([doc_lookup[doc_id] for doc_id, _ in top_docs[:3] if doc_id in doc_lookup])
    response = generate_response(context, question['question'])
    
    results.append({
        'question_id': question['question_id'],
        'retrieved_docs': [doc_id for doc_id, _ in top_docs],
        'answer': response
        
    })
print("RAG Pipeline completed. Responses generated for all questions.")

# Save results to a JSON file
with open('pred.json', 'w', encoding='utf-8') as file:
    json.dump(results, file, indent=2, ensure_ascii=False)
print("Results saved to pred.json")
