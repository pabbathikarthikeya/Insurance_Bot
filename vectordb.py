from sentence_transformers import SentenceTransformer
import faiss
import pickle

def split_text_into_chunks(text, chunk_size=500):
    """
    Splits a long text into smaller chunks for better embedding and retrieval.
    
    Args:
        text (str): The full text to split.
        chunk_size (int): Approximate size of each chunk (in characters).
        
    Returns:
        list: List of text chunks.
    """
    chunks = []
    while len(text) > chunk_size:
        split_at = text[:chunk_size].rfind('.')
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[:split_at + 1].strip())
        text = text[split_at + 1:]
    if text:
        chunks.append(text.strip())
    return chunks

def create_vector_store(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Creates a FAISS vector store from text chunks using SentenceTransformer embeddings.
    
    Args:
        chunks (list): List of text chunks.
        model_name (str): Pretrained sentence transformer model.
        
    Returns:
        index, chunk_texts
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index, chunks

def save_vector_store(index, chunks, index_path="vector.index", chunks_path="chunks.pkl"):
    """
    Saves the FAISS index and chunks to disk.
    
    Args:
        index: FAISS index.
        chunks (list): List of chunks.
        index_path (str): Path to save FAISS index.
        chunks_path (str): Path to save chunks.
    """
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    # Load your extracted text
    with open("extracted_text.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    
    # 1. Split text into manageable chunks
    text_chunks = split_text_into_chunks(full_text)
    
    # 2. Create FAISS vector store
    index, chunks = create_vector_store(text_chunks)
    
    # 3. Save the index and chunks
    save_vector_store(index, chunks)
    
    print("✅ Vector database created and saved!")
