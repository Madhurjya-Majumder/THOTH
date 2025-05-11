import time
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Load raw PDF(s)
DATA_PATH="data/"
def load_pdf_files(data):
    start_time = time.time()
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    elapsed_time = time.time() - start_time
    print(f"[Timing] PDF loading time: {elapsed_time:.4f} seconds")
    return documents

documents=load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ", len(documents))

# Step 2: Create Chunks
def create_chunks(extracted_data):
    start_time = time.time()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    elapsed_time = time.time() - start_time
    print(f"[Timing] Chunk creation time: {elapsed_time:.4f} seconds")
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    start_time = time.time()
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elapsed_time = time.time() - start_time
    print(f"[Timing] Embedding model loading time: {elapsed_time:.4f} seconds")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
start_time = time.time()
db.save_local(DB_FAISS_PATH)
elapsed_time = time.time() - start_time
print(f"[Timing] FAISS vectorstore saving time: {elapsed_time:.4f} seconds")

