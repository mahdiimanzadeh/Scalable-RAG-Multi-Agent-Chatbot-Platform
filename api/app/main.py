from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import uvicorn

# --- Configuration ---
QDRANT_URL = "http://qdrant:6333"
EMBEDDING_MODEL_NAME = "/app/models/bge-small-en-v1.5"
COLLECTION_NAME = "knowledge_base"

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Customer Support RAG API",
    description="An API for querying a customer support knowledge base.",
    version="1.0.0",
)

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str

class Document(BaseModel):
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    source_documents: list[Document]

# --- Global Components ---
# These are initialized on startup to avoid reloading on every request
embeddings = None
vector_store = None

# --- Application Events ---
@app.on_event("startup")
def startup_event():
    """
    Initialize and load the embedding model and vector store on startup.
    """
    global embeddings, vector_store
    
    print("--> Loading embedding model...")
    # Use a local model path if available, otherwise download
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    print("--> Embedding model loaded.")

    print("--> Connecting to Qdrant and initializing vector store...")
    client = QdrantClient(url=QDRANT_URL)
    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )
    print("--> Vector store initialized successfully.")

# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse)
def query_knowledge_base(request: QueryRequest):
    """
    Receives a question, searches the vector store for relevant documents,
    and returns them. This is the core RAG retrieval step.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store is not available.")

    print(f"--> Received query: {request.question}")
    
    # Perform the similarity search in the vector store
    retrieved_docs = vector_store.similarity_search(request.question, k=3)
    
    # For now, we will just combine the content of the retrieved documents.
    # In a full RAG pipeline, this context would be fed to an LLM.
    answer_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    print(f"--> Found {len(retrieved_docs)} relevant documents.")

    return QueryResponse(answer=answer_text, source_documents=retrieved_docs)

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Welcome to the RAG API. Navigate to /docs for API documentation."}

# --- Main Entry Point (for local debugging) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

