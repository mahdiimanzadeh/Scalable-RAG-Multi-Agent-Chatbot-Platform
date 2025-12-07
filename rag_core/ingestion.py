# --- Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# --- Configuration ---
QDRANT_URL = "http://qdrant:6333"
# The path to the model inside the container, mounted via docker-compose.yml
EMBEDDING_MODEL_PATH = "/app/models/bge-small-en-v1.5"


# --- Main Ingestion Function ---
def run_ingestion(docs_path: str, client: QdrantClient):
    """
    Loads, splits, and embeds documents, then stores them in Qdrant.
    """
    print(f"Starting ingestion process from directory: {docs_path}")
    
    # 1. Load and split documents
    # Configure the loader to handle both .txt and .csv files
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.*",  # Load all files
        show_progress=True,
        use_multithreading=True,
        loader_map={
            ".txt": TextLoader,
            ".csv": CSVLoader(source_column="utterance"),
        }
    ) 
    
    try:
        documents = loader.load()
    except Exception as e:
        print(f"❌ Error loading documents from {docs_path}: {e}")
        # If FileNotFoundError, it means the data volume mount has an issue
        raise

    if not documents:
        print("⚠️ No documents found. Ensure 'test.txt' is correctly mounted in the 'data' directory.")
        return

    print(f"Loaded {len(documents)} source document(s).")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Splitted into {len(chunks)} chunks.")
    
    # 2. Create Embeddings
    print("Generating Embeddings using local BGE-small-en-v1.5...")
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cpu'}
    )
    
    # 3. Store in Qdrant
    collection_name = "knowledge_base"

    # Connect and store
    Qdrant.from_documents(
        chunks,
        embedding_model,
        url=QDRANT_URL,
        collection_name=collection_name
    )
    
    print("-" * 50)
    print(f"✅ Success: {len(chunks)} vectors stored in Qdrant collection: '{collection_name}'")
    print("-" * 50)


# --- Execution ---
if __name__ == "__main__":
    # This part is for standalone testing of the ingestion script
    client = QdrantClient(url=QDRANT_URL)
    # Use the path inside the container as defined in docker-compose.yml
    run_ingestion("/app/data", client)