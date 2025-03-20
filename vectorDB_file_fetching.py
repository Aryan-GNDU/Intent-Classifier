import os
import dotenv
import logging
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pinecone import Pinecone, ServerlessSpec
import torch
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------
# Load environment variables (Pinecone API key etc.)
dotenv.load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "enter-the-region-name")

# --------------------
# File and index settings (edit these as needed)
PDF_FOLDER = "pdf"  # Folder with PDFs
INDEX_NAME = "hname_of_your_index"
EMBEDDING_DIM = 384  # Matches the all-MiniLM-L6-v2 model's output dimension

# --------------------
# 1. Load and split PDFs
logger.info(f"Loading PDFs from folder: {PDF_FOLDER}")
if not os.path.exists(PDF_FOLDER):
    logger.info(f"Creating PDF folder: {PDF_FOLDER}")
    os.makedirs(PDF_FOLDER)
    logger.warning(f"Please add your PDF files to the '{PDF_FOLDER}' directory and run the script again.")
    exit(0)

# Check if PDFs exist
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
if not pdf_files:
    logger.warning(f"No PDF files found in '{PDF_FOLDER}' directory. Please add PDFs and run again.")
    exit(0)

logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")

# Process each PDF file individually for better debugging
all_docs = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    logger.info(f"Processing PDF: {pdf_file}")
    
    try:
        # Load individual PDF
        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        logger.info(f"Loaded {len(pdf_docs)} pages from {pdf_file}")
        
        # Add better source metadata
        for doc in pdf_docs:
            doc.metadata["source"] = pdf_file
            doc.metadata["filename"] = pdf_file
        
        all_docs.extend(pdf_docs)
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_file}: {e}")

logger.info(f"Total pages loaded from all PDFs: {len(all_docs)}")

# Split documents
logger.info("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)
logger.info(f"Created {len(docs)} document chunks")

# --------------------
# 2. Initialize embeddings
logger.info("Initializing embeddings...")
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# --------------------
# 3. Setup Pinecone and upsert vectors
logger.info("Initializing Pinecone...")
pc = Pinecone(api_key=api_key)

# Check available indices
try:
    available_indexes = [index.name for index in pc.list_indexes()]
    logger.info(f"Available Pinecone indexes: {available_indexes}")
except Exception as e:
    logger.error(f"Error listing Pinecone indexes: {e}")
    available_indexes = []

# Create index if it does not exist
if INDEX_NAME not in available_indexes:
    logger.info(f"Creating Pinecone index: {INDEX_NAME}")
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region=PINECONE_API_ENV)
        )
        logger.info(f"Created Pinecone index: {INDEX_NAME}")
        time.sleep(10)
    except Exception as e:
        logger.error(f"Error creating Pinecone index: {e}")
        exit(1)
else:
    logger.info(f"Pinecone index '{INDEX_NAME}' already exists.")

# Connect to the index
try:
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    logger.info(f"Current index stats: {stats}")
except Exception as e:
    logger.error(f"Error connecting to Pinecone index: {e}")
    exit(1)

# --------------------
# 4. Initialize LangChain's Pinecone wrapper for retrieval
logger.info("Initializing LangChain Pinecone wrapper...")
docsearch = LC_Pinecone(index, embeddings, text_key="text")

# --------------------
# 5. Function to handle queries
def handle_query(query):
    """Handles user queries by retrieving relevant documents from Pinecone."""
    try:
        results = docsearch.similarity_search(query, k=1)
        
        if not results:
            return "No relevant content found in the indexed documents."
        
        response = "\nFound information in the following sources:\n"
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            content_snippet = doc.page_content[:1000] + "..." if len(doc.page_content) > 150 else doc.page_content
            
            response += f"\n{i}. File: {source}\n   Page: {page}\n   Content: {content_snippet}\n"
        
        return response.strip()
    
    except Exception as e:
        logger.error(f"Error searching for '{query}': {e}")
        return "An error occurred while processing your query. Please try again."
