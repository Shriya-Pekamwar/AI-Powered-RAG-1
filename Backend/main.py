import os
import io
import logging
import json
import tempfile
import fitz  # PyMuPDF
import base64
import boto3
import openai
import chromadb
import re
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Form,  Query
from fastapi.responses import JSONResponse
from mistralai import Mistral
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chunking_evaluation.chunking import (
    FixedTokenChunker,
    RecursiveTokenChunker,
    KamradtModifiedChunker
)
from chunking_evaluation.utils import openai_token_count
from dotenv import load_dotenv
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
 
load_dotenv()  # Load environment variables from .env file
 
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
# AWS S3 Configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_PREFIX = "pdf_documents"
 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
pc = Pinecone(api_key=PINECONE_API_KEY)
 
# ChromaDB configuration
COLLECTION_NAME = "rag_documents"
PERSIST_DIRECTORY = "./local_chromadb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize SentenceTransformer model globally
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"SentenceTransformer model '{EMBEDDING_MODEL}' loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    embedding_model = None
# Initialize FastAPI
app = FastAPI()
 
# Initialize Mistral Client
API_KEY = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY)

class QueryPayload(BaseModel):
    question: str
    vector_db: str
    quarters: list

class ProcessChunkPayload(BaseModel):
    filename: str
    chunking_strategy: str
    vector_db: str
    quarter: str
 
# Ensure S3 Client Setup
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=S3_REGION,
    )
 
def extract_text_pymupdf(pdf_file_io: BytesIO):
    try:
        doc = fitz.open(stream=pdf_file_io, filetype="pdf")
        text_data = ""
        images = []
        tables = []
 
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text_data += f"### Page {page_num + 1}\n\n"
            text_data += page.get_text("text") + "\n\n"
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(BytesIO(image_bytes))
                img_filename = f"image_{page_num + 1}_{img_index + 1}.png"
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images.append({
                    "filename": img_filename,
                    "base64": img_base64
                })
            
            table = page.get_text("dict")
            tables.append(table)
        
        markdown_content = "# Extracted Data from PDF\n\n" + text_data + "\n"
        
        markdown_content += "## Extracted Images\n"
        for img in images:
            markdown_content += f"![{img['filename']}](data:image/png;base64,{img['base64']})\n"
        
        markdown_content += "\n## Extracted Tables\n"
        for table in tables:
            markdown_content += "### Table\n"
            for block in table["blocks"]:
                if block['type'] == 0:
                    for line in block["lines"]:
                        line_text = " | ".join([span["text"] for span in line["spans"]])
                        markdown_content += line_text + "\n"
        
        return markdown_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
def process_pdf_with_mistral(uploaded_pdf):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            temp_pdf_path = Path(tmp_file.name)
            temp_pdf_path.write_bytes(uploaded_pdf)
 
        uploaded_file = client.files.upload(
            file={"file_name": temp_pdf_path.stem, "content": temp_pdf_path.read_bytes()},
            purpose="ocr",
        )
 
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model="mistral-ocr-latest",
            include_image_base64=True
        )
        return json.loads(pdf_response.model_dump_json())
    except Exception as e:
        return {"error": str(e)}
 
def process_pdf_with_docling(input_pdf_path: Path):
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        conv_res = doc_converter.convert(input_pdf_path)
        if conv_res is None:
            return None
        return conv_res.document.save_as_markdown()
    except Exception as e:
        raise Exception(f"Docling error: {str(e)}")


 
def upload_to_s3(content: str, filename: str):
    s3_client = get_s3_client()
    
    # Debugging logs
    logger.info(f"Uploading {filename} to S3 Bucket: {S3_BUCKET_NAME} in Region: {S3_REGION}")
    
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME is None. Check your environment variables.")
        raise Exception("S3_BUCKET_NAME is not set. Please check your environment variables.")
 
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=f"{S3_PREFIX}/{filename}",
            Body=content.encode("utf-8"),
            ContentType="text/markdown",
        )
        logger.info(f"Successfully uploaded {filename} to S3.")
        return filename
    except Exception as e:
        logger.error(f"Failed to upload {filename} to S3: {str(e)}")
        raise Exception(f"Failed to upload {filename} to S3: {str(e)}")
 
    
@app.post("/process_pdf")
async def process_pdf(file: UploadFile = File(...), parser: str = Form(...)):
    try:
        pdf_bytes = file.file.read()
        if parser == "pymupdf":
            markdown_content = extract_text_pymupdf(BytesIO(pdf_bytes))
        elif parser == "mistral_ocr":
            extracted_data = process_pdf_with_mistral(pdf_bytes)
            markdown_content = extracted_data.get("pages", [{}])[0].get("markdown", "")
        elif parser == "docling":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_bytes)
                pdf_path = Path(tmp_file.name)
            markdown_content = process_pdf_with_docling(pdf_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid processing method")
 
        if not markdown_content:
            logger.error("Markdown content is None. Extraction failed.")
            raise HTTPException(status_code=500, detail="Failed to generate markdown content")
 
        filename = f"{file.filename}.md"
        logger.info(f"Uploading {filename} to S3...")
        upload_to_s3(markdown_content, filename)
        logger.info(f"Successfully uploaded {filename} to S3.")
 
        return {"message": "File processed and uploaded successfully", "filename": filename}
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
 
 
def create_sentence_transformer_embeddings(chunks):
    """Creates embeddings for the given chunks using SentenceTransformer."""
    global embedding_model
    if embedding_model is None:
        raise ValueError("Embedding model not initialized.")

    texts = [chunk['text'] for chunk in chunks]
    
    # Convert tensor directly to NumPy
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings)
 
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "embedding": embeddings[i].tolist(),  # Ensure this is a list
            "metadata": chunk["metadata"],
            "text": chunk["text"][:300]  # Truncate text
        })
 
    return result
 
 
 
def fetch_files_from_s3():
    s3_client = get_s3_client()
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
    return [obj["Key"].split("/")[-1] for obj in response.get("Contents", [])]
 
def chunk_Recursive_document(text):
    """
    Split document text into chunks using RecursiveTokenChunker.
    
    Args:
        text (str): The document text to chunk
        
    Returns:
        list: List of text chunks
    """
    # Initialize the chunker with our configured settings
    chunker = RecursiveTokenChunker(
        chunk_size=400,
        chunk_overlap=200,
        length_function=openai_token_count,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    
    # Split the text into chunks
    chunks = chunker.split_text(text)
    
    # Return the chunks
    return chunks
 
def chunk_Token_document(text):
    """
    Split document text into chunks using RecursiveTokenChunker.
    
    Args:
        text (str): The document text to chunk
        
    Returns:
        list: List of text chunks
    """
    # Initialize the chunker with our configured settings
    fixed_token_chunker = FixedTokenChunker(
    chunk_size=400,  # 400 tokens per chunk
    chunk_overlap=0,  # No overlap
    encoding_name="cl100k_base"  # Use OpenAI's cl100k tokenizer
)
    
    # Split the text into chunks
    chunks = fixed_token_chunker.split_text(text)
    
    # Return the chunks
    return chunks
 
def chunk_Semantics_document(text):
    """
    Split document text into chunks using RecursiveTokenChunker.
    
    Args:
        text (str): The document text to chunk
        
    Returns:
        list: List of text chunks
    """
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    # Initialize the chunker with our configured settings
    kamradt_chunker = KamradtModifiedChunker(
        avg_chunk_size=300,      # Target size in tokens
        min_chunk_size=50,       # Initial split size
        embedding_function=embedding_function  # Pass your embedding function
    )
    
    # Split the text into chunks
    chunks = kamradt_chunker.split_text(text)
    
    # Return the chunks
    return chunks
 
def chunk_data(uploaded_files, chunking_strategy):
    s3_client = get_s3_client()
    all_chunks = []
    for file_key in uploaded_files:
        try:
            file_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX}/{file_key}")
            text_content = file_content['Body'].read().decode('utf-8')
            filename = file_key.replace('.md', '')
 
            if chunking_strategy == "Recursive":
                chunks = chunk_Recursive_document(text_content)
            elif chunking_strategy == "Token":
                chunks = chunk_Token_document(text_content)
            else:
                chunks = chunk_Semantics_document(text_content)
 
            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "filename": filename,
                        "chunk_type": chunking_strategy,
                        "chunk_index": j
                    }
                })
        except Exception as e:
            print(f"Error processing {file_key}: {e}")
 
    return all_chunks


def get_or_create_collection():
    """
    Get or create a ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
        print(f"Using existing collection: {COLLECTION_NAME}")

    except:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {COLLECTION_NAME}")

    return collection


def store_in_chromadb(all_chunks):
    """
    Store embeddings in ChromaDB collection.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
 
    # Convert embeddings from tensor to a numpy array (or list of lists) before storing
    embeddings = embeddings.cpu().numpy()  # Convert tensor to numpy array
 
    collection = get_or_create_collection()
 
    ids = [
        f"chunk_{chunk['metadata']['filename']}_{chunk['metadata']['chunk_type']}_{chunk['metadata']['chunk_index']}"
        for chunk in all_chunks
    ]
 
    metadatas = [
        {"filename": chunk['metadata']['filename'], "chunk_type": chunk['metadata']['chunk_type'], "chunk_index": chunk['metadata']['chunk_index']}
        for chunk in all_chunks
    ]
 
    collection.add(
        embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
    )
 
    count = collection.count()
    print(f"Total documents in collection: {count}")
 
    return f"Stored {count} embeddings in ChromaDB"


def store_in_pinecone(embeddings):
    print("⚡ Storing embeddings in Pinecone")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists, if not create it
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # Ensure this matches your embedding size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )

    index = pc.Index(PINECONE_INDEX_NAME)

    batch_size = 50  # Reduce batch size to avoid exceeding 4MB payload
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]

        # Prepare vectors for Pinecone upsert
        vectors = [
            (
                f"{emb['metadata']['filename']}_{emb['metadata']['chunk_type']}_{emb['metadata']['chunk_index']}",
                emb['embedding'],
                {**emb['metadata'], 'text': emb['text'][:300]}  # Truncate long text
            )
            for emb in batch
        ]

        # Filter out zero vectors
        vectors = filter_zero_vectors(vectors)

        if vectors:
            try:
                index.upsert(vectors=vectors)
                print(f"✅ Uploaded batch of {len(vectors)} vectors to Pinecone")
            except Exception as e:
                print(f"❌ Error uploading batch to Pinecone: {e}")
        else:
            print("⚠️ No valid vectors to upload in this batch.")

    return "📦 Stored embeddings in Pinecone"


def filter_zero_vectors(vectors):
    """
    Filters out vectors that contain only zeros.
    """
    non_zero_vectors = []
    for vector in vectors:
        if np.any(np.array(vector[1]) != 0):  # Check if any element in the vector is non-zero
            non_zero_vectors.append(vector)
        else:
            print(f"Vector {vector[0]} contains only zeros and will not be upserted.")  # Log for debugging
    return non_zero_vectors

def create_query_vector(question: str) -> np.ndarray:
    """
    Creates an embedding vector for the given question using SentenceTransformer.

    Args:
        question (str): The question to embed.

    Returns:
        np.ndarray: The embedding vector of the question.
    """
    global embedding_model  # Use the global instance
    if embedding_model is None:
        raise ValueError("Embedding model not initialized.")

    try:
        embedding = embedding_model.encode(question)
        return embedding
    except Exception as e:
        logger.error(f"Error creating embedding for question: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating embedding: {str(e)}")

def query_pinecone(question: str, quarters: list):
    try:
        if PINECONE_API_KEY is None or PINECONE_INDEX_NAME is None:
            raise ValueError("Pinecone API key or index name not configured.")

        # Create the query vector
        query_vector = create_query_vector(question)

        # Ensure query_vector has the correct dimensionality (384)
        if len(query_vector) != 384:
            raise ValueError(f"Query vector must have a dimension of 384, but got {len(query_vector)}.")

        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)

        # Query Pinecone with filter
        results = index.query(
            vector=query_vector.tolist(),
            top_k=5,
            include_metadata=True,
            filter={"quarter": {"$in": quarters}}  # Filter by selected quarters
        )

        # Extract chunks from results
        retrieved_chunks = [match.metadata for match in results.matches]
        return retrieved_chunks

    except ValueError as ve:
        logger.error(f"Value error in query_pinecone: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying Pinecone: {str(e)}")


def query_chromadb(question: str, quarters: list):
    """
    Queries ChromaDB database with the given question and quarter filters.
    """
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured.")

        collection = get_or_create_collection()
        results = collection.query(
            query_texts=[question],
            n_results=5,
            where={"quarter": {"$in": quarters}}  # Filter by selected quarters
        )

        # Extract chunks from results
        retrieved_chunks = []
        for i in range(len(results['ids'][0])):
            retrieved_chunks.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return retrieved_chunks

    except ValueError as ve:
        logger.error(f"Value error in query_chromadb: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying ChromaDB: {str(e)}")


@app.post("/process_chunks")
async def process_chunks(payload: ProcessChunkPayload):
    """Process a file, chunk it, and store embeddings in a vector database."""
    try:
        filename = payload.filename
        chunking_strategy = payload.chunking_strategy
        vector_db = payload.vector_db
        quarter = payload.quarter

        # Validate inputs
        if not filename or not chunking_strategy or not vector_db or not quarter:
            raise HTTPException(status_code=400, detail="Missing required parameters.")

        uploaded_files = [filename]  # Wrap filename in a list
        chunks = chunk_data(uploaded_files, chunking_strategy)
        logger.info(f"Generated {len(chunks)} chunks from {filename} using {chunking_strategy} strategy.")

        # Add quarter to chunk metadata
        for chunk in chunks:
            chunk['metadata']['quarter'] = quarter  # Add quarter to metadata

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated.")

        # Create embeddings for chunks
        embeddings = create_sentence_transformer_embeddings(chunks)
        logger.info(f"Created {len(embeddings)} embeddings.")

        if vector_db == "pinecone":
            storage_result = store_in_pinecone(embeddings)
        else:
            storage_result = store_in_chromadb(embeddings)

        logger.info(storage_result)
        return {"message": "File processed and embeddings stored successfully!"}

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    """List available files in the S3 bucket."""
    try:
        files = fetch_files_from_s3()
        return {"files": files}
    except Exception as e:
        logger.error(f"Error fetching files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_documents(payload: QueryPayload):
    """Query stored embeddings and return an answer from OpenAI."""
    try:
        question = payload.question
        vector_db = payload.vector_db
        quarters = payload.quarters

        # Now handle the query based on vector_db
        if vector_db == "pinecone":
            retrieved_chunks = query_pinecone(question, quarters)
        else:
            retrieved_chunks = query_chromadb(question, quarters)

        if not retrieved_chunks:
            # Return a default context if no chunks are found
            return {"answer": "No relevant information found.", "context": []}

        # Prepare context for OpenAI
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])

        # Query OpenAI for an answer
        prompt = f"Answer the following question based on the context provided:\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
        response = openai.Completion.create(
            engine="text-davinci-003",  # Or any other suitable engine
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

        answer = response.choices[0].text.strip()

        # Ensure "context" is always returned, even if empty
        return {"answer": answer, "context": [chunk['text'] for chunk in retrieved_chunks]}

    except Exception as e:
        logger.error(f"Error querying documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))