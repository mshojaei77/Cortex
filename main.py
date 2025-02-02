from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware  # Add CORS support
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from rag_system import RAGSystem, RAGSettings
import uvicorn
import os  # For file system operations
from datetime import datetime
import aiofiles
import uuid
import re
import subprocess
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add API metadata
app = FastAPI(
    title="Cortex MVP API",
    description="An MVP API for file uploads and basic document processing.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str
    
    @field_validator('query')
    @classmethod  # Required for field_validator
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is Distillation?"
            }
        }

class Response(BaseModel):
    answer: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# New request model for the chat (RAG) endpoint with additional parameters
class ChatRequest(BaseModel):
    query: str = Field(..., description="User query or question")
    history: Optional[List[Message]] = Field(
        default_factory=list, 
        description="Optional conversation history for context"
    )
    conversation_id: Optional[str] = Field(
        None, 
        description="Unique conversation identifier"
    )
    # System parameters
    temperature: Optional[float] = Field(
        None, 
        description="Override LLM temperature",
        ge=0.0,
        le=1.0
    )
    top_k: Optional[int] = Field(
        None, 
        description="Override number of top candidate contexts",
        ge=1,
        le=20
    )
    query_expansion: Optional[bool] = Field(
        None,
        description="Override whether to use query expansion"
    )
    re_ranking: Optional[bool] = Field(
        None,
        description="Override whether to perform re-ranking"
    )
    
    @field_validator('query')
    @classmethod  # Required for field_validator
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()

# New response model for the chat endpoint with extra details to return to the frontend
class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    prompt_used: str = Field(..., description="Complete prompt sent to LLM")
    citations: Dict[str, str] = Field(
        default_factory=dict, 
        description="Citation markers to sources mapping"
    )
    parameters_used: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters used for this response"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation identifier"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )

class UploadRequest(BaseModel):
    filename: str
    file_path: str
    file_url: str
    user_id: str
    file_size: int
    upload_date: str

class DeleteSourceRequest(BaseModel):
    filename: str
    source_id: str

class SourceResponse(BaseModel):
    source_id: str
    filename: str
    file_path: str
    upload_date: str
    file_size: int

def is_docker_running():
    """Check if Docker daemon is running"""
    try:
        subprocess.run(["docker", "info"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def start_docker_daemon():
    """Start Docker daemon with appropriate command based on OS"""
    try:
        if os.name == 'nt':  # Windows
            logger.info("Starting Docker Desktop on Windows...")
            # Path to Docker Desktop
            docker_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
            if not os.path.exists(docker_path):
                raise FileNotFoundError("Docker Desktop not found in standard location")
            
            # Start Docker Desktop
            subprocess.Popen([docker_path])
            
            # Wait for Docker to be ready (up to 60 seconds)
            for _ in range(60):
                if is_docker_running():
                    logger.info("Docker Desktop is ready!")
                    return
                time.sleep(1)
            raise TimeoutError("Docker Desktop failed to start within 60 seconds")
            
        else:  # Linux
            logger.info("Starting Docker daemon on Linux...")
            subprocess.run(["sudo", "systemctl", "start", "docker"], check=True)
            
            # Wait for Docker to be ready
            for _ in range(30):
                if is_docker_running():
                    logger.info("Docker daemon is ready!")
                    return
                time.sleep(1)
            raise TimeoutError("Docker daemon failed to start within 30 seconds")
            
    except Exception as e:
        logger.error(f"Failed to start Docker daemon: {e}")
        raise

def ensure_milvus_is_running():
    """Ensure Milvus server is running via Docker Compose"""
    try:
        # Check if Docker Compose file exists
        if not os.path.exists("docker-compose.yml"):
            raise FileNotFoundError("docker-compose.yml not found in current directory")
        
        # Ensure Docker is running
        if not is_docker_running():
            logger.info("Docker is not running. Attempting to start Docker...")
            start_docker_daemon()
        
        # Start Docker Compose
        logger.info("Starting Milvus via Docker Compose...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        
        # Wait for Milvus to be ready (up to 30 seconds)
        for _ in range(30):
            try:
                from pymilvus import connections
                connections.connect(host='localhost', port=19530)
                connections.disconnect()
                logger.info("Milvus is ready!")
                return
            except Exception:
                time.sleep(1)
        
        raise TimeoutError("Milvus failed to start within 30 seconds")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Docker Compose: {e}")
        raise
    except Exception as e:
        logger.error(f"Error ensuring Milvus is running: {e}")
        raise

# Initialize RAG system with enhanced settings
default_settings = RAGSettings(
    chunk_size=500,
    chunk_overlap=150,
    milvus_search_params={"nprobe": 50},
    search_limit=25,
    top_k=5,
    llm_model="gpt-4o-mini",
    re_ranking=True,
    re_rank_method="cross-encoder",
    re_rank_threshold=0.6,
    custom_similarity_metric="cosine",
    normalize_embeddings=True,
    candidate_multiplier=3,
    query_expansion=True,
    query_expansion_method="transformer-based",
    enable_hybrid_search=True,
    hybrid_retrieval_weight=0.7,
    enable_advanced_chunking=True,
    sentence_boundary_detection=True,
    embedding_inference_batch_size=8
)

# Ensure Milvus is running before initializing RAG system
ensure_milvus_is_running()

# Initialize RAG system
rag_system = RAGSystem(settings=default_settings)

@app.post("/query", 
    response_model=Response,
    summary="Basic query endpoint",
    description="Simple query endpoint for quick questions",
    response_description="Returns the answer with timestamp"
)
async def query_endpoint(query: Query):
    """
    Query the RAG system with a question.
    
    - **query**: The question you want to ask about the documents
    
    Returns:
    - **answer**: The generated answer based on the document context
    """
    try:
        answer = rag_system.run_query(query.query)
        return Response(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New chat endpoint for an enhanced chatbot RAG system with more control parameters.
@app.post("/chat", 
    response_model=ChatResponse,
    summary="Enhanced chat endpoint",
    description="Advanced chat endpoint with parameter control and detailed response",
    response_description="Returns comprehensive response with meta-information"
)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that accepts a query, optional conversation history, and various override parameters.
    
    - **query**: The main user query.
    - **history**: An optional list of previous conversation turns.
    - **conversation_id**: Unique conversation identifier.
    - **temperature**: (Optional) Override the default temperature for answer generation.
    - **top_k**: (Optional) Override the number of candidate contexts to consider.
    - **query_expansion**: (Optional) Whether to apply query expansion.
    - **re_ranking**: (Optional) Whether to perform candidate re-ranking.
    
    Returns:
    - **answer**: The final answer generated by the RAG system.
    - **prompt_used**: The complete processed prompt sent to the LLM.
    - **citations**: A mapping of citation markers to document names and pages.
    - **parameters_used**: The effective parameters used in generating the response.
    """
    try:
        # Backup original settings
        original_settings = {
            "temperature": rag_system.settings.llm_temperature,
            "top_k": rag_system.top_k,
            "query_expansion": rag_system.settings.query_expansion,
            "re_ranking": rag_system.settings.re_ranking
        }
        
        parameters_used = {}
        
        try:
            # Apply overrides if provided
            if request.temperature is not None:
                rag_system.settings.llm_temperature = request.temperature
                parameters_used["temperature"] = request.temperature
            else:
                parameters_used["temperature"] = original_settings["temperature"]

            if request.top_k is not None:
                rag_system.top_k = request.top_k
                parameters_used["top_k"] = request.top_k
            else:
                parameters_used["top_k"] = original_settings["top_k"]

            if request.query_expansion is not None:
                rag_system.settings.query_expansion = request.query_expansion
                parameters_used["query_expansion"] = request.query_expansion
            else:
                parameters_used["query_expansion"] = original_settings["query_expansion"]

            if request.re_ranking is not None:
                rag_system.settings.re_ranking = request.re_ranking
                parameters_used["re_ranking"] = request.re_ranking
            else:
                parameters_used["re_ranking"] = original_settings["re_ranking"]

            # Format conversation history
            conversation_prefix = ""
            if request.history:
                conversation_prefix = "Conversation History:\n" + "\n".join(
                    f"{msg.role}: {msg.content}" for msg in request.history
                ) + "\n\n"
            
            combined_query = conversation_prefix + "User Question: " + request.query

            # Process documents and prepare collection
            existing_docs = rag_system.milvus_manager.get_existing_docs()
            rag_system.doc_processor.process_pdfs(existing_docs=existing_docs)

            # Get embedding dimension
            embedding_dim = None
            if rag_system.doc_processor.new_vectors:
                embedding_dim = len(rag_system.doc_processor.new_vectors[0])
            else:
                from pymilvus import Collection
                collection = Collection(name=rag_system.milvus_manager.collection_name)
                for field in collection.schema.fields:
                    if field.name == "embedding":
                        embedding_dim = field.params.get('dim')
                        break
            
            if embedding_dim is None:
                raise ValueError("Could not determine embedding dimension")

            # Setup and use collection
            schema = rag_system.milvus_manager.create_collection_schema(embedding_dim)
            collection = rag_system.milvus_manager.create_or_use_collection(schema)
            
            if rag_system.doc_processor.new_documents:
                rag_system.milvus_manager.insert_documents(
                    collection,
                    rag_system.doc_processor.new_documents,
                    rag_system.doc_processor.new_vectors,
                    rag_system.doc_processor.new_texts
                )
            
            rag_system.milvus_manager.load_collection(collection)
            
            # Generate response
            prompt = rag_system.milvus_manager.perform_query(
                combined_query, 
                collection, 
                rag_system.doc_processor.embeddings,
                top_k=rag_system.top_k
            )
            
            answer = rag_system.llm_generator.generate_answer(prompt)
            
            # Extract citations
            citations = {}
            if "\nCitations:\n" in prompt:
                citations_block = prompt.split("\nCitations:\n")[-1]
                for line in citations_block.splitlines():
                    if line.strip():
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            citations[parts[0]] = parts[1].strip()

            return ChatResponse(
                answer=answer,
                prompt_used=prompt,
                citations=citations,
                parameters_used=parameters_used,
                conversation_id=request.conversation_id
            )

        finally:
            # Restore original settings
            rag_system.settings.llm_temperature = original_settings["temperature"]
            rag_system.top_k = original_settings["top_k"]
            rag_system.settings.query_expansion = original_settings["query_expansion"]
            rag_system.settings.re_ranking = original_settings["re_ranking"]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@app.post("/upload", status_code=201)
async def upload_file(request: UploadRequest):
    """
    Process a file that has been uploaded to Supabase storage.
    
    Instead of handling the file upload directly, this endpoint now receives
    the file metadata and processes the file from the Supabase URL.
    """
    try:
        # Here you would process the file from the Supabase URL
        # For example, downloading it temporarily for processing if needed
        
        # Generate a source ID for the frontend
        source_id = hash(f"{request.user_id}_{request.filename}_{request.upload_date}")
        
        return {
            "detail": f"File '{request.filename}' processed successfully.",
            "source_id": source_id,
            "file_path": request.file_path,
            "timestamp": request.upload_date
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

# Health check endpoint
@app.get("/health",
    summary="Health check",
    description="Check if the API is running",
    response_description="Returns OK if the API is running"
)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "version": app.version
    }

@app.delete("/delete-source")
async def delete_source(request: 
                        DeleteSourceRequest):
    """
    Handle deletion of a source.
    The actual file deletion is handled by the frontend in Supabase storage.
    This endpoint handles any additional cleanup needed in the backend.
    """
    try:
        # Perform any necessary cleanup in your backend systems
        # For example, removing entries from your vector store
        
        return {
            "detail": f"Source '{request.filename}' deleted successfully",
            "source_id": request.source_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting source: {str(e)}"
        )

@app.get("/sources/{user_id}", 
    response_model=List[SourceResponse],
    summary="Get user sources",
    description="Retrieve all sources for a specific user"
)
async def get_user_sources(user_id: str):
    """
    Get all sources for a specific user from the vector store.
    
    - **user_id**: The ID of the user whose sources to retrieve
    """
    try:
        # Get documents from Milvus collection for this user
        collection = rag_system.milvus_manager.get_collection()
        
        # Query to get all documents for this user
        expr = f'user_id == "{user_id}"'
        results = collection.query(
            expr=expr,
            output_fields=["source_id", "filename", "file_path", "upload_date", "file_size"]
        )
        
        return [
            SourceResponse(
                source_id=doc["source_id"],
                filename=doc["filename"],
                file_path=doc["file_path"],
                upload_date=doc["upload_date"],
                file_size=doc["file_size"]
            ) for doc in results
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sources: {str(e)}"
        )

if __name__ == "__main__":
    try:
        ensure_milvus_is_running()
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    finally:
        # Optionally, stop Docker Compose when the application exits
        try:
            subprocess.run(["docker-compose", "down"], check=True)
        except Exception as e:
            logger.error(f"Error stopping Docker Compose: {e}")