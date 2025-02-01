import os
import logging
import hashlib
import numpy as np
from dataclasses import dataclass, field
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class RAGSettings:
    # Document and cache paths:
    pdf_dir: str = "backend/data/documents"
    embedding_cache_dir: str = "backend/data/embeddings"
    
    # Text splitting and pre-processing settings:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    custom_text_splitter: object = None          # Optionally override the text splitter.
    apply_text_normalization: bool = True          # Normalize texts before splitting.
    lower_case: bool = True                        # Convert text to lower-case.
    remove_stopwords: bool = False                 # Whether to remove stop words.
    custom_text_preprocessor: object = None        # Option to pass a custom text preprocessor.
    
    # Milvus (vector store) settings:
    collection_name: str = "document_store"
    host: str = "localhost"
    port: str = "19530"
    milvus_index_params: dict = field(default_factory=lambda: {
        "metric_type": "L2",                      # Use 'L2' or 'cosine' based on your similarity measure.
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    })
    milvus_search_params: dict = field(default_factory=lambda: {
        "nprobe": 20                            # Increased for broader search coverage.
    })
    search_limit: int = 10
    top_k: int = 3                               # How many candidates after search, post re-ranking.
    
    # LLM (Answer Generator) settings:
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0                 # Temperature for creativity control.
    llm_top_p: float = 1.0                       # Controls diversity via nucleus sampling.
    llm_max_tokens: int = 1024                   # Maximum tokens for LLM responses.
    llm_frequency_penalty: float = 0.0           # Penalizes repeated tokens.
    llm_presence_penalty: float = 0.0            # Penalizes tokens based on their presence so far.
    
    # Advanced retrieval and re-ranking options:
    re_ranking: bool = True                      # Enable re-ranking of retrieved candidates.
    re_rank_method: str = "cross-encoder"        # Options: "cross-encoder", "bm25", etc.
    re_rank_threshold: float = 0.5               # Filtering threshold for candidate quality.
    candidate_multiplier: int = 2                # Retrieve extra candidates for later re-ranking.
    custom_similarity_metric: str = "cosine"     # Options: "cosine", "L2", etc.
    normalize_embeddings: bool = True            # Normalize embeddings for consistent similarity comparisons.
    
    # Query expansion options:
    query_expansion: bool = True                 # Enable query expansion.
    query_expansion_method: str = "transformer-based"  # Options: "simple", "transformer-based", etc.
    query_expansion_context: str = "with detailed explanation"  # Additional context added to query.
    
    # Multi-query/aggregation options:
    multi_query: bool = False                    # Enable multi-query generation.
    multi_query_count: int = 3                   # Number of query variations to generate.
    embedding_aggregation_method: str = "mean"   # How to aggregate multiple embeddings (mean, max, concat).
    
    # Embedding inference settings:
    embedding_inference_batch_size: int = 16     # Controls how many texts to embed at once.
    
    # Debug and verbosity:
    debug_verbose: bool = False                  # Enable detailed debug logging.

    # --- Additional options for enhanced accuracy ---

    # Retrieval Quality Enhancements
    enable_hybrid_search: bool = False           # Combine dense and sparse retrieval strategies.
    sparse_retrieval_method: str = "BM25"          # Options include BM25, TF-IDF, etc.
    hybrid_retrieval_weight: float = 0.5           # Balance factor between dense and sparse scores.
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Model for re-ranking.
    cross_encoder_threshold: float = 0.5           # Threshold to filter low scoring candidates.
    metadata_filtering_enabled: bool = False       # Enable filtering based on document metadata.
    metadata_filter_fields: list = field(default_factory=lambda: ["date", "source"])  # Fields to filter by.

    # Vectorization and Embedding Improvements
    fine_tune_embeddings: bool = False           # Option to fine-tune embedding model parameters.
    domain_specific_finetune: bool = False       # Fine-tune on domain-specific data.
    custom_embedding_dimension: int = 768        # Increase precision via higher embedding dimensions.

    # Contextual Augmentation Options
    query_expansion_variants: int = 3             # Number of variants to generate during query expansion.
    multi_source_integration: bool = False         # Aggregate data from diverse sources.
    integrated_sources: list = field(default_factory=list) # List of additional data source identifiers.
    few_shot_prompts: list = field(default_factory=list)     # Include few-shot examples to guide the LLM.
    contrastive_in_context: bool = False         # Use contrastive in-context learning to refine results.

    # Advanced Chunking and Preprocessing
    enable_advanced_chunking: bool = False        # Use advanced techniques for text chunking.
    sentence_boundary_detection: bool = False      # Detect sentence boundaries to improve chunk quality.
    dynamic_chunk_size: bool = False              # Dynamically adjust chunk sizes based on document type.
    chunk_size_strategy: str = "fixed"            # Options: "fixed" or "adaptive"
    custom_chunk_strategy: str = ""               # Placeholder for a custom chunking strategy identifier.

    # Emerging Techniques and Monitoring
    enable_blended_retrieval: bool = False        # Combine dense and sparse retrieval with hybrid queries.
    enable_focus_mode: bool = False               # Retrieve context at the sentence level to reduce noise.
    enable_agent_validation: bool = False         # Integrate agent-based workflows for response validation.
    agent_validation_endpoint: str = ""           # API endpoint for external validation services.
    enable_metrics_monitoring: bool = False        # Turn on continuous performance monitoring.
    monitoring_metrics: dict = field(default_factory=lambda: {
        "MAP": True, "MRR": True, "Precision": True
    })  # Metrics to monitor performance.
    continuous_monitoring_interval: int = 300     # Interval (seconds) for monitoring updates.

class DocumentProcessor:
    """
    Handles loading PDFs, splitting into chunks, computing embeddings and caching results.
    """
    def __init__(self, 
                 settings: RAGSettings,
                 text_splitter=None,
                 embeddings=None,
                 logger=logger):
        self.pdf_dir = settings.pdf_dir
        self.embedding_cache_dir = settings.embedding_cache_dir
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
        )
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.logger = logger
        self.new_documents = []  # List of Document objects
        self.new_vectors = []    # List of embedding vectors
        self.new_texts = []      # List of text chunks

    @staticmethod
    def get_file_hash(file_path):
        """
        Compute an MD5 hash for the file to use as a cache key.
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def process_pdfs(self, existing_docs=None):
        """
        Process PDFs in the specified directory.
        If `existing_docs` is provided, skip PDF files whose paths already exist.
        """
        self.logger.debug(f"Starting PDF document processing from directory: {self.pdf_dir}")
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"Directory {self.pdf_dir} does not exist")
        
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.pdf_dir}")
            raise ValueError(f"No PDF files found in {self.pdf_dir}")
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        processed_count = 0

        for filename in pdf_files:
            pdf_path = os.path.join(self.pdf_dir, filename)
            # Skip if this PDF was processed before.
            if existing_docs is not None and pdf_path in existing_docs:
                self.logger.debug(f"Skipping already processed PDF: {filename}")
                continue

            self.logger.info(f"Processing PDF: {filename}")
            file_hash = self.get_file_hash(pdf_path)
            embedding_cache_path = os.path.join(self.embedding_cache_dir, f"{file_hash}_embeddings.npy")
            text_cache_path = os.path.join(self.embedding_cache_dir, f"{file_hash}_texts.npy")

            if os.path.exists(embedding_cache_path) and os.path.exists(text_cache_path):
                self.logger.debug(f"Loading cached embeddings for {filename}")
                file_vectors = np.load(embedding_cache_path, allow_pickle=True)
                file_texts = np.load(text_cache_path, allow_pickle=True)
                for text in file_texts:
                    self.new_documents.append(Document(page_content=text, metadata={'source': pdf_path}))
                self.new_vectors.extend(file_vectors.tolist() if isinstance(file_vectors, np.ndarray) else file_vectors)
                self.new_texts.extend(file_texts.tolist() if isinstance(file_texts, np.ndarray) else file_texts)
                processed_count += 1
                continue

            try:
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                pdf_chunks = self.text_splitter.split_documents(pdf_docs)
                file_vectors = []
                file_texts = []
                for chunk in pdf_chunks:
                    try:
                        vector = self.embeddings.embed_query(chunk.page_content)
                    except Exception as api_exception:
                        self.logger.error(f"Error embedding chunk from {filename}: {api_exception}")
                        continue
                    file_vectors.append(vector)
                    file_texts.append(chunk.page_content)
                    # Ensure page numbers start from 1 instead of 0
                    if 'page' in chunk.metadata:
                        # Add 1 to the page number since PDFs typically start at page 1
                        chunk.metadata['page'] = str(int(chunk.metadata.get('page', 0)))
                    else:
                        # If no page number is available, mark it as unknown
                        chunk.metadata['page'] = "unknown"
                    self.new_documents.append(chunk)

                # Cache embeddings and text chunks
                np.save(embedding_cache_path, np.array(file_vectors))
                np.save(text_cache_path, np.array(file_texts, dtype=object))
                self.new_vectors.extend(file_vectors)
                self.new_texts.extend(file_texts)
                self.logger.debug(f"Processed and cached embeddings for {filename}")
            except Exception as e:
                self.logger.error(f"Error processing PDF {filename}: {e}")
                continue

            processed_count += 1
        
        if not self.new_documents:
            msg = ("No documents were loaded successfully. Check if PDFs are readable and not corrupted."
                   if existing_docs is None else "No new documents to process - all PDFs have already been processed.")
            self.logger.warning(msg)
        
        self.logger.info(f"Processed {processed_count} PDF files")
        return processed_count


class MilvusManager:
    """
    Encapsulates Milvus operations:
      - Establishing connection,
      - Creating or loading collections
      - Inserting documents
      - Searching documents
    """
    def __init__(self, settings: RAGSettings, logger=logger):
        self.collection_name = settings.collection_name
        self.host = settings.host
        self.port = settings.port
        self.index_params = settings.milvus_index_params
        self.search_params = settings.milvus_search_params
        self.search_limit = settings.search_limit
        self.logger = logger
        self.settings = settings  # Use for additional accuracy settings
        self.connect()

    def connect(self):
        try:
            self.logger.debug("Attempting to connect to Milvus server")
            connections.connect(alias="default", host=self.host, port=self.port)
            self.logger.info("Connected to Milvus database")
        except Exception as e:
            self.logger.critical(f"Failed to connect to Milvus server: {e}")
            raise

    def disconnect(self):
        try:
            self.logger.debug("Initiating Milvus connection cleanup")
            connections.disconnect("default")
            self.logger.info("Closed database connection")
        except Exception as e:
            self.logger.error(f"Error disconnecting from Milvus: {e}")

    def get_existing_docs(self):
        """
        Returns a set of document source paths that have already been inserted.
        """
        if not utility.has_collection(self.collection_name):
            self.logger.debug("No existing collection found. All documents will be processed.")
            return None
        try:
            collection = Collection(name=self.collection_name)
            self.logger.debug("Loading existing collection to check for processed documents")
            collection.load()
            results = collection.query(
                expr="source != ''",
                output_fields=["source"],
                consistency_level="Strong",
                limit=10000
            )
            existing = {doc['source'] for doc in results}
            self.logger.debug(f"Found {len(existing)} existing documents in collection")
            return existing
        except Exception as e:
            self.logger.error(f"Error loading existing collection: {e}")
            raise

    def create_collection_schema(self, embedding_dim):
        """
        Create the collection schema using the embedding dimension,
        now including a page field to capture page numbers.
        """
        try:
            # If collection exists then use its schema
            if utility.has_collection(self.collection_name):
                collection = Collection(name=self.collection_name)
                schema = collection.schema
                for field in schema.fields:
                    if field.name == "embedding":
                        return schema

            self.logger.debug(f"Creating collection schema with embedding dimension {embedding_dim}")
            id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
            vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
            text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
            source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535)
            # New field to store page numbers (as a string for flexibility)
            page_field = FieldSchema(name="page", dtype=DataType.VARCHAR, max_length=10)

            schema = CollectionSchema(
                fields=[id_field, vector_field, text_field, source_field, page_field],
                description="Document store for RAG system with page number metadata"
            )
            self.logger.info("Created collection schema with page metadata")
            return schema
        except Exception as e:
            self.logger.error(f"Error creating collection schema: {e}")
            raise

    def create_or_use_collection(self, schema):
        """
        Create a new collection if not already exists; else use the existing one.
        """
        try:
            if not utility.has_collection(self.collection_name):
                self.logger.debug(f"Creating new collection with name: {self.collection_name}")
                collection = Collection(name=self.collection_name, schema=schema)
                collection.create_index(field_name="embedding", index_params=self.index_params)
                self.logger.info("Created new collection and index")
            else:
                self.logger.debug(f"Using existing collection: {self.collection_name}")
                collection = Collection(name=self.collection_name)
            return collection
        except Exception as e:
            self.logger.error(f"Error creating or retrieving collection: {e}")
            raise

    def insert_documents(self, collection, documents, vectors, texts):
        """
        Inserts newly processed documents into the Milvus collection.
        Now includes page numbers from each document's metadata.
        """
        try:
            if not documents:
                self.logger.info("No new documents to insert")
                return
            self.logger.debug("Preparing new data for insertion")
            sources = [doc.metadata.get('source', '') for doc in documents]
            # Get the page number from each document's metadata.
            pages = [str(doc.metadata.get('page', 'N/A')) for doc in documents]
            entities = [
                [np.array(v, dtype=np.float32) for v in vectors],  # embeddings
                texts,                                              # text content
                sources,                                            # source paths
                pages                                               # page numbers
            ]
            self.logger.debug(f"Inserting {len(texts)} new entities into collection")
            collection.insert(entities)
            collection.flush()
            self.logger.info("Inserted new documents into vector database")
        except Exception as e:
            self.logger.error(f"Error during document insertion: {e}")
            raise

    def load_collection(self, collection):
        """
        Loads the collection into memory.
        """
        try:
            self.logger.debug("Loading collection into memory")
            collection.load()
            self.logger.info("Loaded collection into memory")
        except Exception as e:
            self.logger.error(f"Error loading collection: {e}")
            raise

    def perform_query(self, query_str, collection, embeddings, top_k):
        """
        Performs a similarity search in the Milvus collection using the provided query string,
        and includes page numbers in the citation output.
        """
        try:
            self.logger.debug(f"Processing search query: '{query_str}'")
            # Optional query expansion
            expanded_query = query_str
            if self.settings.query_expansion:
                expanded_query = query_str + " additional context"

            query_vector = embeddings.embed_query(expanded_query)
            if self.settings.normalize_embeddings:
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm

            search_limit = self.search_limit
            if self.settings.re_ranking:
                search_limit *= self.settings.candidate_multiplier

            search_params = {
                "metric_type": self.index_params.get("metric_type", "L2"),
                "params": self.search_params
            }
            self.logger.debug("Executing vector similarity search")
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=search_limit,
                output_fields=["text", "source", "page"]
            )
            if not results:
                raise ValueError("Search returned no results")
            
            candidates = results[0]
            candidates.sort(key=lambda hit: hit.score)
            top_candidates = candidates[:top_k]
            
            # Retrieve context, source filename, and page numbers
            contexts = [hit.get('text') for hit in top_candidates]
            sources = [os.path.basename(hit.get('source')) for hit in top_candidates]
            pages = [hit.get('page') for hit in top_candidates]
            
            # Create citations by adding page numbers
            unique_sources = {}
            cited_contexts = []
            citation_number = 1
            for context, source, page in zip(contexts, sources, pages):
                # Create a unique key combining source and page
                key = f"{source}_pg{page}"
                if key not in unique_sources:
                    unique_sources[key] = f"[{citation_number}]"
                    citation_number += 1
                citation = unique_sources[key]
                cited_contexts.append(f"{context} {citation}")

            citations_text = "\nCitations:\n" + "\n".join(f"{v} {k.replace('_pg', ', pg. ')}" for k, v in unique_sources.items())
            
            self.logger.info("Retrieved relevant contexts from search")
            self.logger.debug(f"Search scores: {[hit.score for hit in top_candidates]}")
            
            prompt = f"""Answer the question based on the following contexts:

Contexts:
{' '.join(cited_contexts)}

{citations_text}

Question: {query_str}
Answer:"""
            return prompt
        except Exception as e:
            self.logger.error(f"Error during similarity search: {e}")
            raise


class LLMAnswerGenerator:
    """
    Uses the OpenAI API to generate answers given a context prompt.
    """
    def __init__(self, settings: RAGSettings, logger=logger):
        self.logger = logger
        self.llm_model = settings.llm_model

    def generate_answer(self, prompt):
        try:
            from openai import OpenAI
            client = OpenAI()
            self.logger.debug("Initializing OpenAI API request")
            
            system_prompt = (
                "You are a cutting-edge research assistant, an expert in retrieving and synthesizing information exclusively from the provided DOCUMENTS. "
                "While maintaining a warm and friendly academic tone, your role is to deliver precise, evidence-based answers. "
                "Follow these advanced guidelines strictly:\n\n"
                "1. ROLE & PURPOSE: Answer all queries solely using the information from the DOCUMENTS. If the DOCUMENTS do not provide enough detail, respond with 'I don't have enough information to provide an accurate answer.'\n\n"
                "2. CONTEXTUAL GROUNDING: Base your responses entirely on the provided DOCUMENTS without incorporating any external assumptions or prior knowledge.\n\n"
                "3. RESPONSE FORMAT: Structure your reply in exactly this format:\n"
                "   Answer: Your clear, concise response with citation numbers [1], [2], etc.\n\n"
                "   Citations:\n"
                "   [1] \"document_name.pdf\"\n"
                "   [2] \"another_document.pdf\"\n\n"
                "4. ERROR HANDLING: If the question is unrelated or the DOCUMENTS do not supply sufficient context, state 'This query is outside my current scope.'\n\n"
                "5. ALWAYS include the Citations section at the end of your response, listing all document names used.\n\n"
                "6. ACADEMIC TONE & FRIENDLINESS: Ensure your response is both technically precise and approachable."
            )
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            if not response or not response.choices:
                raise ValueError("No response received from OpenAI")
            answer = response.choices[0].message.content
            self.logger.info("Generated answer using LLM")
            print("\nQuestion:", prompt.split("Question:")[-1].split("Answer:")[0].strip())
            print("\nAnswer:", answer)
        except Exception as e:
            self.logger.error(f"Error in OpenAI API interaction: {e}")
            raise


class RAGSystem:
    """
    Orchestrates the overall workflow by coordinating the PDF processing, Milvus operations
    and answer generation.
    """
    def __init__(self, settings: RAGSettings = RAGSettings()):
        self.logger = logger
        self.settings = settings
        # Setup Document Processor with accuracy-related settings
        self.doc_processor = DocumentProcessor(
            settings=settings,
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size, 
                chunk_overlap=settings.chunk_overlap
            ),
            embeddings=OpenAIEmbeddings(),
            logger=logger
        )
        # Setup Milvus Manager with advanced settings
        self.milvus_manager = MilvusManager(
            settings=settings,
            logger=logger
        )
        # Setup LLM Answer Generator with settings for model choice etc.
        self.llm_generator = LLMAnswerGenerator(
            settings=settings,
            logger=logger
        )
        # New configuration: control the accuracy (i.e., number of candidates returned, re-ranking, etc.)
        self.top_k = settings.top_k

    def run(self, query_str="Please provide a comprehensive summary of the key points and main ideas from these documents"):
        try:
            # Retrieve existing document paths to avoid reprocessing
            existing_docs = self.milvus_manager.get_existing_docs()
            # Process PDFs (skip ones already in Milvus if available)
            self.doc_processor.process_pdfs(existing_docs=existing_docs)
            
            # Determine the embedding dimension
            if self.doc_processor.new_vectors:
                embedding_dim = len(self.doc_processor.new_vectors[0])
            elif existing_docs is not None:
                # Fallback: try to extract embedding dimension from the existing collection
                collection = Collection(name=self.milvus_manager.collection_name)
                for field in collection.schema.fields:
                    if field.name == "embedding":
                        embedding_dim = field.params.get('dim')
                        break
                else:
                    raise ValueError("Could not determine embedding dimension")
            else:
                raise ValueError("No embeddings available to determine dimension")
            
            # Create or load collection schema
            schema = self.milvus_manager.create_collection_schema(embedding_dim)
            collection = self.milvus_manager.create_or_use_collection(schema)
            
            # Insert new documents if any
            if self.doc_processor.new_documents:
                self.milvus_manager.insert_documents(
                    collection,
                    self.doc_processor.new_documents,
                    self.doc_processor.new_vectors,
                    self.doc_processor.new_texts
                )
            
            # Load the collection into memory
            self.milvus_manager.load_collection(collection)
            
            # Perform vector search query and generate observation prompt using accuracy settings
            prompt = self.milvus_manager.perform_query(
                query_str, 
                collection, 
                self.doc_processor.embeddings,
                top_k=self.top_k
            )
            # Generate answer using the LLM
            self.llm_generator.generate_answer(prompt)
        except Exception as e:
            self.logger.critical(f"Critical error in RAG system operation: {e}")
            raise
        finally:
            self.milvus_manager.disconnect()

if __name__ == '__main__':
    # Enhanced settings for better accuracy
    settings = RAGSettings(
        # Smaller chunks for more precise context
        chunk_size=500,
        chunk_overlap=150,  # Increased overlap to maintain context
        
        # More thorough vector search
        milvus_search_params={"nprobe": 50},  # Increased from 30 for wider search
        search_limit=25,    # Increased from 15 to consider more candidates
        top_k=5,           # Increased from 3 to get more diverse context
        
        # Model and retrieval settings
        llm_model="gpt-4o-mini",  
        re_ranking=True,
        re_rank_method="cross-encoder",
        re_rank_threshold=0.6,  # Slightly higher threshold for better quality
        
        # Enhanced similarity and embedding settings
        custom_similarity_metric="cosine",
        normalize_embeddings=True,
        candidate_multiplier=3,  # Increased from 2 for more candidates before re-ranking
        
        # Query enhancement
        query_expansion=True,
        query_expansion_method="transformer-based",
        
        # Additional accuracy improvements
        enable_hybrid_search=True,  # Enable both dense and sparse retrieval
        hybrid_retrieval_weight=0.7,  # Favor dense retrieval but consider sparse
        enable_advanced_chunking=True,
        sentence_boundary_detection=True,
        
        embedding_inference_batch_size=8  # Reduced for more careful processing
    )
    rag_system = RAGSystem(settings=settings)
    query_str = "What is Distillation?"  # Fixed typo in query string
    rag_system.run(query_str)  # Pass query_str as parameter
