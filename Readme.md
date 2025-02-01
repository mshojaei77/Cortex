# RAG System: Retrieval-Augmented Generation Application

The RAG System is a sophisticated application designed for document retrieval and answer generation. It combines state-of-the-art vector search techniques with large language models to create precise, evidence-based responses from a curated collection of PDF documents. This README focuses on the key features and functionalities of the app.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Components](#architecture--components)
- [Advanced Configuration](#advanced-configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview
The RAG System processes a directory of PDF documents by:
- Extracting and chunking text content.
- Caching embeddings for future queries.
- Indexing vectors into a Milvus vector database.
- Executing advanced similarity searches.
- Generating answers using an LLM with citation support based on document context.

This seamless integration of document processing, vector indexing, and answer generation makes the RAG System a powerful tool for knowledge extraction and retrieval.

## Key Features

- **PDF Processing and Caching**  
  - **Document Parsing:** Uses PDF loaders to extract text from PDFs.
  - **Text Chunking:** Splits documents into manageable chunks using configurable sizes and overlaps, ensuring context preservation.
  - **Caching Mechanism:** Computes and caches MD5 hashes, embeddings, and text chunks to avoid reprocessing, improving efficiency.

- **Advanced Vector Search with Milvus**  
  - **Milvus Integration:** Connects to a Milvus vector database to store embeddings and perform vector similarity searches.
  - **Customizable Schema:** Dynamically creates a collection schema including document metadata such as source and page numbers.
  - **Enhanced Search:** Supports candidate pooling and re-ranking using methods like cross-encoder to ensure high-quality, relevant document retrieval.
  - **Normalization and Query Expansion:** Provides options to normalize embeddings and expand queries to capture broader context.

- **Answer Generation Using LLM**  
  - **LLM Integration:** Uses OpenAI models (e.g., gpt-4o-mini) to generate context-based answers.
  - **Structured Output:** Formats answers with clear citations, linking responses back to original documents and specific page numbers.
  - **System Prompts:** Incorporates advanced system prompts to ensure that the LLM adheres to strict academic and evidence-based response guidelines.

- **Configurable and Scalable Architecture**  
  - **Modular Design:** Separates document processing, vector management, and LLM answer generation into distinct, clearly defined classes.
  - **Flexible Settings:** Allows customization of settings such as chunk sizes, overlapping regions, embedding dimensions, search and re-ranking parameters, and many more via a comprehensive configuration class (`RAGSettings`).
  - **Advanced Retrieval Options:** Supports hybrid search methods, multi-query generation, and aggregation techniques for enhanced retrieval accuracy.

## Architecture & Components
- **DocumentProcessor**  
  Handles loading PDFs, splitting texts, computing embeddings, and caching intermediate results to boost performance.

- **MilvusManager**  
  Manages all Milvus operations including connecting to the database, creating or using document collections, inserting documents, and performing similarity searches with citation enrichment.

- **LLMAnswerGenerator**  
  Orchestrates interactions with the OpenAI API to generate precise answers based on context enriched with document citations.

- **RAGSystem**  
  Serves as the orchestration layer that ties all components together. It manages the end-to-end workflow from PDF processing through vector search to answer generation.

## Advanced Configuration
- **Enhanced Retrieval and Re-ranking:**  
  Customize parameters such as candidate multipliers, re-rank thresholds, and cross-encoder settings to fine-tune the quality of retrieval and answer generation.

- **Query Expansion:**  
  Enable transformer-based query expansion to enrich the search queries with broader context.

- **Hybrid Search:**  
  Combine dense and sparse retrieval strategies, weighting dense retrieval while still considering sparse methods for improved candidate selection.

- **Chunking & Preprocessing Enhancements:**  
  Fine-tune advanced chunking parameters like sentence boundary detection, dynamic chunk sizing, and custom pre-processing routines for optimal document understanding.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with the improvements or fixes. When contributing:
- Ensure all new features are well-documented.
- Write or update tests.
- Follow the existing project code style.

## License
This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.