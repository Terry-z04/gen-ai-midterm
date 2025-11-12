#!/usr/bin/env python3
"""
Enhanced ChromaDB Loader with Fine-tuned Sentence Transformer Embeddings
Group 5 | UChicago MS-ADS RAG System

Features:
- Fine-tuned sentence-transformers model for domain-specific embeddings
- Batch processing for efficient embedding generation
- Enhanced metadata structure
- Progress tracking and error handling
"""

import json
import time
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import uuid
import os

# Import Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    print("⚠ sentence-transformers library not installed. Run: pip install sentence-transformers")
    ST_AVAILABLE = False


class FinetunedChromaLoader:
    """Load content into ChromaDB using fine-tuned sentence transformer embeddings"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db_finetuned",
        collection_name: str = "uchicago_msads_finetuned",
        embedding_model_path: str = "../finetune-embedding/exp_finetune",
        batch_size: int = 100
    ):
        """
        Initialize ChromaDB loader with fine-tuned sentence transformer embeddings
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name for the collection
            embedding_model_path: Path to fine-tuned model directory
            batch_size: Number of documents to process per batch
        """
        if not ST_AVAILABLE:
            raise ImportError("sentence-transformers library is required. Run: pip install sentence-transformers")
        
        # Resolve model path
        if not os.path.isabs(embedding_model_path):
            # Make path relative to this script's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            embedding_model_path = os.path.join(script_dir, embedding_model_path)
        
        if not os.path.exists(embedding_model_path):
            raise ValueError(f"Model path does not exist: {embedding_model_path}")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.embedding_model_path = embedding_model_path
        self.batch_size = batch_size
        self.collection = None
        
        # Initialize fine-tuned sentence transformer model
        print(f"Loading fine-tuned model from: {embedding_model_path}")
        self.model = SentenceTransformer(embedding_model_path)
        print(f"✓ Fine-tuned sentence transformer model loaded successfully")
        print(f"✓ Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def create_collection(self, drop_existing: bool = False):
        """Create ChromaDB collection"""
        if drop_existing:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"✓ Dropped existing collection: {self.collection_name}")
            except:
                pass
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "UChicago MS-ADS content with fine-tuned embeddings"}
        )
        print(f"✓ Collection ready: {self.collection_name}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from fine-tuned sentence transformer model
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use the fine-tuned model to encode texts
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=self.batch_size
            )
            return embeddings.tolist()
        except Exception as e:
            print(f"⚠ Error getting embeddings: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return chunks
    
    def flatten_documents(self, data: Dict[str, Any], parent_url: str = None) -> List[Dict[str, Any]]:
        """
        Recursively flatten nested document structure
        
        Args:
            data: Document data with potential subsections
            parent_url: Parent URL for tracking hierarchy
            
        Returns:
            List of flattened documents
        """
        documents = []
        
        doc = {
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "parent_url": data.get("parent_url") or parent_url or "",
            "category": data.get("category", "general"),
            "depth": data.get("depth", 0),
            "content": data.get("content", ""),
            "crawl_date": data.get("crawl_date", data.get("last_scraped", ""))
        }
        documents.append(doc)
        
        for subsection in data.get("subsections", []):
            documents.extend(self.flatten_documents(subsection, data.get("url")))
        
        return documents
    
    def prepare_data_for_insertion(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Load and prepare data from JSON file
        
        Args:
            json_file_path: Path to JSON file
            
        Returns:
            List of documents ready for insertion
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        flat_docs = self.flatten_documents(data)
        print(f"✓ Flattened {len(flat_docs)} documents from nested structure")
        
        all_chunks = []
        for doc in flat_docs:
            content = doc.get("content", "")
            
            if not content or len(content.strip()) == 0:
                all_chunks.append({
                    "text": f"Title: {doc['title']}",
                    "title": doc["title"] or "",
                    "url": doc["url"] or "",
                    "parent_url": doc.get("parent_url", "") or "",
                    "category": doc.get("category", "general"),
                    "depth": str(doc["depth"]),
                    "crawl_date": doc.get("crawl_date", ""),
                    "doc_type": "metadata_only"
                })
                continue
            
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "title": doc["title"] or "",
                    "url": doc["url"] or "",
                    "parent_url": doc.get("parent_url", "") or "",
                    "category": doc.get("category", "general"),
                    "depth": str(doc["depth"]),
                    "crawl_date": doc.get("crawl_date", ""),
                    "doc_type": f"chunk_{i}" if len(chunks) > 1 else "full_content"
                })
        
        print(f"✓ Created {len(all_chunks)} total chunks for insertion")
        return all_chunks
    
    def insert_data(self, chunks: List[Dict[str, Any]]):
        """
        Insert data into ChromaDB using fine-tuned sentence transformer embeddings
        
        Args:
            chunks: List of document chunks
        """
        total_chunks = len(chunks)
        print(f"\n{'='*60}")
        print(f"Embedding {total_chunks} chunks using fine-tuned model")
        print(f"{'='*60}\n")
        
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            # Get texts to embed
            texts = [chunk["text"] for chunk in batch]
            
            # Get embeddings from fine-tuned model
            try:
                embeddings = self.get_embeddings(texts)
            except Exception as e:
                print(f"✗ Failed to get embeddings for batch {batch_num}: {e}")
                continue
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in range(len(batch))]
            documents = [chunk["text"] for chunk in batch]
            metadatas = [
                {
                    "title": chunk["title"] or "",
                    "url": chunk["url"] or "",
                    "parent_url": chunk["parent_url"] or "",
                    "category": chunk["category"] or "general",
                    "depth": chunk["depth"] or "",
                    "crawl_date": chunk["crawl_date"] or "",
                    "doc_type": chunk["doc_type"] or ""
                }
                for chunk in batch
            ]
            
            # Insert batch
            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"✓ Inserted batch {batch_num}/{total_batches}")
            except Exception as e:
                print(f"✗ Failed to insert batch {batch_num}: {e}")
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully inserted {total_chunks} chunks into {self.collection_name}")
        print(f"{'='*60}\n")
    
    def load_from_json(self, json_file_path: str):
        """
        Complete pipeline: load JSON, prepare data, and insert into ChromaDB
        
        Args:
            json_file_path: Path to JSON file
        """
        print(f"\n{'='*60}")
        print(f"Loading data from: {json_file_path}")
        print(f"{'='*60}\n")
        
        chunks = self.prepare_data_for_insertion(json_file_path)
        self.insert_data(chunks)
        
        # Get collection stats
        count = self.collection.count()
        print(f"✓ Collection '{self.collection_name}' contains {count} documents")
        print(f"✓ Embedding model path: {self.embedding_model_path}")


def main():
    """Main execution function"""
    
    # Configuration
    JSON_FILE_PATH = "uchicago_msads_content_rag.json"  # Your existing JSON
    PERSIST_DIRECTORY = "./chroma_db_finetuned"
    COLLECTION_NAME = "uchicago_msads_finetuned"
    MODEL_PATH = "../finetune-embedding/exp_finetune"  # Fine-tuned model path
    
    # Initialize loader
    loader = FinetunedChromaLoader(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model_path=MODEL_PATH,
        batch_size=100
    )
    
    # Create collection (drop existing if you want to reload)
    loader.create_collection(drop_existing=True)
    
    # Load and insert data
    loader.load_from_json(JSON_FILE_PATH)
    
    print("\n" + "="*60)
    print("✓ Data loading complete!")
    print(f"✓ Collection '{COLLECTION_NAME}' is ready for queries")
    print(f"✓ Using fine-tuned sentence transformer model")
    print("="*60)


if __name__ == "__main__":
    main()
