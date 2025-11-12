import json
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

class UCMSADSChromaLoader:
    """Load UChicago MS-ADS content into ChromaDB vector database"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "uchicago_msads_docs",
        embedding_model: str = "../finetune-embedding/exp_finetune"
    ):
        """
        Initialize ChromaDB loader
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name for the collection
            embedding_model: Path to fine-tuned model or HuggingFace model name
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model)
        self.collection = None
        
    def create_collection(self, drop_existing: bool = False):
        """Create ChromaDB collection"""
        if drop_existing:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"Dropped existing collection: {self.collection_name}")
            except:
                pass
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "UChicago MS-ADS program content for RAG"}
        )
        print(f"Collection ready: {self.collection_name}")
    
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
            "depth": data.get("depth", 0),
            "content": data.get("content", ""),
            "last_scraped": data.get("last_scraped", "")
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
        print(f"Flattened {len(flat_docs)} documents from nested structure")
        
        all_chunks = []
        for doc in flat_docs:
            content = doc.get("content", "")
            
            if not content or len(content.strip()) == 0:
                all_chunks.append({
                    "text": f"Title: {doc['title']}",
                    "title": doc["title"] or "",
                    "url": doc["url"] or "",
                    "parent_url": doc.get("parent_url", "") or "",
                    "depth": str(doc["depth"]),
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
                    "depth": str(doc["depth"]),
                    "doc_type": f"chunk_{i}" if len(chunks) > 1 else "full_content"
                })
        
        print(f"Created {len(all_chunks)} total chunks for insertion")
        return all_chunks
    
    def insert_data(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Insert data into ChromaDB
        
        Args:
            chunks: List of document chunks
            batch_size: Number of documents to insert per batch
        """
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in batch]
            embeddings = self.encoder.encode(texts, show_progress_bar=False)
            
            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in range(len(batch))]
            documents = [chunk["text"] for chunk in batch]
            metadatas = [
                {
                    "title": chunk["title"] or "",
                    "url": chunk["url"] or "",
                    "parent_url": chunk["parent_url"] or "",
                    "depth": chunk["depth"] or "",
                    "doc_type": chunk["doc_type"] or ""
                }
                for chunk in batch
            ]
            
            # Insert batch
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas
            )
            
            print(f"Inserted batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size} "
                  f"({len(batch)} documents)")
        
        print(f"\n✓ Successfully inserted {total_chunks} chunks into {self.collection_name}")
    
    def load_from_json(self, json_file_path: str, batch_size: int = 100):
        """
        Complete pipeline: load JSON, prepare data, and insert into ChromaDB
        
        Args:
            json_file_path: Path to JSON file
            batch_size: Batch size for insertion
        """
        print(f"Loading data from: {json_file_path}")
        chunks = self.prepare_data_for_insertion(json_file_path)
        
        print(f"\nInserting data into collection: {self.collection_name}")
        self.insert_data(chunks, batch_size)
        
        # Get collection stats
        count = self.collection.count()
        print(f"\nCollection contains {count} documents")
    
def main():
    """Main execution function"""
    
    # Update this path to point to your JSON file
    JSON_FILE_PATH = "uchicago_msads_content_rag.json"  # Assumes file is in same directory
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "uchicago_msads_docs_finetuned"  # Updated name to reflect fine-tuned embeddings
    
    # Initialize loader with fine-tuned embeddings
    print("🚀 Using FINE-TUNED embedding model for better retrieval!")
    loader = UCMSADSChromaLoader(
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model="../finetune-embedding/exp_finetune"  # Fine-tuned model
    )
    
    # Create collection (drop existing if you want to reload)
    loader.create_collection(drop_existing=True)
    
    # Load and insert data
    loader.load_from_json(JSON_FILE_PATH, batch_size=100)
    
    print("\n✓ Data loading complete!")
    print(f"✓ Collection '{COLLECTION_NAME}' is ready for queries")
    print(f"\nTo query the data, run: python query_chromadb.py")


if __name__ == "__main__":
    main()
