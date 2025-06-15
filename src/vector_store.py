"""Vector store implementation using ChromaDB."""
import os
import logging
import chromadb
from typing import List, Dict, Any, Optional, Tuple
from chromadb.config import Settings

from config import Config, get_absolute_path
from document_loader import Document


class VectorStore:
    """Vector database for storing and retrieving document embeddings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_path = get_absolute_path(config.vector_db.path)
        self.collection_name = config.vector_db.collection_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB
        os.makedirs(self.db_path, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            self.logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": f"Document embeddings for {config.data.company_name}"}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = doc.metadata.get('chunk_id', f"doc_{i}")
            ids.append(doc_id)
            texts.append(doc.content)
            
            # Prepare metadata (ChromaDB requires string values)
            metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
                else:
                    metadata[key] = str(value)
            
            metadatas.append(metadata)
        
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using embedding similarity."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.get_document_count()),
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def search_by_text(self, query: str, llm_client, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search using text query (generates embedding first)."""
        try:
            query_embedding = llm_client.generate_embedding(query)
            return self.search(query_embedding, n_results)
        except Exception as e:
            self.logger.error(f"Text search failed: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store."""
        try:
            return self.collection.count()
        except:
            return 0
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            metadata = self.collection.metadata or {}
            
            return {
                'name': self.collection_name,
                'document_count': count,
                'metadata': metadata,
                'db_path': self.db_path
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": f"Document embeddings for {self.config.data.company_name}"}
            )
            self.logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            raise
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document with the given ID exists."""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except:
            return False
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by its ID."""
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if result['documents'] and result['documents'][0]:
                return {
                    'id': doc_id,
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def update_document(self, doc_id: str, content: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update an existing document."""
        try:
            update_data = {'ids': [doc_id]}
            
            if content is not None:
                update_data['documents'] = [content]
            
            if metadata is not None:
                # Convert metadata values to strings
                str_metadata = {}
                for key, value in metadata.items():
                    str_metadata[key] = str(value)
                update_data['metadatas'] = [str_metadata]
            
            self.collection.update(**update_data)
            self.logger.info(f"Updated document: {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update document {doc_id}: {e}")
            return False