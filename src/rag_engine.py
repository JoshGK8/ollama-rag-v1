"""Main RAG engine that orchestrates document retrieval and answer generation."""
import logging
from typing import List, Dict, Any, Optional, Tuple

from config import Config
from llm_interface import OllamaClient
from vector_store import VectorStore
from document_loader import DocumentLoader


class RAGEngine:
    """Main RAG system that combines retrieval and generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.llm_client = OllamaClient(config)
        self.vector_store = VectorStore(config)
        self.document_loader = DocumentLoader(config)
    
    def initialize_knowledge_base(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """Initialize the knowledge base by loading and processing documents."""
        collection_info = self.vector_store.get_collection_info()
        
        # Check if we need to rebuild
        if not force_rebuild and collection_info.get('document_count', 0) > 0:
            self.logger.info(f"Knowledge base already exists with {collection_info['document_count']} documents")
            return collection_info
        
        self.logger.info("Building knowledge base...")
        
        # Clear existing data if force rebuild
        if force_rebuild:
            self.vector_store.clear_collection()
        
        # Load documents
        documents = self.document_loader.load_documents()
        if not documents:
            raise ValueError("No documents found to process")
        
        # Chunk documents
        chunked_docs = self.document_loader.chunk_documents(documents)
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        texts = [doc.content for doc in chunked_docs]
        embeddings = self.llm_client.generate_embeddings(texts)
        
        # Store in vector database
        self.vector_store.add_documents(chunked_docs, embeddings)
        
        # Get final stats
        doc_stats = self.document_loader.get_document_stats(documents)
        collection_info = self.vector_store.get_collection_info()
        
        result = {
            **doc_stats,
            **collection_info,
            'chunks_created': len(chunked_docs)
        }
        
        self.logger.info(f"Knowledge base initialized: {result}")
        return result
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Answer a question using RAG."""
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.search_by_text(
                question, 
                self.llm_client, 
                n_results=self.config.processing.max_retrieved_chunks
            )
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Generate answer
            answer = self._generate_answer(question, relevant_docs)
            
            # Prepare sources
            sources = []
            if include_sources:
                sources = self._format_sources(relevant_docs[:self.config.cli.max_sources])
            
            # Calculate confidence based on similarity scores
            confidence = self._calculate_confidence(relevant_docs)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'retrieved_chunks': len(relevant_docs)
            }
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    def _generate_answer(self, question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate an answer using the LLM with retrieved context."""
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source = doc['metadata'].get('source', f'Document {i+1}')
            content = doc['content']
            context_parts.append(f"Source: {source}\nContent: {content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create the prompt
        system_message = f"""You are a helpful assistant that answers questions based on the provided context from {self.config.data.company_name} documentation. 

Instructions:
1. Answer the question using only the information provided in the context
2. Be accurate and specific
3. If the context doesn't contain enough information to answer the question, say so
4. Cite specific sources when possible
5. Keep your answer concise but comprehensive

Context:
{context}"""
        
        user_message = f"Question: {question}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return self.llm_client.chat_completion(messages)
    
    def _format_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for display."""
        sources = []
        for doc in docs:
            metadata = doc['metadata']
            sources.append({
                'title': metadata.get('title', metadata.get('source', 'Unknown')),
                'source': metadata.get('source', 'Unknown'),
                'similarity': round(doc['similarity'], 3),
                'chunk_info': f"Chunk {metadata.get('chunk_index', 0) + 1}/{metadata.get('total_chunks', 1)}"
            })
        return sources
    
    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on similarity scores."""
        if not docs:
            return 0.0
        
        # Use the highest similarity score as base confidence
        max_similarity = max(doc['similarity'] for doc in docs)
        
        # Boost confidence if we have multiple relevant documents
        num_relevant = sum(1 for doc in docs if doc['similarity'] > 0.7)
        relevance_boost = min(num_relevant * 0.1, 0.3)
        
        confidence = min(max_similarity + relevance_boost, 1.0)
        return round(confidence, 3)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG system."""
        # Check Ollama connection
        ollama_connected = self.llm_client.check_connection()
        model_status = self.llm_client.check_model_availability()
        
        # Get knowledge base info
        collection_info = self.vector_store.get_collection_info()
        
        return {
            'ollama_connected': ollama_connected,
            'embedding_model_available': model_status.get('embedding_model', False),
            'chat_model_available': model_status.get('chat_model', False),
            'available_models': model_status.get('available_models', []),
            'knowledge_base': collection_info,
            'config': {
                'company_name': self.config.data.company_name,
                'source_dir': self.config.data.source_dir,
                'embedding_model': self.config.ollama.embedding_model,
                'chat_model': self.config.ollama.chat_model
            }
        }
    
    def suggest_questions(self) -> List[str]:
        """Suggest sample questions based on the loaded documents."""
        # This is a simple implementation - could be enhanced with actual content analysis
        company = self.config.data.company_name
        return [
            f"What is {company}?",
            f"How do I get started with {company}?",
            f"What are the main features of {company}?",
            f"How do I configure {company}?",
            f"What APIs does {company} provide?",
            "How do I troubleshoot common issues?",
            "What are the system requirements?",
            "How do I update or upgrade the system?"
        ]