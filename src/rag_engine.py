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
        
        # Create ultra-strict prompt to prevent hallucination
        system_message = f"""You are a documentation assistant with ABSOLUTE CONSTRAINTS:

IMMUTABLE RULES:
- ONLY use information EXPLICITLY stated in the provided context
- NEVER add, interpret, assume, or infer beyond the exact text
- NEVER use synonyms or related terms not in the context
- NEVER provide instructions unless they appear verbatim
- ALWAYS say "The provided documentation does not contain specific information about [topic]" when unsure

FORBIDDEN SUBSTITUTIONS:
- "crypto policy" → ONLY use "transaction policy" if that's what appears
- "cryptocurrency dashboard" → ONLY use "dashboard" if that's what appears  
- "digital asset settings" → ONLY use the exact terms from context

RESPONSE FORMAT:
- Start with "According to the documentation provided:" 
- Quote directly from context
- End with limitations if information is incomplete

CONTEXT:
{context}

IMPORTANT: If you cannot answer fully using ONLY the context above, say so explicitly."""
        
        user_message = f"Based ONLY on the context above, answer: {question}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Generate initial answer
        answer = self.llm_client.chat_completion(messages)
        
        # Validate answer against source material
        validated_answer = self._validate_answer_against_sources(answer, relevant_docs, question)
        
        return validated_answer
    
    def _format_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for display."""
        sources = []
        for doc in docs:
            metadata = doc['metadata']
            sources.append({
                'title': metadata.get('title', metadata.get('source', 'Unknown')),
                'source': metadata.get('source', 'Unknown'),
                'similarity': round(doc['similarity'], 3),
                'chunk_info': f"Chunk {int(metadata.get('chunk_index', 0)) + 1}/{metadata.get('total_chunks', 1)}"
            })
        return sources
    
    def _validate_answer_against_sources(self, answer: str, relevant_docs: List[Dict[str, Any]], question: str) -> str:
        """Validate the generated answer against source documents to prevent hallucination."""
        # Extract key terms from the answer for validation
        answer_lower = answer.lower()
        
        # Common hallucination patterns to catch and correct
        problematic_terms = {
            'crypto policy': 'transaction policy',
            'cryptocurrency policy': 'transaction policy', 
            'digital asset policy': 'transaction policy',
            'crypto dashboard': 'dashboard',
            'crypto settings': 'settings',
            'digital wallet policy': 'wallet policy',
            'blockchain policy': 'transaction policy'
        }
        
        # Combine all source content for validation
        source_text = ' '.join([doc['content'].lower() for doc in relevant_docs])
        
        # Check for hallucinated terms and flag for replacement
        hallucination_detected = False
        for bad_term, correct_term in problematic_terms.items():
            if bad_term in answer_lower:
                # Check if the correct term actually exists in sources
                if bad_term not in source_text:
                    hallucination_detected = True
                    # Only replace if we have evidence of the correct term
                    if correct_term in source_text:
                        answer = answer.replace(bad_term, correct_term)
                        answer = answer.replace(bad_term.title(), correct_term.title())
        
        # Check for unsupported claims or instructions
        if hallucination_detected or self._contains_unsupported_claims(answer, relevant_docs):
            self.logger.warning(f"Potential hallucination detected in answer: {answer[:100]}...")
            return self._generate_conservative_answer(question, relevant_docs)
        
        # Additional validation: check if answer makes claims not in sources
        if self._answer_exceeds_source_scope(answer, source_text):
            return self._generate_conservative_answer(question, relevant_docs)
        
        return answer
    
    def _contains_unsupported_claims(self, answer: str, relevant_docs: List[Dict[str, Any]]) -> bool:
        """Check if answer contains claims not supported by source documents."""
        # Combine all source content
        source_content = ' '.join([doc['content'].lower() for doc in relevant_docs])
        answer_lower = answer.lower()
        
        # Flag potentially unsupported instructions
        instruction_phrases = [
            'you can check', 'you should update', 'consult with', 
            'navigate to', 'click on', 'go to the', 'access the'
        ]
        
        for phrase in instruction_phrases:
            if phrase in answer_lower and phrase not in source_content:
                return True
        
        return False
    
    def _generate_conservative_answer(self, question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate a conservative answer that strictly adheres to source material."""
        # Find the most relevant document
        if not relevant_docs:
            return "I cannot find information about this topic in the provided documentation."
        
        best_doc = relevant_docs[0]
        source_name = best_doc['metadata'].get('source', 'the documentation')
        
        # Create a conservative response template
        if 'policy' in question.lower() and 'error' in question.lower():
            return f"""According to {source_name}, there are transaction-related APIs and error handling mechanisms available. However, the specific error you're encountering and its resolution steps are not detailed in the provided documentation. 

The documentation mentions various transaction policies and approval workflows, but I cannot provide specific troubleshooting steps without more detailed error information in the source material."""
        
        return f"The provided documentation from {source_name} contains related information, but does not include specific details needed to fully answer your question. Please refer to the complete documentation or contact support for detailed troubleshooting steps."
    
    def _answer_exceeds_source_scope(self, answer: str, source_text: str) -> bool:
        """Check if the answer makes claims that go beyond the source material scope."""
        answer_lower = answer.lower()
        
        # Patterns that suggest the AI is making up information
        overreach_patterns = [
            'you can do this by',
            'follow these steps',
            'the system allows you to',
            'simply navigate to',
            'you will find',
            'this feature enables',
            'to resolve this issue'
        ]
        
        for pattern in overreach_patterns:
            if pattern in answer_lower and pattern not in source_text:
                return True
        
        return False
    
    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on similarity scores."""
        if not docs:
            return 0.0
        
        # Convert distance-based similarity to actual similarity (0-1 scale)
        # ChromaDB returns distances, so lower is better
        similarities = []
        for doc in docs:
            # Convert distance to similarity (assuming cosine distance)
            distance = abs(doc['similarity'])
            similarity = max(0, 1 - (distance / 2))  # Normalize to 0-1 range
            similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Use the highest similarity score as base confidence
        max_similarity = max(similarities)
        
        # Boost confidence if we have multiple relevant documents
        num_relevant = sum(1 for sim in similarities if sim > 0.3)
        relevance_boost = min(num_relevant * 0.05, 0.2)
        
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