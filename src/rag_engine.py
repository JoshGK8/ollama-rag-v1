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
    
    def query(self, question: str, include_sources: bool = True, interface_preference: str = None) -> Dict[str, Any]:
        """Answer a question using RAG with optional interface preference."""
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
            
            # Check for multiple interfaces if user is asking "how to" do something
            if interface_preference is None and self._is_how_to_question(question):
                interfaces = self._detect_multiple_interfaces(relevant_docs)
                if len(interfaces) > 1:
                    return {
                        'answer': self._generate_interface_clarification(question, interfaces),
                        'sources': self._format_sources(relevant_docs[:self.config.cli.max_sources]) if include_sources else [],
                        'confidence': self._calculate_confidence(relevant_docs),
                        'interfaces_available': interfaces,
                        'needs_clarification': True
                    }
            
            # Filter documents by interface preference if specified
            if interface_preference:
                relevant_docs = self._filter_docs_by_interface(relevant_docs, interface_preference)
            
            # Generate answer
            answer = self._generate_answer(question, relevant_docs, interface_preference)
            
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
    
    def _generate_answer(self, question: str, relevant_docs: List[Dict[str, Any]], interface_preference: str = None) -> str:
        """Generate an answer using the LLM with retrieved context."""
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source = doc['metadata'].get('source', f'Document {i+1}')
            content = doc['content']
            context_parts.append(f"Source: {source}\nContent: {content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Create focused, concise prompt with optional interface preference
        interface_instruction = ""
        if interface_preference:
            interface_instruction = f"\n- Focus on {interface_preference.upper()} instructions when multiple interface options are available"
        
        system_message = f"""You are a technical documentation assistant. Answer the question using ONLY the provided context.

RULES:
- Use ONLY information explicitly stated in the context
- Answer ONLY what is directly asked - ignore unrelated information in the context
- Be concise and focus on the specific question asked
- Quote exactly from the context - do not paraphrase technical details
- If the context doesn't contain the answer, say "The documentation does not contain information about [topic]"
- Use exact terms from the context (don't substitute "crypto" for other terms)
- DO NOT include information about policies, management, or configuration unless specifically asked{interface_instruction}

CONTEXT:
{context}

Question: {question}

Provide a direct, focused answer that addresses ONLY the question asked, using exact quotes from the context when describing technical details."""
        
        user_message = f"Based ONLY on the context above, answer this specific question: {question}\n\nFocus only on what is asked - ignore any unrelated information in the context."
        
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
        
        # Check if answer stays focused on the question asked
        if self._answer_includes_irrelevant_info(answer, question):
            return self._extract_relevant_parts(answer, question, relevant_docs)
        
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
    
    def _answer_includes_irrelevant_info(self, answer: str, question: str) -> bool:
        """Check if answer includes information not relevant to the question."""
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Define question categories and what should NOT be included unless asked
        irrelevant_patterns = {
            'infrastructure': ['policy', 'management', 'configuration', 'login', 'admin'],
            'communication': ['policy', 'management', 'configuration', 'login', 'admin'],
            'architecture': ['policy', 'management', 'configuration', 'login', 'admin']
        }
        
        # Detect question type
        question_type = None
        if any(word in question_lower for word in ['communicate', 'connection', 'network']):
            question_type = 'communication'
        elif any(word in question_lower for word in ['infrastructure', 'architecture', 'system']):
            question_type = 'infrastructure'
        
        if question_type and question_type in irrelevant_patterns:
            # Check if answer includes irrelevant information
            for irrelevant_term in irrelevant_patterns[question_type]:
                if irrelevant_term in answer_lower and irrelevant_term not in question_lower:
                    return True
        
        return False
    
    def _extract_relevant_parts(self, answer: str, question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Extract only the parts of the answer relevant to the question."""
        question_lower = question.lower()
        
        # For communication questions, focus on technical connectivity
        if any(word in question_lower for word in ['communicate', 'connection', 'network']):
            # Look for communication-related sentences in the answer
            sentences = answer.split('. ')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(comm_word in sentence_lower for comm_word in 
                      ['connect', 'network', 'cable', 'proxy', 'internet', 'communicate', 'message']):
                    # Exclude policy/management sentences
                    if not any(policy_word in sentence_lower for policy_word in 
                              ['policy', 'management', 'admin', 'login', 'configure']):
                        relevant_sentences.append(sentence)
            
            if relevant_sentences:
                return '. '.join(relevant_sentences) + '.'
        
        # Fallback to conservative answer
        return self._generate_conservative_answer(question, relevant_docs)
    
    def _is_how_to_question(self, question: str) -> bool:
        """Detect if this is a 'how to' question that might have multiple interface solutions."""
        question_lower = question.lower()
        how_to_patterns = [
            'how to', 'how do i', 'how can i', 'how should i',
            'steps to', 'way to', 'process to', 'method to',
            'configure', 'setup', 'create', 'manage', 'access'
        ]
        return any(pattern in question_lower for pattern in how_to_patterns)
    
    def _detect_multiple_interfaces(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Detect which interfaces are mentioned in the retrieved documents."""
        interfaces = set()
        
        # Generic interface patterns that work for any company
        interface_patterns = {
            'gui': ['web interface', 'dashboard', 'web ui', 'graphical', 'browser', 'click', 'button', 'menu', 'page'],
            'cli': ['command line', 'terminal', 'cli', 'command', 'script', 'shell', 'console'],
            'api': ['api', 'endpoint', 'rest', 'http', 'post', 'get', 'json', 'curl'],
            'mobile': ['mobile app', 'smartphone', 'android', 'ios', 'app'],
            'desktop': ['desktop app', 'application', 'software', 'program']
        }
        
        # Combine all document content
        all_content = ' '.join([doc['content'].lower() for doc in docs])
        
        # Check for each interface type
        for interface_type, patterns in interface_patterns.items():
            if any(pattern in all_content for pattern in patterns):
                interfaces.add(interface_type)
        
        return sorted(list(interfaces))
    
    def _generate_interface_clarification(self, question: str, interfaces: List[str]) -> str:
        """Generate a clarification question about which interface to use."""
        interface_descriptions = {
            'gui': 'web dashboard/graphical interface',
            'cli': 'command line interface',
            'api': 'REST API/programmatic interface',
            'mobile': 'mobile application',
            'desktop': 'desktop application'
        }
        
        interface_list = ', '.join([
            f"{interface.upper()} ({interface_descriptions.get(interface, interface)})"
            for interface in interfaces
        ])
        
        # Clean up the question for better readability
        clean_question = question.lower()
        for phrase in ['how to ', 'how do i ', 'how can i ', 'steps to ', 'way to ']:
            clean_question = clean_question.replace(phrase, '')
        clean_question = clean_question.rstrip('?')
        
        return f"""I found instructions for multiple interfaces to {clean_question}.

Available interfaces: {interface_list}

Which interface would you prefer to use? Please specify one of: {', '.join(interfaces)}"""
    
    def _filter_docs_by_interface(self, docs: List[Dict[str, Any]], interface_preference: str) -> List[Dict[str, Any]]:
        """Filter documents to focus on the preferred interface."""
        interface_patterns = {
            'gui': ['web interface', 'dashboard', 'web ui', 'graphical', 'browser', 'click', 'button', 'menu', 'page'],
            'cli': ['command line', 'terminal', 'cli', 'command', 'script', 'shell', 'console'],
            'api': ['api', 'endpoint', 'rest', 'http', 'post', 'get', 'json', 'curl'],
            'mobile': ['mobile app', 'smartphone', 'android', 'ios', 'app'],
            'desktop': ['desktop app', 'application', 'software', 'program']
        }
        
        preference_patterns = interface_patterns.get(interface_preference.lower(), [])
        if not preference_patterns:
            return docs  # Return all docs if interface not recognized
        
        # Score documents based on interface preference
        scored_docs = []
        for doc in docs:
            content_lower = doc['content'].lower()
            interface_score = sum(1 for pattern in preference_patterns if pattern in content_lower)
            
            # Boost score for this document if it mentions the preferred interface
            if interface_score > 0:
                doc_copy = doc.copy()
                doc_copy['interface_relevance'] = interface_score
                scored_docs.append(doc_copy)
        
        # If we found interface-specific docs, prioritize them; otherwise return all
        if scored_docs:
            scored_docs.sort(key=lambda x: x.get('interface_relevance', 0), reverse=True)
            return scored_docs
        
        return docs
    
    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on similarity scores."""
        if not docs:
            return 0.0
        
        # Get similarity scores (now properly calculated in vector_store.py)
        similarities = [doc['similarity'] for doc in docs if 'similarity' in doc]
        
        if not similarities:
            return 0.0
        
        # Use the highest similarity score as base confidence
        max_similarity = max(similarities)
        
        # Boost confidence if we have multiple relevant documents
        num_relevant = sum(1 for sim in similarities if sim > 0.3)
        relevance_boost = min(num_relevant * 0.1, 0.2)
        
        confidence = min(max_similarity + relevance_boost, 1.0)
        return round(confidence * 100, 1)  # Return as percentage
    
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