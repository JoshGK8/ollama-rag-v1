"""Document loader for processing markdown and text files."""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config, get_absolute_path


@dataclass
class Document:
    """Represents a loaded document with metadata."""
    content: str
    metadata: Dict[str, Any]
    source: str


class DocumentLoader:
    """Loads and processes documents from the specified directory."""
    
    def __init__(self, config: Config):
        self.config = config
        self.source_dir = get_absolute_path(config.data.source_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self) -> List[Document]:
        """Load all supported documents from the source directory."""
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        
        documents = []
        file_count = 0
        
        self.logger.info(f"Loading documents from {self.source_dir}")
        
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in self.config.data.file_extensions:
                    try:
                        doc = self._load_single_file(file_path)
                        if doc:
                            documents.append(doc)
                            file_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to load {file_path}: {e}")
        
        self.logger.info(f"Loaded {file_count} documents")
        return documents
    
    def _load_single_file(self, file_path: str) -> Optional[Document]:
        """Load a single file and return a Document object."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if len(content) < self.config.processing.min_chunk_size:
                self.logger.debug(f"Skipping {file_path}: content too short")
                return None
            
            # Create metadata
            relative_path = os.path.relpath(file_path, self.source_dir)
            metadata = {
                'source': relative_path,
                'file_path': file_path,
                'file_size': len(content),
                'file_type': Path(file_path).suffix.lower(),
                'company': self.config.data.company_name
            }
            
            # Extract title from content if possible
            title = self._extract_title(content, relative_path)
            if title:
                metadata['title'] = title
            
            return Document(
                content=content,
                metadata=metadata,
                source=relative_path
            )
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _extract_title(self, content: str, filename: str) -> str:
        """Extract title from document content or filename."""
        lines = content.split('\n')
        
        # Look for markdown h1 header
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fall back to filename without extension
        return Path(filename).stem.replace('-', ' ').replace('_', ' ').title()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        chunked_docs = []
        
        self.logger.info(f"Chunking {len(documents)} documents")
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.content)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) >= self.config.processing.min_chunk_size:
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': f"{doc.source}_{i}",
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk)
                    })
                    
                    chunked_docs.append(Document(
                        content=chunk.strip(),
                        metadata=chunk_metadata,
                        source=f"{doc.source} (chunk {i+1}/{len(chunks)})"
                    ))
        
        self.logger.info(f"Created {len(chunked_docs)} chunks")
        return chunked_docs
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the loaded documents."""
        if not documents:
            return {}
        
        total_chars = sum(len(doc.content) for doc in documents)
        file_types = {}
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'average_doc_size': total_chars // len(documents),
            'file_types': file_types,
            'company': self.config.data.company_name,
            'dataset_description': self.config.data.dataset_description
        }