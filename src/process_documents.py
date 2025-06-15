#!/usr/bin/env python3
"""Script to process documents and build the knowledge base."""
import os
import sys
import logging
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from rag_engine import RAGEngine


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('rag_system.log', mode='a')
        ]
    )


def main():
    """Main function to process documents."""
    parser = argparse.ArgumentParser(description='Process documents for RAG system')
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of the knowledge base'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine(config)
        
        # Check system status
        status = rag_engine.get_system_status()
        
        if not status['ollama_connected']:
            logger.error("Ollama is not running or not accessible. Please start Ollama first.")
            sys.exit(1)
        
        if not status['embedding_model_available']:
            logger.error(f"Embedding model '{config.ollama.embedding_model}' not found.")
            logger.info(f"Available models: {status['available_models']}")
            logger.info(f"Run: ollama pull {config.ollama.embedding_model}")
            sys.exit(1)
        
        if not status['chat_model_available']:
            logger.error(f"Chat model '{config.ollama.chat_model}' not found.")
            logger.info(f"Available models: {status['available_models']}")
            logger.info(f"Run: ollama pull {config.ollama.chat_model}")
            sys.exit(1)
        
        # Process documents
        logger.info("Processing documents...")
        result = rag_engine.initialize_knowledge_base(force_rebuild=args.force_rebuild)
        
        # Print results
        print(f"\n✅ Knowledge base successfully built!")
        print(f"📁 Company: {result.get('company', 'Unknown')}")
        print(f"📄 Documents processed: {result.get('total_documents', 0)}")
        print(f"🧩 Chunks created: {result.get('chunks_created', 0)}")
        print(f"💾 Total characters: {result.get('total_characters', 0):,}")
        print(f"📊 Average document size: {result.get('average_doc_size', 0):,} characters")
        
        file_types = result.get('file_types', {})
        if file_types:
            print(f"\n📋 File types processed:")
            for file_type, count in file_types.items():
                print(f"  {file_type}: {count} files")
        
        print(f"\n🚀 Ready to answer questions! Run: python src/rag_cli.py")
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        logger.info("Copy config.example.yaml to config.yaml and customize it.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()