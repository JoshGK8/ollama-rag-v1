# Ollama RAG System

A generic Retrieval-Augmented Generation (RAG) system powered by Ollama that allows users to ask questions and get accurate answers from documentation datasets.

## Features

- **Document Processing**: Automatically processes markdown files from documentation datasets
- **Vector Embeddings**: Generates embeddings using Ollama models for semantic search
- **Question Answering**: Interactive CLI for asking questions about your documentation
- **Configurable**: Easy to switch between different documentation datasets
- **Local First**: Runs entirely locally using Ollama - no external API calls required

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- At least one embedding model available in Ollama (e.g., `nomic-embed-text`)
- At least one chat model available in Ollama (e.g., `llama2`, `mistral`)

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ollama-rag-system
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure your dataset**
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml to point to your documentation directory
   ```

4. **Process your documents**
   ```bash
   python src/process_documents.py
   ```

5. **Start asking questions**
   ```bash
   python src/rag_cli.py
   ```

## Configuration

The system uses a YAML configuration file to specify:
- Documentation source directory
- Ollama model settings
- Chunking parameters
- Vector database settings

See `config.example.yaml` for all available options.

## Project Structure

```
ollama-rag-system/
├── src/
│   ├── config.py           # Configuration management
│   ├── document_loader.py  # Document processing
│   ├── embeddings.py       # Embedding generation
│   ├── vector_store.py     # Vector database operations
│   ├── retriever.py        # Document retrieval
│   ├── llm_interface.py    # Ollama integration
│   ├── rag_engine.py       # Main RAG logic
│   ├── process_documents.py # Document processing script
│   └── rag_cli.py          # Interactive CLI
├── config.example.yaml     # Example configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## Supported Document Formats

- Markdown (.md)
- Plain text (.txt)
- More formats can be added via the document loader

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details