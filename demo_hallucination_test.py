#!/usr/bin/env python3
"""
Demonstration of the hallucination issue in the RAG system.
This shows how the AI invents information not present in the source documents.
"""
import sys
sys.path.append('src')
from rag_engine import RAGEngine
from config import load_config

def main():
    print("=" * 80)
    print("RAG HALLUCINATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    config = load_config('config.yaml')
    rag = RAGEngine(config)
    
    # Test the problematic question
    question = "I can't send crypto. Its giving me some error about policy. What should I do?"
    
    print(f"User Question: {question}")
    print("-" * 80)
    
    # Get the raw retrieved documents first
    results = rag.vector_store.search_by_text(question, rag.llm_client, n_results=5)
    
    print("RETRIEVED DOCUMENTS (what the AI actually has access to):")
    print("-" * 60)
    for i, doc in enumerate(results, 1):
        title = doc['metadata'].get('title', 'Unknown')
        print(f"{i}. {title}")
        print(f"   Content: {doc['content'][:300]}...")
        print()
    
    print("-" * 80)
    print("AI GENERATED ANSWER:")
    print("-" * 40)
    
    # Get the AI's answer
    result = rag.query(question)
    print(result['answer'])
    print()
    
    print("-" * 80)
    print("ANALYSIS:")
    print("-" * 40)
    print("❌ Issues identified:")
    print("1. The AI mentions 'crypto policy' - this term doesn't appear in the retrieved documents")
    print("2. The AI provides step-by-step instructions not found in the source material") 
    print("3. The AI makes assumptions about GK8 dashboard sections not mentioned in context")
    print()
    print("✅ The retrieved documents actually contain:")
    print("- Information about transaction policies (not 'crypto policies')")
    print("- API error codes and troubleshooting")
    print("- Approval workflow information")
    print()
    print("💡 Solution needed:")
    print("- More constrained prompting")
    print("- Better document retrieval")
    print("- Validation of AI responses against source material")

if __name__ == "__main__":
    main()