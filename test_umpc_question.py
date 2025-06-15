#!/usr/bin/env python3
"""
Test the specific uMPC communication question.
"""
import sys
sys.path.append('src')
from rag_engine import RAGEngine
from config import load_config

def test_umpc_question():
    config = load_config('config.yaml')
    rag = RAGEngine(config)
    
    question = "How does the Impenetrable Vault communicate with the uMPC?"
    
    print("=" * 80)
    print("TESTING uMPC COMMUNICATION QUESTION")
    print("=" * 80)
    print(f"Question: {question}")
    print("-" * 60)
    
    result = rag.query(question)
    
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Sources: {len(result['sources'])}")
    print()
    
    if result['sources']:
        print("Top Sources:")
        for i, source in enumerate(result['sources'][:3], 1):
            print(f"{i}. {source['title']}")
            print(f"   Similarity: {source['similarity']:.3f}")
            print()

if __name__ == "__main__":
    test_umpc_question()