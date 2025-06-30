#!/usr/bin/env python3
"""
Test the transaction question that showed issues.
"""
import sys
sys.path.append('src')
from rag_engine import RAGEngine
from config import load_config

def test_transaction_question():
    config = load_config('config.yaml')
    rag = RAGEngine(config)
    
    question = "How do I send a transaction?"
    
    print("=" * 80)
    print("TESTING TRANSACTION QUESTION")
    print("=" * 80)
    print(f"Question: {question}")
    print()
    
    # Test initial query
    result = rag.query(question, include_sources=False)
    
    if result.get('needs_clarification'):
        print("✅ INTERFACE CLARIFICATION TRIGGERED")
        print(f"Available interfaces: {result.get('interfaces_available', [])}")
        print("\nClarification message:")
        print(result['answer'])
        print()
        
        # Test each interface option
        for interface in result.get('interfaces_available', []):
            print(f"=" * 60)
            print(f"TESTING {interface.upper()} PREFERENCE:")
            print(f"=" * 60)
            
            specific_result = rag.query(question, include_sources=True, interface_preference=interface)
            print(f"Answer: {specific_result['answer']}")
            print(f"Confidence: {specific_result['confidence']}")
            
            # Show sources to verify interface relevance
            if specific_result.get('sources'):
                print("\nTop sources:")
                for source in specific_result['sources'][:2]:
                    print(f"- {source['title']} (similarity: {source['similarity']:.1%})")
            print()
    else:
        print("❌ No interface clarification - this should have triggered")
        print(f"Direct answer: {result['answer']}")

if __name__ == "__main__":
    test_transaction_question()