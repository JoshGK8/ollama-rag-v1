#!/usr/bin/env python3
"""
Test the multi-interface detection and clarification system.
"""
import sys
sys.path.append('src')
from rag_engine import RAGEngine
from config import load_config

def test_interface_detection():
    """Test interface detection with various how-to questions."""
    
    config = load_config('config.yaml')
    rag = RAGEngine(config)
    
    test_questions = [
        "How do I create a new wallet?",
        "How to configure transaction policies?", 
        "How can I access the system?",
        "Steps to setup authentication?",
        "What is the system architecture?"  # Non-how-to question
    ]
    
    print("=" * 80)
    print("MULTI-INTERFACE DETECTION TEST")
    print("=" * 80)
    print()
    
    for question in test_questions:
        print(f"Question: {question}")
        print("-" * 60)
        
        # Test initial query (should detect interfaces)
        result = rag.query(question, include_sources=False)
        
        if result.get('needs_clarification'):
            print("✅ INTERFACE CLARIFICATION TRIGGERED")
            print(f"Available interfaces: {result.get('interfaces_available', [])}")
            print(f"Clarification: {result['answer']}")
            print()
            
            # Test with specific interface preference
            for interface in result.get('interfaces_available', []):
                print(f"Testing with {interface.upper()} preference:")
                specific_result = rag.query(question, include_sources=False, interface_preference=interface)
                print(f"Answer: {specific_result['answer'][:150]}...")
                print()
        else:
            print("ℹ️  No interface clarification needed")
            print(f"Direct answer: {result['answer'][:150]}...")
            print()
        
        print("=" * 60)
        print()

if __name__ == "__main__":
    test_interface_detection()