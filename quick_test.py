#!/usr/bin/env python3
"""Quick test of the interface detection system."""
import sys
sys.path.append('src')
from rag_engine import RAGEngine
from config import load_config

def quick_test():
    config = load_config('config.yaml')
    rag = RAGEngine(config)
    
    question = "How do I create a wallet?"
    print(f"Question: {question}")
    print("=" * 50)
    
    result = rag.query(question, include_sources=False)
    
    if result.get('needs_clarification'):
        print("🎉 INTERFACE CLARIFICATION WORKING!")
        print(f"Available interfaces: {result.get('interfaces_available', [])}")
        print("\nClarification message:")
        print(result['answer'])
        
        # Test with GUI preference
        print("\n" + "="*50)
        print("Testing with GUI preference:")
        gui_result = rag.query(question, include_sources=False, interface_preference="gui")
        print(f"GUI Answer: {gui_result['answer'][:200]}...")
        
    else:
        print("Direct answer:")
        print(result['answer'])

if __name__ == "__main__":
    quick_test()