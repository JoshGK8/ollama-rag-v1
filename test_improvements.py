#!/usr/bin/env python3
"""
Test script to demonstrate the improvements in hallucination prevention.
"""
import sys
sys.path.append('src')
from rag_engine import RAGEngine
from config import load_config

def test_hallucination_prevention():
    """Test various scenarios that previously caused hallucinations."""
    
    config = load_config('config.yaml')
    rag = RAGEngine(config)
    
    test_cases = [
        {
            "question": "I can't send crypto. Its giving me some error about policy. What should I do?",
            "expected_issues": ["crypto policy", "step-by-step instructions not in docs"]
        },
        {
            "question": "How do I access the crypto dashboard?",
            "expected_issues": ["crypto dashboard", "navigation instructions"]
        },
        {
            "question": "What is the cryptocurrency policy?", 
            "expected_issues": ["cryptocurrency policy vs transaction policy"]
        }
    ]
    
    print("=" * 80)
    print("IMPROVED RAG SYSTEM - HALLUCINATION PREVENTION TEST")
    print("=" * 80)
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        
        print(f"TEST {i}: {question}")
        print("-" * 60)
        
        # Get answer from improved system
        result = rag.query(question)
        answer = result['answer']
        confidence = result['confidence']
        
        print(f"Answer: {answer}")
        print(f"Confidence: {confidence}")
        
        # Basic hallucination check
        answer_lower = answer.lower()
        hallucination_detected = False
        
        for issue in test_case["expected_issues"]:
            if any(term in answer_lower for term in issue.split(" vs ")[0].split()):
                if "crypto policy" in answer_lower or "cryptocurrency policy" in answer_lower:
                    hallucination_detected = True
                    break
        
        if hallucination_detected:
            print("⚠️  POTENTIAL HALLUCINATION DETECTED")
        else:
            print("✅ NO OBVIOUS HALLUCINATIONS")
        
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("The improved system should:")
    print("1. ✅ Avoid using 'crypto policy' when docs only mention 'transaction policy'")
    print("2. ✅ Be more conservative when information is not available")
    print("3. ✅ Validate answers against source material")
    print("4. ✅ Provide fallback responses when unsure")

if __name__ == "__main__":
    test_hallucination_prevention()