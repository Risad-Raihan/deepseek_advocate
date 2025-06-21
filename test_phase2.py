"""
Phase 2 Testing Script for Bengali Legal Advocate
Tests all components including LM Studio integration
"""

import os
import sys
import time
import logging
import json
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lm_studio_connection():
    """Test LM Studio connection and model availability"""
    print("🔗 Testing LM Studio Connection...")
    
    try:
        import requests
        
        # Test connection
        response = requests.get("http://localhost:1234/v1/models", timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio connection successful!")
            print(f"Available models: {len(models.get('data', []))}")
            return True
        else:
            print(f"❌ LM Studio connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        return False

def test_components_import():
    """Test if all Phase 2 components can be imported"""
    print("\n📦 Testing Component Imports...")
    
    try:
        from src.query_processor import LegalQueryProcessor
        print("✅ Query Processor imported successfully")
        
        from src.legal_rag import LegalRAGEngine
        print("✅ Legal RAG Engine imported successfully")
        
        from src.context_builder import LegalContextBuilder
        print("✅ Context Builder imported successfully")
        
        from src.response_generator import BengaliLegalResponseGenerator
        print("✅ Response Generator imported successfully")
        
        from src.retrieval_strategies import RetrievalStrategyFactory
        print("✅ Retrieval Strategies imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Component import error: {e}")
        return False

def test_bengali_processor():
    """Test Bengali text processing capabilities"""
    print("\n🔤 Testing Bengali Text Processing...")
    
    try:
        from src.bengali_processor import BengaliProcessor
        
        processor = BengaliProcessor()
        
        # Test text preprocessing
        test_text = "তালাকের পর খোরপোশের নিয়ম কি? ধারা ১২৫ অনুযায়ী।"
        processed = processor.preprocess_bengali_legal_text(test_text)
        print(f"✅ Text preprocessing: '{test_text[:30]}...' → '{processed[:30]}...'")
        
        # Test entity extraction
        entities = processor.extract_legal_entities(test_text)
        print(f"✅ Entity extraction: Found {len(entities)} entity types")
        
        # Test intent classification
        intent = processor.extract_legal_intent(test_text)
        print(f"✅ Intent classification: {intent.get('legal_domain', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bengali processor test failed: {e}")
        return False

def test_query_processor():
    """Test legal query processing"""
    print("\n❓ Testing Query Processing...")
    
    try:
        from src.bengali_processor import BengaliProcessor
        from src.query_processor import LegalQueryProcessor
        
        bengali_processor = BengaliProcessor()
        query_processor = LegalQueryProcessor(bengali_processor)
        
        test_queries = [
            "তালাকের পর খোরপোশের নিয়ম কি?",
            "সংবিধানের ২৭ অনুচ্ছেদে কি বলা হয়েছে?",
            "জমি রেজিস্ট্রেশনের জন্য কি কি কাগজ লাগে?"
        ]
        
        for query in test_queries:
            processed = query_processor.process_legal_query(query)
            domain = processed.get('domain', {}).get('domain', 'unknown')
            complexity = processed.get('complexity', 'unknown')
            print(f"✅ Query: '{query[:40]}...' → Domain: {domain}, Complexity: {complexity}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query processor test failed: {e}")
        return False

def test_vector_store_connectivity():
    """Test vector store loading and search"""
    print("\n🗂️  Testing Vector Store Connectivity...")
    
    try:
        from src.vector_store import VectorStore
        from configs.model_config import EMBEDDING_MODEL, VECTOR_DB_PATH
        
        if not os.path.exists(VECTOR_DB_PATH):
            print("❌ Vector store not found! Please run Phase 1 first.")
            return False
        
        # Initialize vector store
        vector_store = VectorStore(
            embedding_model=EMBEDDING_MODEL,
            vector_db_path=VECTOR_DB_PATH
        )
        
        # Test search
        results = vector_store.hybrid_search("তালাক", level='paragraph', top_k=3)
        print(f"✅ Vector store search: Found {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_retrieval_strategies():
    """Test different retrieval strategies"""
    print("\n🔍 Testing Retrieval Strategies...")
    
    try:
        from src.vector_store import VectorStore
        from src.bengali_processor import BengaliProcessor
        from src.retrieval_strategies import RetrievalStrategyFactory
        from configs.model_config import EMBEDDING_MODEL, VECTOR_DB_PATH
        
        if not os.path.exists(VECTOR_DB_PATH):
            print("❌ Vector store not found! Skipping retrieval test.")
            return False
        
        # Initialize components
        vector_store = VectorStore(EMBEDDING_MODEL, VECTOR_DB_PATH)
        bengali_processor = BengaliProcessor()
        
        # Test different strategies
        strategies = ['direct_legal_retrieval', 'conceptual_retrieval', 'multi_hop_retrieval']
        
        for strategy_name in strategies:
            strategy = RetrievalStrategyFactory.create_strategy(
                strategy_name, vector_store, bengali_processor
            )
            
            # Test retrieval
            processed_query = {
                'clean_query': 'তালাক',
                'domain': {'domain': 'family_law'},
                'entities': {'legal_term': ['তালাক']}
            }
            
            results = strategy.retrieve(processed_query, top_k=3)
            print(f"✅ {strategy_name}: Retrieved {len(results)} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Retrieval strategies test failed: {e}")
        return False

def test_response_generator():
    """Test response generation with LM Studio"""
    print("\n🤖 Testing Response Generation...")
    
    try:
        from src.response_generator import BengaliLegalResponseGenerator
        
        # Initialize response generator
        generator = BengaliLegalResponseGenerator()
        
        # Test with sample RAG output
        sample_rag_output = {
            'query_analysis': {
                'original_query': 'তালাকের নিয়ম কি?',
                'clean_query': 'তালাকের নিয়ম কি',
                'domain': {'domain': 'family_law', 'confidence': 0.9}
            },
            'response_context': 'তালাক সংক্রান্ত আইনি বিধান...',
            'citations': ['মুসলিম পারিবারিক আইন, ১৯৬১'],
            'legal_domain': 'family_law',
            'confidence_score': 0.8
        }
        
        # Generate response
        response = generator.generate_comprehensive_legal_response(sample_rag_output)
        
        if response.get('response'):
            print("✅ Response generation successful")
            print(f"Response length: {len(response['response'])} characters")
            print(f"Generation method: {response.get('processing_metadata', {}).get('generation_method', 'unknown')}")
        else:
            print("❌ Response generation failed - no response content")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Response generator test failed: {e}")
        return False

def test_complete_rag_pipeline():
    """Test complete RAG pipeline"""
    print("\n🔄 Testing Complete RAG Pipeline...")
    
    try:
        from src.vector_store import VectorStore
        from src.bengali_processor import BengaliProcessor
        from src.query_processor import LegalQueryProcessor
        from src.legal_rag import LegalRAGEngine
        from src.response_generator import BengaliLegalResponseGenerator
        from configs.model_config import EMBEDDING_MODEL, VECTOR_DB_PATH
        
        if not os.path.exists(VECTOR_DB_PATH):
            print("❌ Vector store not found! Skipping pipeline test.")
            return False
        
        # Initialize all components
        vector_store = VectorStore(EMBEDDING_MODEL, VECTOR_DB_PATH)
        bengali_processor = BengaliProcessor()
        query_processor = LegalQueryProcessor(bengali_processor)
        response_generator = BengaliLegalResponseGenerator()
        
        rag_engine = LegalRAGEngine(
            vector_store=vector_store,
            bengali_processor=bengali_processor,
            query_processor=query_processor
        )
        
        # Test with sample query
        test_query = "তালাকের পর খোরপোশের নিয়ম কি?"
        
        print(f"Processing query: {test_query}")
        
        # Process through RAG
        rag_output = rag_engine.process_legal_query(test_query)
        
        # Generate response
        final_response = response_generator.generate_comprehensive_legal_response(rag_output)
        
        # Validate results
        if rag_output.get('retrieved_context'):
            print("✅ RAG processing successful")
        else:
            print("❌ RAG processing failed")
            return False
        
        if final_response.get('response'):
            print("✅ Response generation successful")
        else:
            print("❌ Response generation failed")
            return False
        
        print(f"Documents retrieved: {len(rag_output.get('retrieved_context', {}).get('documents', []))}")
        print(f"Response length: {len(final_response.get('response', ''))}")
        print(f"Confidence score: {rag_output.get('confidence_score', 0.0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete RAG pipeline test failed: {e}")
        return False

def generate_test_report(test_results):
    """Generate comprehensive test report"""
    print("\n📊 Test Report")
    print("=" * 50)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if not test_results.get('lm_studio_connection'):
        print("  • Start LM Studio and load DeepSeek model")
    if not test_results.get('vector_store_connectivity'):
        print("  • Run Phase 1 to create vector store")
    if not test_results.get('complete_rag_pipeline'):
        print("  • Check all component dependencies")

def main():
    """Main testing function"""
    print("🧪 Bengali Legal Advocate - Phase 2 Testing")
    print("=" * 60)
    
    # Run all tests
    test_results = {}
    
    test_results['lm_studio_connection'] = test_lm_studio_connection()
    test_results['components_import'] = test_components_import()
    test_results['bengali_processor'] = test_bengali_processor()
    test_results['query_processor'] = test_query_processor()
    test_results['vector_store_connectivity'] = test_vector_store_connectivity()
    test_results['retrieval_strategies'] = test_retrieval_strategies()
    test_results['response_generator'] = test_response_generator()
    test_results['complete_rag_pipeline'] = test_complete_rag_pipeline()
    
    # Generate report
    generate_test_report(test_results)
    
    # Overall result
    all_passed = all(test_results.values())
    if all_passed:
        print("\n🎉 All tests passed! Phase 2 is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main() 