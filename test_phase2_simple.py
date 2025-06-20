"""
Simplified Phase 2 Testing Script for Bengali Legal Advocate
Tests components directly without complex imports
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lm_studio_connection():
    """Test LM Studio connection"""
    print("🔗 Testing LM Studio Connection...")
    
    try:
        import requests
        
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        
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
        print("Please make sure LM Studio is running on http://localhost:1234")
        return False

def test_basic_imports():
    """Test basic component imports"""
    print("\n📦 Testing Basic Component Imports...")
    
    try:
        # Test Bengali processor
        from bengali_processor import BengaliLegalProcessor
        print("✅ Bengali Processor imported")
        
        # Test vector store
        from vector_store import LegalVectorStore
        print("✅ Vector Store imported")
        
        # Test query processor
        from query_processor import BengaliLegalQueryProcessor
        print("✅ Query Processor imported")
        
        # Test response generator
        from response_generator import BengaliLegalResponseGenerator
        print("✅ Response Generator imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_bengali_processing():
    """Test Bengali text processing"""
    print("\n🔤 Testing Bengali Text Processing...")
    
    try:
        from bengali_processor import BengaliLegalProcessor
        
        processor = BengaliLegalProcessor()
        
        # Test text preprocessing
        test_text = "তালাকের পর খোরপোশের নিয়ম কি? ধারা ১২৫ অনুযায়ী।"
        processed = processor.preprocess_bengali_legal_text(test_text)
        print(f"✅ Text preprocessing successful")
        
        # Test entity extraction
        entities = processor.extract_legal_entities(test_text)
        print(f"✅ Entity extraction: Found {len(entities)} entity types")
        
        return True
        
    except Exception as e:
        print(f"❌ Bengali processing test failed: {e}")
        return False

def test_query_processing():
    """Test query processing"""
    print("\n❓ Testing Query Processing...")
    
    try:
        from bengali_processor import BengaliLegalProcessor
        from query_processor import BengaliLegalQueryProcessor
        
        bengali_processor = BengaliLegalProcessor()
        query_processor = BengaliLegalQueryProcessor()
        
        test_query = "তালাকের পর খোরপোশের নিয়ম কি?"
        processed = query_processor.process_legal_query(test_query)
        
        domain = processed.get('domain', {}).get('domain', 'unknown')
        print(f"✅ Query processing: Domain = {domain}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        return False

def test_response_generator():
    """Test response generation"""
    print("\n🤖 Testing Response Generator...")
    
    try:
        from response_generator import BengaliLegalResponseGenerator
        
        generator = BengaliLegalResponseGenerator()
        
        # Test with sample data
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
        
        response = generator.generate_comprehensive_legal_response(sample_rag_output)
        
        if response.get('response'):
            print("✅ Response generation successful")
            print(f"Response length: {len(response['response'])} characters")
            generation_method = response.get('processing_metadata', {}).get('generation_method', 'unknown')
            print(f"Generation method: {generation_method}")
        else:
            print("❌ Response generation failed - no response content")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Response generator test failed: {e}")
        return False

def test_vector_store():
    """Test vector store if available"""
    print("\n🗂️  Testing Vector Store...")
    
    try:
        from vector_store import LegalVectorStore
        import sys
        sys.path.append('configs')
        from model_config import EMBEDDING_MODEL, VECTOR_DB_PATH
        
        if not os.path.exists(VECTOR_DB_PATH):
            print("❌ Vector store not found! Please run Phase 1 first.")
            return False
        
        # Initialize vector store
        vector_store = LegalVectorStore(
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

def main():
    """Main testing function"""
    print("🧪 Bengali Legal Advocate - Simplified Phase 2 Testing")
    print("=" * 65)
    
    # Run tests
    test_results = {}
    
    test_results['lm_studio_connection'] = test_lm_studio_connection()
    test_results['basic_imports'] = test_basic_imports()
    test_results['bengali_processing'] = test_bengali_processing()
    test_results['query_processing'] = test_query_processing()
    test_results['response_generator'] = test_response_generator()
    test_results['vector_store'] = test_vector_store()
    
    # Generate report
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
    if not test_results.get('vector_store'):
        print("  • Run Phase 1 to create vector store")
    
    # Overall result
    if passed_tests >= 4:  # At least 4 out of 6 tests should pass
        print("\n🎉 Core components are working! Ready for Phase 2.")
    else:
        print("\n⚠️  Some critical components failed. Please check the issues above.")
    
    return passed_tests >= 4

if __name__ == "__main__":
    main() 