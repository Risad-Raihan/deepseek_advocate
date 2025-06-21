"""
Simple Phase 2 test without Unicode characters to avoid Windows console issues
"""

import os
import sys
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.vector_store import LegalVectorStore
from src.bengali_processor import BengaliLegalProcessor
from src.query_processor import BengaliLegalQueryProcessor
from src.legal_rag import LegalRAGEngine
from src.context_builder import LegalContextBuilder
from src.response_generator import BengaliLegalResponseGenerator
from configs.model_config import *

def check_lm_studio():
    """Check LM Studio connection"""
    try:
        import requests
        response = requests.get(f"{LM_STUDIO_CONFIG['base_url']}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("LM Studio is running!")
            print(f"Available models: {[model.get('id', 'Unknown') for model in models.get('data', [])]}")
            return True
        return False
    except Exception as e:
        print(f"LM Studio connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("Bengali Legal Advocate - Phase 2 Test")
    print("=" * 50)
    
    # Check LM Studio
    if not check_lm_studio():
        print("Please start LM Studio first!")
        return
    
    try:
        # Initialize components
        print("Initializing components...")
        
        # Vector store
        vector_store = LegalVectorStore(
            embedding_model=EMBEDDING_MODEL,
            vector_db_path=VECTOR_DB_PATH
        )
        
        # Bengali processor
        bengali_processor = BengaliLegalProcessor()
        
        # Query processor
        query_processor = BengaliLegalQueryProcessor()
        
        # Context builder
        context_builder = LegalContextBuilder(
            max_context_length=PHASE2_CONFIG['context_building']['max_context_length']
        )
        
        # Response generator
        response_generator = BengaliLegalResponseGenerator(
            lm_studio_url=LM_STUDIO_CONFIG['base_url'],
            model_name=LM_STUDIO_CONFIG['model_name']
        )
        
        # RAG engine
        rag_engine = LegalRAGEngine(
            vector_store=vector_store,
            bengali_processor=bengali_processor,
            query_processor=query_processor
        )
        
        print("All components initialized successfully!")
        
        # Test with a simple query
        test_query = "তালাকের পর খোরপোশের নিয়ম কি?"
        print(f"\nTesting query: {test_query}")
        
        start_time = time.time()
        
        # Process query
        rag_output = rag_engine.process_legal_query(test_query)
        final_response = response_generator.generate_comprehensive_legal_response(rag_output)
        
        end_time = time.time()
        
        print(f"\nQuery processed successfully!")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Legal domain: {rag_output.get('legal_domain', 'unknown')}")
        print(f"Confidence: {rag_output.get('confidence_score', 0.0):.2f}")
        print(f"Response length: {len(final_response.get('response', ''))}")
        
        print("\nPhase 2 test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 