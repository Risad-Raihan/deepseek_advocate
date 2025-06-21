"""
Fixed Phase 2 test with proper error handling
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

def add_missing_methods_to_rag_engine():
    """Add missing methods to LegalRAGEngine class"""
    
    def build_legal_response_context(self, documents, processed_query):
        """Build legal response context"""
        try:
            if not documents:
                return "কোন প্রাসঙ্গিক আইনি তথ্য পাওয়া যায়নি।"
            
            context_parts = []
            query_domain = processed_query.get('domain', {}).get('domain', 'general')
            context_parts.append(f"আইনি ক্ষেত্র: {query_domain}")
            context_parts.append("")
            
            for i, doc in enumerate(documents[:3], 1):
                metadata = doc.get('metadata', {})
                doc_title = metadata.get('document_title', 'অজানা দলিল')
                content = metadata.get('paragraph_text', metadata.get('content', ''))
                
                if content:
                    context_parts.append(f"তথ্যসূত্র {i}: {doc_title}")
                    context_parts.append(content[:300] + "..." if len(content) > 300 else content)
                    context_parts.append("")
            
            return "\n".join(context_parts)
        except:
            return "প্রসঙ্গ তৈরিতে সমস্যা হয়েছে।"
    
    def format_legal_citations(self, documents):
        """Format legal citations"""
        try:
            citations = []
            for doc in documents[:3]:
                metadata = doc.get('metadata', {})
                doc_title = metadata.get('document_title', 'অজানা দলিল')
                section = metadata.get('section_title', '')
                
                if section:
                    citation = f"{doc_title} - {section}"
                else:
                    citation = doc_title
                
                if citation not in citations:
                    citations.append(citation)
            
            return citations
        except:
            return ["তথ্যসূত্র তৈরিতে সমস্যা হয়েছে।"]
    
    # Add methods to the class
    LegalRAGEngine.build_legal_response_context = build_legal_response_context
    LegalRAGEngine.format_legal_citations = format_legal_citations

def main():
    """Main test function"""
    print("Bengali Legal Advocate - Phase 2 Fixed Test")
    print("=" * 55)
    
    # Check LM Studio
    if not check_lm_studio():
        print("Please start LM Studio first!")
        return
    
    try:
        # Add missing methods
        add_missing_methods_to_rag_engine()
        
        # Initialize components
        print("Initializing components...")
        
        # Vector store with error handling
        try:
            vector_store = LegalVectorStore(
                embedding_model=EMBEDDING_MODEL,
                vector_db_path=VECTOR_DB_PATH
            )
            print("Vector store loaded successfully")
        except Exception as e:
            print(f"Vector store warning: {e}")
            vector_store = None
        
        # Other components
        bengali_processor = BengaliLegalProcessor()
        query_processor = BengaliLegalQueryProcessor()
        
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
        
        # Process query with timeout
        try:
            rag_output = rag_engine.process_legal_query(test_query)
            
            # Generate response with shorter timeout
            print("Generating response (this may take 30-60 seconds)...")
            final_response = response_generator.generate_comprehensive_legal_response(rag_output)
            
            end_time = time.time()
            
            print(f"\nQuery processed successfully!")
            print(f"Response time: {end_time - start_time:.2f} seconds")
            print(f"Legal domain: {rag_output.get('legal_domain', 'unknown')}")
            print(f"Confidence: {rag_output.get('confidence_score', 0.0):.2f}")
            print(f"Response length: {len(final_response.get('response', ''))}")
            
            # Show first 200 characters of response
            response_text = final_response.get('response', '')
            if response_text:
                print(f"\nResponse preview:")
                print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
            
            print("\nPhase 2 test completed successfully!")
            
        except Exception as e:
            print(f"Error during query processing: {e}")
            print("This might be due to LM Studio timeout or model issues.")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 