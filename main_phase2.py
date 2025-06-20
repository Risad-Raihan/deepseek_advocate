"""
Phase 2: Legal RAG System Implementation
Main execution script for Bengali Legal Advocate with LM Studio Integration
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.vector_store import VectorStore
from src.bengali_processor import BengaliProcessor
from src.query_processor import LegalQueryProcessor
from src.legal_rag import LegalRAGEngine
from src.context_builder import LegalContextBuilder
from src.response_generator import BengaliLegalResponseGenerator
from configs.model_config import *

def setup_logging():
    """Setup comprehensive logging for Phase 2"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/phase2_execution.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def check_lm_studio_availability():
    """Check if LM Studio is running and accessible"""
    try:
        import requests
        response = requests.get(
            f"{LM_STUDIO_CONFIG['base_url']}/models",
            timeout=5
        )
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio is running!")
            print(f"Available models: {[model.get('id', 'Unknown') for model in models.get('data', [])]}")
            return True
        else:
            print(f"❌ LM Studio responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        print("Please make sure LM Studio is running on http://localhost:1234")
        return False

def initialize_phase2_components(logger):
    """Initialize all Phase 2 components"""
    try:
        logger.info("Initializing Phase 2 components...")
        
        # Initialize vector store (from Phase 1)
        logger.info("Loading vector store...")
        vector_store = VectorStore(
            embedding_model=EMBEDDING_MODEL,
            vector_db_path=VECTOR_DB_PATH
        )
        
        # Check if vector store exists
        if not os.path.exists(VECTOR_DB_PATH):
            logger.error("Vector store not found! Please run Phase 1 first.")
            return None
        
        # Initialize Bengali processor
        logger.info("Initializing Bengali processor...")
        bengali_processor = BengaliProcessor()
        
        # Initialize query processor
        logger.info("Initializing query processor...")
        query_processor = LegalQueryProcessor(bengali_processor)
        
        # Initialize context builder
        logger.info("Initializing context builder...")
        context_builder = LegalContextBuilder(
            max_context_length=PHASE2_CONFIG['context_building']['max_context_length']
        )
        
        # Initialize response generator
        logger.info("Initializing response generator...")
        response_generator = BengaliLegalResponseGenerator(
            lm_studio_url=LM_STUDIO_CONFIG['base_url'],
            model_name=LM_STUDIO_CONFIG['model_name']
        )
        
        # Initialize RAG engine
        logger.info("Initializing Legal RAG engine...")
        rag_engine = LegalRAGEngine(
            vector_store=vector_store,
            bengali_processor=bengali_processor,
            query_processor=query_processor
        )
        
        components = {
            'vector_store': vector_store,
            'bengali_processor': bengali_processor,
            'query_processor': query_processor,
            'context_builder': context_builder,
            'response_generator': response_generator,
            'rag_engine': rag_engine
        }
        
        logger.info("✅ All Phase 2 components initialized successfully!")
        return components
        
    except Exception as e:
        logger.error(f"❌ Error initializing Phase 2 components: {e}")
        return None

def test_phase2_system(components, logger):
    """Test Phase 2 system with sample queries"""
    try:
        logger.info("Testing Phase 2 system...")
        
        rag_engine = components['rag_engine']
        response_generator = components['response_generator']
        
        test_queries = TESTING_CONFIG['test_queries']
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            logger.info(f"Testing query {i}/{len(test_queries)}: {test_case['query'][:50]}...")
            
            start_time = time.time()
            
            # Process query through RAG system
            rag_output = rag_engine.process_legal_query(test_case['query'])
            
            # Generate response
            final_response = response_generator.generate_comprehensive_legal_response(rag_output)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Evaluate results
            test_result = {
                'query': test_case['query'],
                'expected_domain': test_case['expected_domain'],
                'actual_domain': rag_output.get('legal_domain', 'unknown'),
                'response_time': response_time,
                'confidence_score': rag_output.get('confidence_score', 0.0),
                'documents_retrieved': len(rag_output.get('retrieved_context', {}).get('documents', [])),
                'response_length': len(final_response.get('response', '')),
                'has_citations': len(rag_output.get('citations', [])) > 0,
                'success': True
            }
            
            results.append(test_result)
            
            # Print results
            print(f"\n📊 Test {i} Results:")
            print(f"Query: {test_case['query']}")
            print(f"Expected Domain: {test_case['expected_domain']}")
            print(f"Actual Domain: {test_result['actual_domain']}")
            print(f"Response Time: {response_time:.2f}s")
            print(f"Confidence: {test_result['confidence_score']:.2f}")
            print(f"Documents Retrieved: {test_result['documents_retrieved']}")
            print(f"Response Preview: {final_response.get('response', '')[:200]}...")
            print("-" * 80)
        
        # Calculate overall performance
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        avg_confidence = sum(r['confidence_score'] for r in results) / len(results)
        domain_accuracy = sum(1 for r in results if r['expected_domain'] == r['actual_domain']) / len(results)
        
        logger.info(f"✅ Phase 2 testing completed successfully!")
        logger.info(f"Average Response Time: {avg_response_time:.2f}s")
        logger.info(f"Average Confidence: {avg_confidence:.2f}")
        logger.info(f"Domain Classification Accuracy: {domain_accuracy:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error testing Phase 2 system: {e}")
        return []

def interactive_query_interface(components, logger):
    """Interactive interface for testing queries"""
    try:
        logger.info("Starting interactive query interface...")
        
        rag_engine = components['rag_engine']
        response_generator = components['response_generator']
        
        print("\n🎯 Bengali Legal Advocate - Interactive Mode")
        print("=" * 60)
        print("Enter your Bengali legal questions (type 'quit' to exit)")
        print("Example: তালাকের পর খোরপোশের নিয়ম কি?")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                query = input("\n❓ আপনার প্রশ্ন: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 ধন্যবাদ! Legal Advocate বন্ধ করা হচ্ছে...")
                    break
                
                if not query:
                    continue
                
                print(f"\n🔍 প্রক্রিয়া করা হচ্ছে: {query}")
                print("⏳ অনুগ্রহ করে অপেক্ষা করুন...")
                
                start_time = time.time()
                
                # Process query
                rag_output = rag_engine.process_legal_query(query)
                final_response = response_generator.generate_comprehensive_legal_response(rag_output)
                
                end_time = time.time()
                
                # Display results
                print("\n" + "="*80)
                print("📋 আইনি পরামর্শ:")
                print("="*80)
                print(final_response.get('response', 'উত্তর তৈরি করতে সমস্যা হয়েছে।'))
                
                # Display metadata
                print(f"\n📊 তথ্য:")
                print(f"• আইনি ক্ষেত্র: {rag_output.get('legal_domain', 'সাধারণ')}")
                print(f"• আস্থার স্কোর: {rag_output.get('confidence_score', 0.0):.2f}")
                print(f"• উত্তরের সময়: {end_time - start_time:.2f} সেকেন্ড")
                print(f"• প্রাপ্ত দলিল: {len(rag_output.get('retrieved_context', {}).get('documents', []))}")
                
                # Display citations if available
                citations = rag_output.get('citations', [])
                if citations:
                    print(f"\n📚 তথ্যসূত্র:")
                    for i, citation in enumerate(citations[:3], 1):
                        print(f"{i}. {citation}")
                
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\n👋 Legal Advocate বন্ধ করা হচ্ছে...")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"❌ ত্রুটি: {e}")
                print("দয়া করে আবার চেষ্টা করুন।")
    
    except Exception as e:
        logger.error(f"Error in interactive interface: {e}")
        print(f"❌ ইন্টারঅ্যাক্টিভ মোড শুরু করতে সমস্যা: {e}")

def main():
    """Main execution function for Phase 2"""
    print("🚀 Bengali Legal Advocate - Phase 2: Legal RAG System")
    print("=" * 70)
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging()
    
    # Check LM Studio availability
    print("\n📡 Checking LM Studio availability...")
    if not check_lm_studio_availability():
        print("\n⚠️  LM Studio is not running. Please:")
        print("1. Start LM Studio")
        print("2. Load the DeepSeek model")
        print("3. Start the server (usually runs on port 1234)")
        print("4. Try again")
        return
    
    # Initialize components
    print("\n🔧 Initializing Phase 2 components...")
    components = initialize_phase2_components(logger)
    
    if not components:
        print("❌ Failed to initialize Phase 2 components!")
        return
    
    # Test system
    print("\n🧪 Testing Phase 2 system...")
    test_results = test_phase2_system(components, logger)
    
    if not test_results:
        print("❌ System testing failed!")
        return
    
    # Start interactive interface
    print("\n🎯 Starting interactive query interface...")
    interactive_query_interface(components, logger)
    
    print("\n✅ Phase 2 execution completed successfully!")
    logger.info("Phase 2 execution completed successfully!")

if __name__ == "__main__":
    main() 