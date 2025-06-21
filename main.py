#!/usr/bin/env python3
"""
Bengali Legal Advocate - Main Execution Script
Handles document processing, vector store creation, and system initialization
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import LegalDocumentProcessor
from src.bengali_processor import BengaliLegalProcessor
from src.vector_store import LegalVectorStore
from configs.model_config import ModelConfig

class BengaliLegalAdvocateSystem:
    """Main system orchestrator for Bengali Legal Advocate"""
    
    def __init__(self):
        self.setup_logging()
        self.config = ModelConfig()
        self.config.ensure_directories()
        
        # Initialize processors
        self.document_processor = LegalDocumentProcessor(
            data_dir=str(self.config.DATA_DIR)
        )
        self.bengali_processor = BengaliLegalProcessor()
        self.vector_store = LegalVectorStore(
            embedding_model=self.config.EMBEDDING_MODEL,
            vector_db_path=str(self.config.VECTOR_DB_DIR)
        )
        
    def setup_logging(self):
        """Setup system-wide logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('legal_advocate_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_phase_1(self):
        """
        Execute Phase 1: Document Processing & Vector Store Creation
        """
        self.logger.info("🚀 Starting Phase 1: Document Processing & Vector Store Creation")
        
        try:
            # Step 1: Process legal PDFs
            self.logger.info("Step 1: Processing Bengali legal PDFs...")
            processed_docs = self.document_processor.process_legal_pdfs(
                output_dir=str(self.config.TRAINING_DATA_DIR)
            )
            
            if processed_docs['total_processed'] == 0:
                self.logger.error("❌ No documents were successfully processed!")
                return False
            
            self.logger.info(f"✅ Successfully processed {processed_docs['total_processed']} documents")
            
            # Print processing summary
            self._print_processing_summary(processed_docs)
            
            # Step 2: Create multi-level vector indexes
            self.logger.info("Step 2: Creating multi-level FAISS indexes...")
            index_stats = self.vector_store.create_multi_level_index(processed_docs)
            
            self.logger.info(f"✅ Created vector indexes: {index_stats['index_sizes']}")
            
            # Step 3: Test retrieval system
            self.logger.info("Step 3: Testing retrieval system...")
            self._test_retrieval_system()
            
            # Step 4: Generate system report
            self._generate_phase1_report(processed_docs, index_stats)
            
            self.logger.info("🎉 Phase 1 completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Phase 1 failed: {e}")
            return False
    
    def _print_processing_summary(self, processed_docs):
        """Print detailed processing summary"""
        print("\n" + "="*60)
        print("📊 DOCUMENT PROCESSING SUMMARY")
        print("="*60)
        
        print(f"Total Documents Processed: {processed_docs['total_processed']}")
        print(f"Total Errors: {len(processed_docs.get('errors', []))}")
        
        if processed_docs.get('errors'):
            print("\n❌ Processing Errors:")
            for error in processed_docs['errors']:
                print(f"  - {error}")
        
        print("\n📄 Document Types Identified:")
        doc_types = {}
        for doc_id, doc_data in processed_docs.get('documents', {}).items():
            doc_type = doc_data.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        for doc_type, count in doc_types.items():
            print(f"  - {doc_type}: {count} documents")
        
        print("\n📈 Text Extraction Statistics:")
        total_chars = sum(
            doc_data.get('text_length', 0) 
            for doc_data in processed_docs.get('documents', {}).values()
        )
        avg_chars = total_chars / max(processed_docs['total_processed'], 1)
        
        print(f"  - Total Characters: {total_chars:,}")
        print(f"  - Average per Document: {avg_chars:,.0f}")
        
        print("="*60)
    
    def _test_retrieval_system(self):
        """Test the retrieval system with sample queries"""
        test_queries = [
            "তালাকের আইনি প্রক্রিয়া কী?",
            "বাড়ি ভাড়া বৃদ্ধির নিয়ম কী?",
            "সংবিধানের মৌলিক অধিকার কী কী?",
            "মামলা দায়ের করার পদ্ধতি কী?",
            "সম্পত্তির উত্তরাধিকার আইন কী?"
        ]
        
        print("\n" + "="*60)
        print("🔍 TESTING RETRIEVAL SYSTEM")
        print("="*60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest Query {i}: {query}")
            
            try:
                # Test paragraph-level search
                results = self.vector_store.hybrid_search(
                    query=query,
                    level='paragraph',
                    top_k=3
                )
                
                if results:
                    print(f"  ✅ Found {len(results)} relevant results")
                    best_result = results[0]
                    print(f"  📊 Best match score: {best_result['combined_score']:.3f}")
                    
                    # Show metadata if available
                    metadata = best_result.get('metadata', {})
                    if metadata:
                        if 'paragraph_text' in metadata:
                            preview = metadata['paragraph_text'][:100] + "..."
                            print(f"  📝 Preview: {preview}")
                else:
                    print("  ❌ No results found")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("="*60)
    
    def _generate_phase1_report(self, processed_docs, index_stats):
        """Generate comprehensive Phase 1 report"""
        report_path = self.config.TRAINING_DATA_DIR / "phase1_report.json"
        
        report_data = {
            "phase": "Phase 1 - Document Processing & Vector Store",
            "completion_date": datetime.now().isoformat(),
            "processing_summary": {
                "total_documents": processed_docs['total_processed'],
                "total_errors": len(processed_docs.get('errors', [])),
                "document_types": {},
                "total_text_length": 0
            },
            "vector_store_summary": {
                "index_levels": list(index_stats.get('index_sizes', {}).keys()),
                "total_vectors": sum(index_stats.get('index_sizes', {}).values()),
                "index_sizes": index_stats.get('index_sizes', {}),
                "embedding_model": self.config.EMBEDDING_MODEL,
                "embedding_dimension": self.config.EMBEDDING_DIM
            },
            "next_steps": [
                "Phase 2: Implement Legal RAG System",
                "Phase 3: Setup Fine-tuning Pipeline", 
                "Phase 4: Create Hybrid Advocate System"
            ]
        }
        
        # Calculate document type distribution
        for doc_id, doc_data in processed_docs.get('documents', {}).items():
            doc_type = doc_data.get('doc_type', 'unknown')
            report_data["processing_summary"]["document_types"][doc_type] = \
                report_data["processing_summary"]["document_types"].get(doc_type, 0) + 1
            report_data["processing_summary"]["total_text_length"] += doc_data.get('text_length', 0)
        
        # Save report
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 Phase 1 report saved to: {report_path}")
        
        # Print summary
        print(f"\n📊 PHASE 1 COMPLETION REPORT")
        print("="*60)
        print(f"✅ Documents Processed: {report_data['processing_summary']['total_documents']}")
        print(f"✅ Vector Indexes Created: {len(report_data['vector_store_summary']['index_levels'])}")
        print(f"✅ Total Vectors: {report_data['vector_store_summary']['total_vectors']:,}")
        print(f"✅ Report Saved: {report_path}")
        print("="*60)

def main():
    """Main execution function"""
    print("🏛️ Bengali Legal Advocate AI System")
    print("Advanced Legal AI using Hybrid RAG + Fine-tuning")
    print("="*60)
    
    # Initialize system
    system = BengaliLegalAdvocateSystem()
    
    # Validate configuration
    config_validation = ModelConfig.validate_config()
    if not config_validation['valid']:
        print("❌ Configuration validation failed:")
        for error in config_validation['errors']:
            print(f"  - {error}")
        return 1
    
    if config_validation['warnings']:
        print("⚠️ Configuration warnings:")
        for warning in config_validation['warnings']:
            print(f"  - {warning}")
    
    # Run Phase 1
    success = system.run_phase_1()
    
    if success:
        print("\n🎉 Phase 1 completed successfully!")
        print("Next: Run Phase 2 to implement Legal RAG System")
        return 0
    else:
        print("\n❌ Phase 1 failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 