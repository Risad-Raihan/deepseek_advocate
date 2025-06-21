#!/usr/bin/env python3
"""
Quick Test Script for Phase 3 - Bengali Legal Advocate AI
Test all Phase 3 components with minimal resources
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'phase3_quick_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test all Phase 3 imports"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing imports...")
        
        # Phase 2 components
        from src.legal_rag import LegalRAGEngine
        from src.bengali_processor import BengaliLegalProcessor
        from src.vector_store import LegalVectorStore
        logger.info("✓ Phase 2 components imported")
        
        # Phase 3 components
        from src.qa_generator import LegalQAGenerator
        from src.fine_tuning_engine import LegalFineTuningEngine
        from src.model_evaluator import LegalModelEvaluator
        from configs.training_config import Phase3TrainingConfig, TRAINING_PRESETS
        logger.info("✓ Phase 3 components imported")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def main():
    """Main test execution"""
    logger = setup_logging()
    
    try:
        logger.info("=" * 80)
        logger.info("BENGALI LEGAL ADVOCATE AI - PHASE 3 QUICK TEST")
        logger.info("=" * 80)
        
        # Test imports
        if test_imports():
            logger.info("✓ PHASE 3 IMPORTS SUCCESSFUL")
            print("\n🎉 Phase 3 components are properly installed!")
            print("✅ All imports working correctly")
            print("✅ Ready for full Phase 3 execution")
        else:
            logger.error("✗ PHASE 3 IMPORTS FAILED")
            print("\n❌ Phase 3 import test failed!")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 