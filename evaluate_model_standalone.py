#!/usr/bin/env python3
"""
Standalone Model Evaluation Script
Evaluate Bengali legal models independently with comprehensive metrics
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import required components
try:
    from src.model_evaluator import LegalModelEvaluator
    from src.bengali_processor import BengaliLegalProcessor
    from configs.training_config import Phase3TrainingConfig
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Make sure all Phase 3 components are properly implemented")
    sys.exit(1)

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'model_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
        ]
    )

def load_test_dataset(dataset_path):
    """Load test dataset from file"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Loading test dataset from: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.suffix == '.jsonl':
                test_data = []
                for line in f:
                    test_data.append(json.loads(line.strip()))
            else:
                test_data = json.load(f)
        
        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data
        
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate Bengali Legal Models - Standalone Script'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to the trained model directory'
    )
    
    parser.add_argument(
        '--test-dataset', '-t',
        type=str,
        required=True,
        help='Path to test dataset (JSON or JSONL file)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='evaluation_results',
        help='Output directory for results (default: evaluation_results)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Bengali Legal Model Evaluation - Standalone Script")
        
        # Load test dataset
        test_dataset = load_test_dataset(Path(args.test_dataset))
        
        # Initialize Bengali processor
        bengali_processor = BengaliLegalProcessor()
        
        # Initialize model evaluator
        evaluation_config = Phase3TrainingConfig.EVALUATION_CONFIG
        evaluator = LegalModelEvaluator(
            bengali_processor=bengali_processor,
            config=evaluation_config
        )
        
        # Evaluate model
        results = evaluator.evaluate_model(
            model_path=args.model_path,
            test_dataset=test_dataset,
            output_dir=args.output_dir
        )
        
        logger.info("Evaluation completed successfully")
        print(f"✓ Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 