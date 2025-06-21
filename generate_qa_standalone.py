#!/usr/bin/env python3
"""
Standalone Q&A Dataset Generation Script
Generate Bengali legal Q&A pairs independently of the full training pipeline
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
    from src.legal_rag import LegalRAGEngine
    from src.bengali_processor import BengaliLegalProcessor
    from src.vector_store import LegalVectorStore
    from src.qa_generator import LegalQAGenerator
    from configs.training_config import Phase3TrainingConfig
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Make sure all Phase 2 and 3 components are properly implemented")
    sys.exit(1)

def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'qa_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
        ]
    )

def initialize_components():
    """Initialize Phase 2 components for Q&A generation"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing components...")
        
        # Initialize Bengali processor
        bengali_processor = BengaliLegalProcessor()
        logger.info("✓ Bengali processor initialized")
        
        # Initialize vector store
        vector_store = LegalVectorStore()
        logger.info("✓ Vector store initialized")
        
        # Initialize RAG engine
        legal_rag = LegalRAGEngine(
            vector_store=vector_store,
            bengali_processor=bengali_processor
        )
        logger.info("✓ Legal RAG engine initialized")
        
        return legal_rag, bengali_processor, vector_store
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

def generate_qa_dataset(num_pairs, quality_threshold, output_file, export_format):
    """Generate Q&A dataset with specified parameters"""
    logger = logging.getLogger(__name__)
    
    # Initialize components
    legal_rag, bengali_processor, vector_store = initialize_components()
    
    # Initialize Q&A generator
    config = Phase3TrainingConfig.QA_GENERATION_CONFIG
    qa_generator = LegalQAGenerator(
        legal_rag=legal_rag,
        bengali_processor=bengali_processor,
        vector_store=vector_store,
        config=config
    )
    
    logger.info(f"Starting Q&A generation: {num_pairs} pairs, quality threshold: {quality_threshold}")
    
    # Generate Q&A pairs
    qa_pairs = qa_generator.generate_qa_dataset(
        num_pairs=num_pairs,
        quality_threshold=quality_threshold
    )
    
    # Export dataset
    if qa_pairs:
        success = qa_generator.export_qa_dataset(output_file, export_format)
        
        if success:
            # Get and display statistics
            stats = qa_generator.get_generation_stats()
            
            logger.info("=" * 60)
            logger.info("Q&A GENERATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total generated: {stats['total_generated']}")
            logger.info(f"Quality stats: {stats['quality_stats']}")
            logger.info(f"Average quality score: {stats['average_quality_score']:.3f}")
            
            logger.info("\nDomain distribution:")
            for domain, count in stats['domain_distribution'].items():
                logger.info(f"  {domain}: {count}")
            
            logger.info("\nQuestion type distribution:")
            for q_type, count in stats['question_type_distribution'].items():
                logger.info(f"  {q_type}: {count}")
            
            logger.info(f"\nDataset exported to: {output_file}")
            logger.info("=" * 60)
            
            # Save statistics separately
            stats_file = Path(output_file).with_suffix('.stats.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Statistics saved to: {stats_file}")
            
            return qa_pairs
        else:
            logger.error("Failed to export Q&A dataset")
            return None
    else:
        logger.error("No Q&A pairs generated")
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate Bengali Legal Q&A Dataset - Standalone Script'
    )
    
    parser.add_argument(
        '--num-pairs', '-n',
        type=int,
        default=1000,
        help='Number of Q&A pairs to generate (default: 1000)'
    )
    
    parser.add_argument(
        '--quality-threshold', '-q',
        type=float,
        default=0.8,
        help='Quality threshold for Q&A pairs (0.0-1.0, default: 0.8)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='qa_dataset.json',
        help='Output file path (default: qa_dataset.json)'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'jsonl'],
        default='json',
        help='Export format (default: json)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Bengali Legal Q&A Generation - Standalone Script")
        logger.info(f"Parameters: {args.num_pairs} pairs, quality {args.quality_threshold}, format {args.format}")
        
        # Generate Q&A dataset
        qa_pairs = generate_qa_dataset(
            num_pairs=args.num_pairs,
            quality_threshold=args.quality_threshold,
            output_file=args.output,
            export_format=args.format
        )
        
        if qa_pairs:
            print(f"\n✓ Successfully generated {len(qa_pairs)} Q&A pairs")
            print(f"✓ Dataset saved to: {args.output}")
        else:
            print("\n✗ Q&A generation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 