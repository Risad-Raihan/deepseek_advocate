#!/usr/bin/env python3
"""
Bengali Legal Advocate AI - Phase 3 Main Execution Script
Complete fine-tuning pipeline with Q&A generation, LoRA training, and evaluation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import traceback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import Phase 2 components
try:
    from src.legal_rag import LegalRAGEngine
    from src.bengali_processor import BengaliLegalProcessor
    from src.vector_store import LegalVectorStore
    from src.response_generator import BengaliLegalResponseGenerator
except ImportError as e:
    print(f"Error importing Phase 2 components: {e}")
    print("Make sure Phase 2 is properly implemented and accessible")
    sys.exit(1)

# Import Phase 3 components
try:
    from src.qa_generator import LegalQAGenerator
    from src.fine_tuning_engine import LegalFineTuningEngine
    from src.model_evaluator import LegalModelEvaluator
    from configs.training_config import Phase3TrainingConfig, TRAINING_PRESETS
except ImportError as e:
    print(f"Error importing Phase 3 components: {e}")
    print("Make sure all Phase 3 components are properly implemented")
    sys.exit(1)

class Phase3ExecutionPipeline:
    """Main execution pipeline for Phase 3 fine-tuning"""
    
    def __init__(self, config_preset='development', custom_config=None):
        """
        Initialize Phase 3 execution pipeline
        
        Args:
            config_preset: Training preset ('quick_test', 'development', 'production', 'high_quality')
            custom_config: Custom configuration overrides
        """
        self.config = Phase3TrainingConfig
        self.config_preset = config_preset
        self.custom_config = custom_config or {}
        
        # Apply preset configuration
        self._apply_config_preset()
        
        # Apply custom overrides
        if self.custom_config:
            self.config.override_config(self.custom_config)
        
        # Validate configuration
        self._validate_configuration()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.phase2_components = {}
        self.phase3_components = {}
        
        # Execution state
        self.execution_state = {
            'phase': 'initialization',
            'start_time': datetime.now(),
            'current_step': 0,
            'total_steps': 7,  # Q&A gen, data prep, model init, training, eval, save, report
            'errors': [],
            'warnings': []
        }
        
        # Results storage
        self.results = {
            'qa_generation': {},
            'training': {},
            'evaluation': {},
            'model_info': {}
        }
    
    def _apply_config_preset(self):
        """Apply configuration preset"""
        if self.config_preset in TRAINING_PRESETS:
            preset_config = {'training_config': TRAINING_PRESETS[self.config_preset]}
            self.config.override_config(preset_config)
            self.logger.info(f"Applied configuration preset: {self.config_preset}")
        else:
            self.logger.warning(f"Unknown preset: {self.config_preset}, using default")
    
    def _validate_configuration(self):
        """Validate configuration before execution"""
        validation_results = self.config.validate_config()
        
        if not validation_results['valid']:
            for error in validation_results['errors']:
                print(f"Configuration Error: {error}")
            sys.exit(1)
        
        for warning in validation_results['warnings']:
            print(f"Configuration Warning: {warning}")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Ensure logging directory exists
        log_dir = self.config.get_logging_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('Phase3Pipeline')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        log_file = log_dir / f"phase3_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("=" * 80)
        self.logger.info("Bengali Legal Advocate AI - Phase 3 Execution Started")
        self.logger.info(f"Configuration preset: {self.config_preset}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 80)
    
    def initialize_phase2_components(self):
        """Initialize required Phase 2 components"""
        try:
            self.logger.info("Initializing Phase 2 components...")
            self.execution_state['phase'] = 'phase2_initialization'
            
            # Initialize Bengali processor
            self.phase2_components['bengali_processor'] = BengaliLegalProcessor()
            self.logger.info("✓ Bengali processor initialized")
            
            # Initialize vector store
            try:
                self.phase2_components['vector_store'] = LegalVectorStore()
                self.logger.info("✓ Vector store initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize vector store: {e}")
                raise
            
            # Initialize RAG engine
            try:
                self.phase2_components['legal_rag'] = LegalRAGEngine(
                    vector_store=self.phase2_components['vector_store'],
                    bengali_processor=self.phase2_components['bengali_processor']
                )
                self.logger.info("✓ Legal RAG engine initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize RAG engine: {e}")
                raise
            
            self.logger.info("Phase 2 components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Phase 2 components: {e}")
            self.execution_state['errors'].append(f"Phase 2 initialization failed: {e}")
            return False
    
    def generate_qa_dataset(self):
        """Generate Q&A dataset for training"""
        try:
            self.logger.info("Starting Q&A dataset generation...")
            self.execution_state['phase'] = 'qa_generation'
            self.execution_state['current_step'] = 1
            
            # Initialize Q&A generator
            qa_generator = LegalQAGenerator(
                legal_rag=self.phase2_components['legal_rag'],
                bengali_processor=self.phase2_components['bengali_processor'],
                vector_store=self.phase2_components['vector_store'],
                config=self.config.QA_GENERATION_CONFIG
            )
            
            self.phase3_components['qa_generator'] = qa_generator
            
            # Generate Q&A pairs
            qa_config = self.config.QA_GENERATION_CONFIG
            qa_pairs = qa_generator.generate_qa_dataset(
                num_pairs=qa_config['num_qa_pairs'],
                quality_threshold=qa_config['quality_threshold']
            )
            
            # Export dataset
            export_dir = self.config.TRAINING_DATA_DIR / "phase3_qa_dataset"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Export in multiple formats
            for fmt in qa_config['export_formats']:
                export_file = export_dir / f"legal_qa_dataset.{fmt}"
                qa_generator.export_qa_dataset(str(export_file), format=fmt)
            
            # Get generation statistics
            generation_stats = qa_generator.get_generation_stats()
            self.results['qa_generation'] = generation_stats
            
            self.logger.info(f"Q&A dataset generation completed:")
            self.logger.info(f"- Generated pairs: {generation_stats['total_generated']}")
            self.logger.info(f"- Average quality: {generation_stats['average_quality_score']:.3f}")
            self.logger.info(f"- Domain distribution: {generation_stats['domain_distribution']}")
            
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"Error generating Q&A dataset: {e}")
            self.logger.error(traceback.format_exc())
            self.execution_state['errors'].append(f"Q&A generation failed: {e}")
            raise
    
    def initialize_fine_tuning_engine(self):
        """Initialize fine-tuning engine"""
        try:
            self.logger.info("Initializing fine-tuning engine...")
            self.execution_state['phase'] = 'fine_tuning_initialization'
            self.execution_state['current_step'] = 2
            
            # Initialize fine-tuning engine
            fine_tuning_engine = LegalFineTuningEngine(
                config=self.config,
                training_config=self.config
            )
            
            self.phase3_components['fine_tuning_engine'] = fine_tuning_engine
            
            # Initialize model and tokenizer
            if not fine_tuning_engine.initialize_model_and_tokenizer():
                raise Exception("Failed to initialize model and tokenizer")
            
            # Setup LoRA
            if not fine_tuning_engine.setup_lora_model():
                raise Exception("Failed to setup LoRA model")
            
            self.logger.info("Fine-tuning engine initialized successfully")
            return fine_tuning_engine
            
        except Exception as e:
            self.logger.error(f"Error initializing fine-tuning engine: {e}")
            self.logger.error(traceback.format_exc())
            self.execution_state['errors'].append(f"Fine-tuning initialization failed: {e}")
            raise
    
    def execute_training(self, qa_pairs):
        """Execute the fine-tuning process"""
        try:
            self.logger.info("Starting fine-tuning process...")
            self.execution_state['phase'] = 'training'
            self.execution_state['current_step'] = 3
            
            fine_tuning_engine = self.phase3_components['fine_tuning_engine']
            
            # Prepare datasets
            self.logger.info("Preparing training datasets...")
            train_dataset, val_dataset, test_dataset = fine_tuning_engine.prepare_dataset(qa_pairs)
            
            # Store test dataset for evaluation
            self.test_dataset = test_dataset
            
            # Setup trainer
            if not fine_tuning_engine.setup_trainer(train_dataset, val_dataset):
                raise Exception("Failed to setup trainer")
            
            # Execute training
            self.logger.info("Executing fine-tuning...")
            training_results = fine_tuning_engine.train(train_dataset, val_dataset)
            
            # Store training results
            self.results['training'] = training_results
            
            # Get training statistics
            training_stats = fine_tuning_engine.get_training_statistics()
            self.results['model_info'] = training_stats
            
            self.logger.info("Fine-tuning completed successfully:")
            self.logger.info(f"- Training loss: {training_results['training_loss']:.4f}")
            self.logger.info(f"- Training duration: {training_results['training_duration']}")
            self.logger.info(f"- Total steps: {training_results['total_steps']}")
            
            return fine_tuning_engine, test_dataset
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            self.logger.error(traceback.format_exc())
            self.execution_state['errors'].append(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, fine_tuning_engine, test_dataset):
        """Evaluate the fine-tuned model"""
        try:
            self.logger.info("Starting model evaluation...")
            self.execution_state['phase'] = 'evaluation'
            self.execution_state['current_step'] = 4
            
            # Initialize evaluator
            evaluator = LegalModelEvaluator(
                config=self.config,
                bengali_processor=self.phase2_components['bengali_processor']
            )
            
            self.phase3_components['evaluator'] = evaluator
            
            # Perform comprehensive evaluation
            evaluation_results = evaluator.evaluate_model_comprehensive(
                model=fine_tuning_engine.peft_model,
                tokenizer=fine_tuning_engine.tokenizer,
                test_dataset=test_dataset
            )
            
            # Store evaluation results
            self.results['evaluation'] = evaluation_results
            
            # Save evaluation results
            eval_dir = self.config.get_output_dir() / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            eval_file = eval_dir / "evaluation_results.json"
            evaluator.save_evaluation_results(eval_file)
            
            # Generate and save evaluation report
            evaluation_report = evaluator.generate_evaluation_report()
            report_file = eval_dir / "evaluation_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(evaluation_report)
            
            # Log key metrics
            composite_scores = evaluation_results.get('composite_scores', {})
            self.logger.info("Model evaluation completed:")
            self.logger.info(f"- Overall score: {composite_scores.get('overall_score', 0):.3f}")
            self.logger.info(f"- Bengali score: {composite_scores.get('bengali_score', 0):.3f}")
            self.logger.info(f"- Legal score: {composite_scores.get('legal_score', 0):.3f}")
            self.logger.info(f"- Citation score: {composite_scores.get('citation_score', 0):.3f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            self.logger.error(traceback.format_exc())
            self.execution_state['errors'].append(f"Evaluation failed: {e}")
            raise
    
    def save_final_results(self):
        """Save final execution results"""
        try:
            self.logger.info("Saving final results...")
            self.execution_state['phase'] = 'saving_results'
            self.execution_state['current_step'] = 5
            
            # Prepare final results
            final_results = {
                'execution_info': {
                    'start_time': self.execution_state['start_time'].isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration': str(datetime.now() - self.execution_state['start_time']),
                    'config_preset': self.config_preset,
                    'success': len(self.execution_state['errors']) == 0
                },
                'execution_state': self.execution_state,
                'configuration': self.config.get_full_config(),
                'results': self.results
            }
            
            # Save to output directory
            output_dir = self.config.get_output_dir()
            results_file = output_dir / "phase3_results.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Final results saved to: {results_file}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error saving final results: {e}")
            self.execution_state['errors'].append(f"Results saving failed: {e}")
    
    def generate_execution_report(self):
        """Generate comprehensive execution report"""
        try:
            self.logger.info("Generating execution report...")
            self.execution_state['phase'] = 'reporting'
            self.execution_state['current_step'] = 6
            
            report_lines = []
            
            # Header
            report_lines.append("# Bengali Legal Advocate AI - Phase 3 Execution Report")
            report_lines.append(f"**Execution Date:** {self.execution_state['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"**Configuration Preset:** {self.config_preset}")
            report_lines.append("")
            
            # Execution Summary
            duration = datetime.now() - self.execution_state['start_time']
            success = len(self.execution_state['errors']) == 0
            
            report_lines.append("## Execution Summary")
            report_lines.append(f"- **Status:** {'✅ SUCCESS' if success else '❌ FAILED'}")
            report_lines.append(f"- **Duration:** {duration}")
            report_lines.append(f"- **Steps Completed:** {self.execution_state['current_step']}/{self.execution_state['total_steps']}")
            
            if self.execution_state['errors']:
                report_lines.append("- **Errors:**")
                for error in self.execution_state['errors']:
                    report_lines.append(f"  - {error}")
            
            if self.execution_state['warnings']:
                report_lines.append("- **Warnings:**")
                for warning in self.execution_state['warnings']:
                    report_lines.append(f"  - {warning}")
            
            report_lines.append("")
            
            # Q&A Generation Results
            qa_results = self.results.get('qa_generation', {})
            if qa_results:
                report_lines.append("## Q&A Generation Results")
                report_lines.append(f"- **Total Generated:** {qa_results.get('total_generated', 0)}")
                report_lines.append(f"- **Average Quality Score:** {qa_results.get('average_quality_score', 0):.3f}")
                
                domain_dist = qa_results.get('domain_distribution', {})
                if domain_dist:
                    report_lines.append("- **Domain Distribution:**")
                    for domain, count in domain_dist.items():
                        report_lines.append(f"  - {domain}: {count}")
                
                report_lines.append("")
            
            # Training Results
            training_results = self.results.get('training', {})
            if training_results:
                report_lines.append("## Training Results")
                report_lines.append(f"- **Final Training Loss:** {training_results.get('training_loss', 0):.4f}")
                report_lines.append(f"- **Training Duration:** {training_results.get('training_duration', 'Unknown')}")
                report_lines.append(f"- **Total Steps:** {training_results.get('total_steps', 0)}")
                report_lines.append(f"- **Epochs Completed:** {training_results.get('epochs_completed', 0)}")
                report_lines.append("")
            
            # Model Information
            model_info = self.results.get('model_info', {})
            if model_info:
                model_details = model_info.get('model_info', {})
                report_lines.append("## Model Information")
                report_lines.append(f"- **Trainable Parameters:** {model_details.get('trainable_parameters', 0):,}")
                report_lines.append(f"- **Total Parameters:** {model_details.get('total_parameters', 0):,}")
                
                if model_details.get('trainable_parameters', 0) > 0 and model_details.get('total_parameters', 0) > 0:
                    ratio = model_details['trainable_parameters'] / model_details['total_parameters']
                    report_lines.append(f"- **Trainable Ratio:** {ratio:.2%}")
                
                report_lines.append("")
            
            # Evaluation Results
            evaluation_results = self.results.get('evaluation', {})
            if evaluation_results:
                composite_scores = evaluation_results.get('composite_scores', {})
                
                report_lines.append("## Evaluation Results")
                report_lines.append(f"- **Overall Score:** {composite_scores.get('overall_score', 0):.3f}")
                report_lines.append(f"- **NLP Score:** {composite_scores.get('nlp_score', 0):.3f}")
                report_lines.append(f"- **Bengali Language Score:** {composite_scores.get('bengali_score', 0):.3f}")
                report_lines.append(f"- **Legal Domain Score:** {composite_scores.get('legal_score', 0):.3f}")
                report_lines.append(f"- **Citation Score:** {composite_scores.get('citation_score', 0):.3f}")
                report_lines.append(f"- **Quality Score:** {composite_scores.get('quality_score', 0):.3f}")
                report_lines.append("")
            
            # Configuration Summary
            report_lines.append("## Configuration Summary")
            model_config = self.config.MODEL_CONFIG
            lora_config = self.config.LORA_CONFIG
            training_config = self.config.TRAINING_CONFIG
            
            report_lines.append(f"- **Base Model:** {model_config['base_model']}")
            report_lines.append(f"- **LoRA Rank:** {lora_config['r']}")
            report_lines.append(f"- **LoRA Alpha:** {lora_config['lora_alpha']}")
            report_lines.append(f"- **Learning Rate:** {training_config['learning_rate']}")
            report_lines.append(f"- **Batch Size:** {training_config['per_device_train_batch_size']}")
            report_lines.append(f"- **Epochs:** {training_config['num_train_epochs']}")
            report_lines.append("")
            
            # Next Steps
            report_lines.append("## Next Steps")
            if success:
                report_lines.append("- Model training completed successfully")
                report_lines.append("- Review evaluation metrics and consider further optimization")
                report_lines.append("- Test model with real-world Bengali legal queries")
                report_lines.append("- Consider deploying for production use")
            else:
                report_lines.append("- Review error logs and fix identified issues")
                report_lines.append("- Check configuration and resource availability")
                report_lines.append("- Retry execution after addressing problems")
            
            # Save report
            output_dir = self.config.get_output_dir()
            report_file = output_dir / "phase3_execution_report.md"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Execution report saved to: {report_file}")
            
            # Also log to console
            self.logger.info("\n" + "\n".join(report_lines))
            
            return '\n'.join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating execution report: {e}")
            self.execution_state['errors'].append(f"Report generation failed: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("Cleaning up resources...")
            
            # Cleanup fine-tuning engine
            if 'fine_tuning_engine' in self.phase3_components:
                self.phase3_components['fine_tuning_engine'].cleanup()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def execute_full_pipeline(self):
        """Execute the complete Phase 3 pipeline"""
        try:
            self.logger.info("Starting complete Phase 3 execution pipeline...")
            
            # Step 1: Initialize Phase 2 components
            if not self.initialize_phase2_components():
                raise Exception("Failed to initialize Phase 2 components")
            
            # Step 2: Generate Q&A dataset
            qa_pairs = self.generate_qa_dataset()
            
            # Step 3: Initialize fine-tuning engine
            fine_tuning_engine = self.initialize_fine_tuning_engine()
            
            # Step 4: Execute training
            fine_tuning_engine, test_dataset = self.execute_training(qa_pairs)
            
            # Step 5: Evaluate model
            evaluation_results = self.evaluate_model(fine_tuning_engine, test_dataset)
            
            # Step 6: Save results
            final_results = self.save_final_results()
            
            # Step 7: Generate report
            execution_report = self.generate_execution_report()
            
            # Update execution state
            self.execution_state['phase'] = 'completed'
            self.execution_state['current_step'] = self.execution_state['total_steps']
            
            self.logger.info("=" * 80)
            self.logger.info("Phase 3 execution completed successfully!")
            self.logger.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.logger.error(traceback.format_exc())
            self.execution_state['phase'] = 'failed'
            
            # Still try to generate a report
            try:
                self.generate_execution_report()
            except:
                pass
            
            raise
        
        finally:
            # Always cleanup
            self.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Bengali Legal Advocate AI - Phase 3 Fine-tuning Pipeline'
    )
    
    parser.add_argument(
        '--preset', 
        choices=['quick_test', 'development', 'production', 'high_quality'],
        default='development',
        help='Training configuration preset'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='Custom configuration file (JSON)'
    )
    
    parser.add_argument(
        '--qa-only',
        action='store_true',
        help='Only generate Q&A dataset, skip training'
    )
    
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation on existing model'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Custom output directory'
    )
    
    args = parser.parse_args()
    
    try:
        # Load custom configuration if provided
        custom_config = {}
        if args.config_file:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
        
        # Override output directory if specified
        if args.output_dir:
            custom_config['training_config'] = custom_config.get('training_config', {})
            custom_config['training_config']['output_dir'] = args.output_dir
        
        # Initialize pipeline
        pipeline = Phase3ExecutionPipeline(
            config_preset=args.preset,
            custom_config=custom_config
        )
        
        # Execute based on mode
        if args.qa_only:
            # Only generate Q&A dataset
            pipeline.initialize_phase2_components()
            qa_pairs = pipeline.generate_qa_dataset()
            pipeline.logger.info(f"Q&A generation completed. Generated {len(qa_pairs)} pairs.")
            
        elif args.eval_only:
            # Only run evaluation (assumes model exists)
            pipeline.logger.info("Evaluation-only mode not implemented yet")
            
        else:
            # Full pipeline execution
            results = pipeline.execute_full_pipeline()
            
            # Print summary
            print("\n" + "=" * 80)
            print("PHASE 3 EXECUTION SUMMARY")
            print("=" * 80)
            
            execution_info = results.get('execution_info', {})
            print(f"Status: {'SUCCESS' if execution_info.get('success', False) else 'FAILED'}")
            print(f"Duration: {execution_info.get('duration', 'Unknown')}")
            
            qa_results = results.get('results', {}).get('qa_generation', {})
            if qa_results:
                print(f"Q&A pairs generated: {qa_results.get('total_generated', 0)}")
            
            evaluation_results = results.get('results', {}).get('evaluation', {})
            if evaluation_results:
                composite_scores = evaluation_results.get('composite_scores', {})
                print(f"Overall evaluation score: {composite_scores.get('overall_score', 0):.3f}")
            
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"Execution failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 