import yaml
import pkg_resources
import sys
from pathlib import Path
from models.animated_diffusion import AnimateDiffModel
from training.trainer import AnimateDiffTrainer
from evaluation.metrics import TemporalCoherenceMetrics

def check_requirements():
    """
    Verify that all required packages are installed with correct versions
    """
    requirements_path = Path(__file__).parent.parent / 'requirements.txt'
    
    if not requirements_path.exists():
        print("Error: requirements.txt not found!")
        sys.exit(1)
        
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    missing_packages = []
    version_issues = []
    
    for requirement in requirements:
        try:
            pkg_resources.require(requirement)
        except pkg_resources.VersionConflict as e:
            version_issues.append(f"{requirement}: {str(e)}")
        except pkg_resources.DistributionNotFound:
            missing_packages.append(requirement)
    
    if missing_packages or version_issues:
        print("\nPackage requirement issues found:")
        if missing_packages:
            print("\nMissing packages:")
            for pkg in missing_packages:
                print(f"  - {pkg}")
        if version_issues:
            print("\nVersion conflicts:")
            for issue in version_issues:
                print(f"  - {issue}")
        print("\nPlease install required packages using:")
        print("pip install -r requirements.txt")
        sys.exit(1)

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_regularization_configs(model, trainer, evaluator, config):
    """
    Test different regularization configurations
    """
    regularization_configs = [
        {'type': 'temporal_consistency', 'weight': 0.1},
        {'type': 'motion_smoothness', 'weight': 0.05},
        {'type': 'combined', 'weights': [0.1, 0.05]}
    ]
    
    results = {}
    for reg_config in regularization_configs:
        print(f"\nTesting configuration: {reg_config['type']}")
        
        # Update model with current regularization settings
        model.update_regularization(reg_config)
        
        # Train with current configuration
        trainer.train()
        
        # Evaluate results
        metrics = evaluator.evaluate_sequence(model.generate_test_sequence())
        results[reg_config['type']] = metrics
        
    return results

def main():
    # Check requirements first
    check_requirements()
    
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Initialize model with temporal regularization support
    model = AnimateDiffModel(config)
    
    # Initialize trainer with enhanced logging for temporal metrics
    trainer = AnimateDiffTrainer(model, config)
    
    # Initialize evaluation metrics
    evaluator = TemporalCoherenceMetrics()
    
    # Get baseline measurements
    print("Evaluating baseline performance...")
    baseline_metrics = evaluator.evaluate_sequence(
        model.generate_test_sequence()
    )
    print("Baseline metrics:", baseline_metrics)
    
    # Test different regularization configurations
    results = test_regularization_configs(model, trainer, evaluator, config)
    
    # Print and save results
    print("\nExperimental Results:")
    print("====================")
    print("Baseline:", baseline_metrics)
    for config_type, metrics in results.items():
        print(f"\n{config_type}:", metrics)
    
    # Save best configuration
    best_config = max(results.items(), key=lambda x: x[1]['temporal_consistency'])
    print(f"\nBest configuration: {best_config[0]}")
    
    # Apply best configuration to model
    model.update_regularization({
        'type': best_config[0],
        'weight': results[best_config[0]]['weight']
    })
    
    # Final training with best configuration
    print("\nTraining with best configuration...")
    trainer.train()

if __name__ == "__main__":
    main() 