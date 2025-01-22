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

def main():
    # Check requirements first
    check_requirements()
    
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Initialize model
    model = AnimateDiffModel(config)
    
    # Initialize trainer
    trainer = AnimateDiffTrainer(model, config)
    
    # Initialize evaluation metrics
    evaluator = TemporalCoherenceMetrics()
    
    # Training and evaluation pipeline
    # Implementation for the main experimental workflow

if __name__ == "__main__":
    main() 