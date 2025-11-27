"""
Quick test script to verify installation and dependencies.
"""

import sys


def check_imports():
    """Check if all required packages are installed."""
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas')
    ]

    print("Checking dependencies...")
    print("-" * 50)

    all_installed = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - NOT FOUND")
            all_installed = False

    print("-" * 50)

    if all_installed:
        print("\n✓ All dependencies are installed!")
        return True
    else:
        print("\n✗ Some dependencies are missing. Please install them:")
        print("  pip install -r requirements.txt")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        print("\nCUDA Information:")
        print("-" * 50)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Device Count: {torch.cuda.device_count()}")
            print(f"Current Device: {torch.cuda.current_device()}")
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("Training will use CPU (slower)")
        print("-" * 50)
    except Exception as e:
        print(f"Error checking CUDA: {e}")


def test_model_import():
    """Test if project modules can be imported."""
    print("\nTesting project imports...")
    print("-" * 50)

    try:
        sys.path.insert(0, 'src')
        from model import BertForMultipleChoice
        from config import TrainingConfig, EvalConfig
        from data_loader import get_dataloader
        print("✓ All project modules imported successfully")
        print("-" * 50)
        return True
    except Exception as e:
        print(f"✗ Error importing project modules: {e}")
        print("-" * 50)
        return False


def main():
    """Run all tests."""
    print("="*50)
    print("Setup Verification")
    print("="*50)
    print()

    # Check imports
    deps_ok = check_imports()

    # Check CUDA
    if deps_ok:
        check_cuda()

    # Test module imports
    if deps_ok:
        modules_ok = test_model_import()
    else:
        modules_ok = False

    # Final summary
    print("\n" + "="*50)
    if deps_ok and modules_ok:
        print("✓ Setup verification PASSED")
        print("="*50)
        print("\nYou can now run training:")
        print("  bash run_training.sh")
        print("\nOr run a quick comparison:")
        print("  bash compare_approaches.sh")
        return 0
    else:
        print("✗ Setup verification FAILED")
        print("="*50)
        print("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
