#!/usr/bin/env python
"""
Diagnostic script to check model checkpoints and identify issues.
"""

import os
import sys
import torch

def check_checkpoint(checkpoint_path):
    """Check a model checkpoint and report its configuration."""
    print(f"\n{'='*60}")
    print(f"Checking: {checkpoint_path}")
    print('='*60)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    # Check for BERT model
    bert_config = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(bert_config):
        print("✓ BERT config found")
    else:
        print("✗ BERT config not found")

    # Check for classifier
    classifier_path = os.path.join(checkpoint_path, 'classifier.pt')
    if os.path.exists(classifier_path):
        print("✓ Classifier checkpoint found")

        # Load and inspect
        checkpoint = torch.load(classifier_path, map_location='cpu')
        num_choices = checkpoint.get('num_choices', 'Unknown')
        dropout_prob = checkpoint.get('dropout_prob', 'Unknown')

        print(f"\nClassifier Configuration:")
        print(f"  Number of choices: {num_choices}")
        print(f"  Dropout probability: {dropout_prob}")

        # Check classifier weights
        if 'classifier' in checkpoint:
            classifier_state = checkpoint['classifier']
            print(f"\nClassifier weights:")
            for key, value in classifier_state.items():
                print(f"  {key}: shape {value.shape}")

        # Validate for different datasets
        print(f"\nCompatibility:")
        if num_choices == 4:
            print("  ✓ Compatible with SWAG (4 choices)")
            print("  ✓ Compatible with HellaSwag (4 choices)")
            print("  ✗ NOT compatible with CommonsenseQA (5 choices)")
            print("    → Will need classifier reinitialization!")
        elif num_choices == 5:
            print("  ✗ NOT compatible with SWAG (4 choices)")
            print("  ✗ NOT compatible with HellaSwag (4 choices)")
            print("  ✓ Compatible with CommonsenseQA (5 choices)")
        else:
            print(f"  Unknown configuration ({num_choices} choices)")
    else:
        print("✗ Classifier checkpoint not found")

    # Check for training history
    history_path = os.path.join(checkpoint_path, 'training_history.json')
    if os.path.exists(history_path):
        print("\n✓ Training history found")
        import json
        with open(history_path) as f:
            history = json.load(f)

        if history:
            last_epoch = history[-1]
            print(f"\nLast epoch results:")
            print(f"  Epoch: {last_epoch.get('epoch', 'N/A')}")
            print(f"  Train accuracy: {last_epoch.get('train_acc', 'N/A'):.4f}")
            print(f"  Validation accuracy: {last_epoch.get('val_acc', 'N/A'):.4f}")

    # Check for evaluation results
    eval_path = os.path.join(checkpoint_path, 'eval_results_validation.json')
    if os.path.exists(eval_path):
        print("\n✓ Evaluation results found")
        import json
        with open(eval_path) as f:
            results = json.load(f)

        print(f"\nEvaluation metrics:")
        print(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}")
        print(f"  Correct: {results.get('correct', 'N/A')}/{results.get('total', 'N/A')}")
        print(f"  Dataset: {results.get('dataset', 'N/A')}")


def main():
    """Main diagnostic function."""
    print("="*60)
    print("BERT Model Checkpoint Diagnostics")
    print("="*60)

    # Check common checkpoint locations
    checkpoint_dirs = [
        'checkpoints/stage1',
        'checkpoints/stage2',
        'checkpoints/stage1_bs16',
        'checkpoints/stage2_bs16',
        'checkpoints/max_accuracy_stage1',
        'checkpoints/max_accuracy_final',
        'checkpoints/baseline',
    ]

    # Also check command line arguments
    if len(sys.argv) > 1:
        checkpoint_dirs = sys.argv[1:]

    found_any = False
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            found_any = True
            check_checkpoint(checkpoint_dir)

    if not found_any:
        print("\n❌ No checkpoints found!")
        print("\nSearching for checkpoints...")

        if os.path.exists('checkpoints'):
            subdirs = [d for d in os.listdir('checkpoints')
                      if os.path.isdir(os.path.join('checkpoints', d))]
            if subdirs:
                print(f"\nFound {len(subdirs)} checkpoint(s):")
                for subdir in subdirs:
                    full_path = os.path.join('checkpoints', subdir)
                    check_checkpoint(full_path)
            else:
                print("No checkpoint directories found in ./checkpoints/")
        else:
            print("No checkpoints directory found.")
            print("\nTo train a model, run:")
            print("  bash run_training_max_accuracy.sh")

    print("\n" + "="*60)
    print("Diagnostic Complete")
    print("="*60)

    print("\nSummary:")
    print("  - Stage 1 should have 4 choices (SWAG/HellaSwag)")
    print("  - Stage 2 should have 5 choices (CommonsenseQA)")
    print("  - If Stage 2 uses Stage 1 checkpoint, classifier will be reinitialized")
    print("\nIf Stage 2 accuracy is stuck at ~54%, the bug has been fixed!")
    print("Run: bash resume_stage2.sh")


if __name__ == '__main__':
    main()
