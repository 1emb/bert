"""
Evaluation script for trained models.
"""

import torch
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os

from model import BertForMultipleChoice
from data_loader import get_dataloader
from config import EvalConfig


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on dataset.

    Returns:
        Dictionary with metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, loss = model(input_ids, attention_mask, labels)

            if loss is not None:
                total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            predictions_list.extend(predictions.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total,
        'predictions': predictions_list,
        'labels': labels_list
    }


def main(config: EvalConfig):
    """Main evaluation function."""

    print(f"Evaluating model from: {config.model_path}")
    print(f"Dataset: {config.dataset}, Split: {config.split}")

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load model
    print("Loading model...")
    if os.path.exists(config.model_path):
        model = BertForMultipleChoice.from_pretrained(config.model_path)
    else:
        raise ValueError(f"Model path does not exist: {config.model_path}")

    model.to(device)

    # Load data
    print(f"Loading {config.dataset} dataset ({config.split} split)...")
    dataloader = get_dataloader(
        config.dataset,
        tokenizer,
        config.split,
        batch_size=config.batch_size,
        max_length=config.max_length,
        shuffle=False
    )

    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_model(model, dataloader, device)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Average Loss: {results['loss']:.4f}")
    print("="*50)

    # Save results
    results_path = os.path.join(config.model_path, f'eval_results_{config.split}.json')
    with open(results_path, 'w') as f:
        # Don't save predictions and labels to keep file small
        save_results = {
            'accuracy': results['accuracy'],
            'loss': results['loss'],
            'correct': results['correct'],
            'total': results['total'],
            'dataset': config.dataset,
            'split': config.split
        }
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate BERT model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='commonsenseqa',
                        choices=['commonsenseqa', 'swag', 'hellaswag'])
    parser.add_argument('--split', type=str, default='validation',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    config = EvalConfig(
        model_path=args.model_path,
        dataset=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )

    main(config)
