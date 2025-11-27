"""
Example inference script for trained BERT model.
"""

import torch
from transformers import AutoTokenizer
from model import BertForMultipleChoice


def predict(model, tokenizer, question, choices, device='cuda'):
    """
    Make a prediction for a single question.

    Args:
        model: Trained BertForMultipleChoice model
        tokenizer: BERT tokenizer
        question: Question text
        choices: List of answer choices
        device: Device to use

    Returns:
        Predicted choice index and scores
    """
    model.eval()
    model.to(device)

    # Tokenize
    encoded = tokenizer(
        [question] * len(choices),
        choices,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].unsqueeze(0).to(device)  # [1, num_choices, seq_len]
    attention_mask = encoded['attention_mask'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)

    # Get probabilities
    probs = torch.softmax(logits, dim=1)[0]
    predicted_idx = torch.argmax(logits, dim=1).item()

    return predicted_idx, probs.cpu().tolist()


def main():
    """Example usage."""

    # Load model and tokenizer
    model_path = '../checkpoints/stage2'
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMultipleChoice.from_pretrained(model_path)

    # Example question
    question = "Where would you find a fox that is made up?"
    choices = [
        "natural habitat",
        "storybook",
        "computer game",
        "hen house",
        "movie hall"
    ]

    print(f"\nQuestion: {question}")
    print(f"\nChoices:")
    for i, choice in enumerate(choices):
        print(f"  {chr(65+i)}. {choice}")

    # Make prediction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predicted_idx, probs = predict(model, tokenizer, question, choices, device)

    print(f"\nPrediction: {chr(65+predicted_idx)} - {choices[predicted_idx]}")
    print(f"\nConfidence scores:")
    for i, (choice, prob) in enumerate(zip(choices, probs)):
        print(f"  {chr(65+i)}. {choice}: {prob:.4f}")


if __name__ == '__main__':
    main()
