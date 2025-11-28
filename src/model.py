"""
BERT model for multiple choice question answering.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple


class BertForMultipleChoice(nn.Module):
    """
    BERT model with a multiple choice classification head.
    """

    def __init__(self, model_name: str = 'bert-base-uncased', num_choices: int = 5,
                 dropout_prob: float = 0.1):
        """
        Initialize the model.

        Args:
            model_name: Pretrained model name or path
            num_choices: Number of choices per question
            dropout_prob: Dropout probability
        """
        super().__init__()

        self.num_choices = num_choices
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, num_choices, seq_length]
            attention_mask: Attention mask [batch_size, num_choices, seq_length]
            labels: Ground truth labels [batch_size]

        Returns:
            logits and optional loss
        """
        batch_size, num_choices, seq_length = input_ids.shape

        # Reshape for BERT: [batch_size * num_choices, seq_length]
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)

        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size * num_choices, hidden_size]

        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size * num_choices, 1]

        # Reshape to [batch_size, num_choices]
        logits = logits.view(batch_size, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss

    def save_pretrained(self, save_path: str):
        """Save model to path."""
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save BERT
        self.bert.save_pretrained(save_path)

        # Save classifier weights
        torch.save({
            'classifier': self.classifier.state_dict(),
            'dropout_prob': self.dropout.p,
            'num_choices': self.num_choices
        }, os.path.join(save_path, 'classifier.pt'))

    @classmethod
    def from_pretrained(cls, load_path: str, num_choices: int = 5):
        """Load model from path."""
        import os

        # Load classifier config
        classifier_path = os.path.join(load_path, 'classifier.pt')
        checkpoint_num_choices = None
        dropout_prob = 0.1

        if os.path.exists(classifier_path):
            checkpoint = torch.load(classifier_path, map_location='cpu')
            checkpoint_num_choices = checkpoint.get('num_choices', None)
            dropout_prob = checkpoint.get('dropout_prob', 0.1)

        # Initialize model with requested num_choices
        model = cls(model_name=load_path, num_choices=num_choices, dropout_prob=dropout_prob)

        # Only load classifier weights if num_choices matches
        if os.path.exists(classifier_path) and checkpoint_num_choices == num_choices:
            model.classifier.load_state_dict(checkpoint['classifier'])
            print(f"Loaded classifier weights for {num_choices} choices")
        elif checkpoint_num_choices is not None and checkpoint_num_choices != num_choices:
            print(f"Checkpoint has {checkpoint_num_choices} choices but model needs {num_choices} choices")
            print(f"Keeping BERT weights, reinitializing classifier for {num_choices} choices")

        return model
