# Two-Stage BERT Fine-tuning for CommonsenseQA

This project implements a two-stage fine-tuning approach to improve BERT's performance on the CommonsenseQA dataset by first fine-tuning on related commonsense reasoning datasets.

## Overview

The two-stage training strategy:

1. **Stage 1: Intermediate Fine-tuning** - Fine-tune BERT on a related dataset (SWAG or HellaSwag) to learn general commonsense reasoning patterns
2. **Stage 2: Task-Specific Fine-tuning** - Fine-tune the model from Stage 1 on CommonsenseQA for the target task

This approach leverages transfer learning from related tasks to improve performance on CommonsenseQA.

## Project Structure

```
.
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── model.py            # BERT model with multiple-choice head
│   ├── train.py            # Two-stage training script
│   ├── evaluate.py         # Evaluation script
│   └── config.py           # Configuration classes
├── requirements.txt        # Python dependencies
├── run_training.sh         # Quick start training script
└── README.md              # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bert
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Using the provided script

```bash
bash run_training.sh
```

### Option 2: Manual training

```bash
cd src
python train.py \
    --stage1_dataset swag \
    --stage1_epochs 3 \
    --stage1_batch_size 8 \
    --stage1_lr 2e-5 \
    --stage2_epochs 5 \
    --stage2_batch_size 8 \
    --stage2_lr 1e-5
```

## Training Options

### Stage 1 Datasets

- **SWAG** (Situations With Adversarial Generations): 113k multiple-choice questions about grounded situations
- **HellaSwag**: More challenging version of SWAG with 70k questions

### Command Line Arguments

**Model Configuration:**
- `--model_name`: Base model (default: `bert-base-uncased`)
- `--max_length`: Maximum sequence length (default: 256)

**Stage 1 (Intermediate Fine-tuning):**
- `--stage1_dataset`: Dataset for stage 1 (`swag` or `hellaswag`)
- `--stage1_epochs`: Number of epochs (default: 3)
- `--stage1_batch_size`: Batch size (default: 8)
- `--stage1_lr`: Learning rate (default: 2e-5)
- `--stage1_output_dir`: Output directory (default: `checkpoints/stage1`)
- `--skip_stage1`: Skip stage 1 training
- `--stage1_checkpoint`: Path to existing stage 1 checkpoint

**Stage 2 (CommonsenseQA Fine-tuning):**
- `--stage2_epochs`: Number of epochs (default: 5)
- `--stage2_batch_size`: Batch size (default: 8)
- `--stage2_lr`: Learning rate (default: 1e-5)
- `--stage2_output_dir`: Output directory (default: `checkpoints/stage2`)

**Other:**
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (`cuda` or `cpu`)

## Evaluation

Evaluate a trained model:

```bash
cd src
python evaluate.py \
    --model_path checkpoints/stage2 \
    --dataset commonsenseqa \
    --split validation
```

## Advanced Usage

### Skip Stage 1 (use existing checkpoint)

If you already have a model fine-tuned on an intermediate dataset:

```bash
cd src
python train.py \
    --skip_stage1 \
    --stage1_checkpoint path/to/checkpoint \
    --stage2_epochs 5
```

### Train only on CommonsenseQA (baseline)

To train directly on CommonsenseQA without intermediate fine-tuning:

```bash
cd src
python train.py \
    --skip_stage1 \
    --stage2_epochs 5
```

### Use HellaSwag instead of SWAG

```bash
cd src
python train.py \
    --stage1_dataset hellaswag \
    --stage1_epochs 3 \
    --stage2_epochs 5
```

## Expected Performance

The two-stage fine-tuning approach typically shows improvement over direct fine-tuning:

- **Baseline** (direct fine-tuning on CommonsenseQA): ~60-65% accuracy
- **Two-stage** (SWAG → CommonsenseQA): ~65-70% accuracy
- **Two-stage** (HellaSwag → CommonsenseQA): ~65-70% accuracy

*Note: Actual performance depends on hyperparameters and training conditions*

## Model Architecture

The model uses:
- **Base Model**: BERT-base-uncased (110M parameters)
- **Classification Head**: Linear layer on [CLS] token
- **Input Format**: Question-answer pairs processed independently, scores aggregated for prediction

## Datasets

### CommonsenseQA
- **Train**: 9,741 questions
- **Validation**: 1,221 questions
- **Format**: 5-choice questions requiring commonsense reasoning

### SWAG
- **Train**: 73,546 questions
- **Validation**: 20,006 questions
- **Format**: 4-choice sentence completion

### HellaSwag
- **Train**: 39,905 questions
- **Validation**: 10,042 questions
- **Format**: 4-choice sentence completion (harder than SWAG)

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py --stage1_batch_size 4 --stage2_batch_size 4
```

Or use gradient accumulation (modify in `config.py`).

### Slow Training

- Use smaller `max_length` (e.g., 128)
- Reduce number of epochs
- Use fewer training examples for testing

## Citation

If you use this code, please cite:

```bibtex
@article{commonsenseqa,
  title={CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge},
  author={Talmor, Alon and Herzig, Jonathan and Lourie, Nicholas and Berant, Jonathan},
  journal={NAACL-HLT},
  year={2019}
}

@article{swag,
  title={SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
  author={Zellers, Rowan and Bisk, Yonatan and Schwartz, Roy and Choi, Yejin},
  journal={EMNLP},
  year={2018}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
