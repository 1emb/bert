#!/bin/bash

# Quick script to check all your trained models

echo "=========================================="
echo "Checking Trained Models"
echo "=========================================="
echo ""

# Function to evaluate a model if it exists
evaluate_model() {
    local model_path=$1
    local model_name=$2

    if [ -d "$model_path" ]; then
        echo "📊 Found: $model_name"

        # Check if results file exists
        if [ -f "$model_path/eval_results_validation.json" ]; then
            echo "   Results:"
            cat "$model_path/eval_results_validation.json" | grep -E "accuracy|correct|total"
        else
            echo "   No evaluation results found. Running evaluation..."
            cd src
            python evaluate.py \
                --model_path ../$model_path \
                --dataset commonsenseqa \
                --split validation \
                --batch_size 16
            cd ..
        fi
        echo ""
    fi
}

# Check for different model configurations
evaluate_model "checkpoints/stage1" "Stage 1 (SWAG)"
evaluate_model "checkpoints/stage2" "Stage 2 (CommonsenseQA) - Default"
evaluate_model "checkpoints/stage1_bs16" "Stage 1 (Batch Size 16)"
evaluate_model "checkpoints/stage2_bs16" "Stage 2 (Batch Size 16)"
evaluate_model "checkpoints/max_accuracy_stage1" "Stage 1 (Max Accuracy)"
evaluate_model "checkpoints/max_accuracy_final" "Stage 2 (Max Accuracy)"
evaluate_model "checkpoints/baseline" "Baseline (No Stage 1)"

echo "=========================================="
echo "Summary"
echo "=========================================="

# Count how many models we have
model_count=$(find checkpoints -name "pytorch_model.bin" 2>/dev/null | wc -l)
echo "Total trained models: $model_count"

if [ $model_count -eq 0 ]; then
    echo ""
    echo "❌ No trained models found!"
    echo ""
    echo "To train a model:"
    echo "  bash run_training_max_accuracy.sh     # For best accuracy"
    echo "  bash run_training.sh                  # For balanced approach"
    echo "  bash run_training_bs16.sh             # For fast training"
else
    echo ""
    echo "✅ Models found and evaluated above"
fi

echo ""
