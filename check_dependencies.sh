#!/bin/bash

# Pre-flight check before training
# Verifies dependencies and environment setup

echo "Checking dependencies..."

# Check Python
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python not found"
        exit 1
    else
        echo "✓ Python3 found"
        PYTHON=python3
    fi
else
    echo "✓ Python found"
    PYTHON=python
fi

# Check required Python packages
echo ""
echo "Checking Python packages..."

$PYTHON -c "import torch" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ PyTorch installed"
else
    echo "❌ PyTorch not installed"
    echo ""
    echo "Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

$PYTHON -c "import transformers" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Transformers installed"
else
    echo "❌ Transformers not installed"
    echo ""
    echo "Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

$PYTHON -c "import datasets" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Datasets installed"
else
    echo "❌ Datasets not installed"
    echo ""
    echo "Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check CUDA
echo ""
echo "Checking CUDA..."
$PYTHON -c "import torch; print('✓ CUDA available:', torch.cuda.is_available())" 2>/dev/null

# Check disk space
echo ""
echo "Checking disk space..."
df -h . | tail -1 | awk '{print "Available disk space: " $4}'

echo ""
echo "✓ All checks passed!"
echo ""
echo "Ready to train. Run one of:"
echo "  bash run_training_max_accuracy.sh  # Best accuracy (recommended)"
echo "  bash run_training.sh               # Balanced approach"
echo "  bash run_training_bs16.sh          # Fast training"
