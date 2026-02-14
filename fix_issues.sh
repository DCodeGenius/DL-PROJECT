#!/bin/bash
# Fix script for GPU and dataset issues

echo "=========================================="
echo "Fixing Setup Issues"
echo "=========================================="

# 1. Check GPU
echo "1. Checking GPU..."
nvidia-smi

# 2. Check PyTorch CUDA
echo ""
echo "2. Checking PyTorch CUDA..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# 3. Fix datasets library (downgrade to version that supports trust_remote_code)
echo ""
echo "3. Fixing datasets library..."
pip install "datasets==2.14.0" --force-reinstall

echo ""
echo "=========================================="
echo "✅ Fix complete!"
echo "Now try: python verify_setup.py"
echo "=========================================="
