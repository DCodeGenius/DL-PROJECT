#!/bin/bash
# Quick setup script for GCP VM
# Run this on your VM after SSH'ing in

set -e  # Exit on error

echo "=========================================="
echo "FEVER Project VM Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install Python and build tools
echo "Installing Python and build tools..."
sudo apt install -y python3 python3-pip python3-venv build-essential git

# Check GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️  NVIDIA driver not found. Installing..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-4
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
echo "Installing project dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txt not found. Installing packages manually..."
    pip install transformers datasets accelerate peft bitsandbytes trl scikit-learn numpy pandas wandb tqdm huggingface-hub
fi

# Verify GPU access
echo "Verifying GPU access..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Login to HuggingFace: huggingface-cli login"
echo "3. Run verification: python verify_setup.py"
echo "=========================================="
