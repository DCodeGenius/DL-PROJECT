# Complete Setup Guide: Google Cloud + Environment + HuggingFace

This guide walks you through setting up everything from scratch.

---

## Part 1: Google Cloud Platform (GCP) Setup

### Step 1: Create GCP Account & Project

1. Go to https://console.cloud.google.com
2. Sign in with your Google account
3. Create a new project (or use existing):
   - Click project dropdown → "New Project"
   - Name: `fever-finetuning` (or whatever you like)
   - Click "Create"

### Step 2: Enable Billing & GPU Quota

1. **Enable Billing:**
   - Go to "Billing" in left menu
   - Link your billing account (you have $50 credit each)
   - Make sure billing is enabled for your project

2. **Request GPU Quota (if needed):**
   - Go to "IAM & Admin" → "Quotas"
   - Filter: "NVIDIA L4" or "NVIDIA A100"
   - If quota is 0, click "Edit Quotas" and request 1-2 GPUs
   - Usually approved within minutes for students

### Step 3: Create VM with GPU

**Option A: Using GCP Console (Easier for first time)**

1. Go to "Compute Engine" → "VM instances"
2. Click "Create Instance"

3. **Basic Settings:**
   - Name: `fever-gpu-vm`
   - Region: Choose closest to you (e.g., `us-central1`)
   - Zone: Any zone with L4 GPUs available (e.g., `us-central1-a`)

4. **Machine Configuration:**
   - Machine family: **GPU**
   - Machine type: `n1-standard-4` (4 vCPUs, 15GB RAM)
   - GPU type: **NVIDIA L4** (or A100 if you want faster)
   - Number of GPUs: **1**
   - GPU availability: **Spot** (cheaper! ~$0.80/hr vs ~$1.20/hr)

5. **Boot Disk:**
   - OS: **Ubuntu 22.04 LTS**
   - Disk size: **50 GB** (enough for model + data)
   - Disk type: **Standard persistent disk** (cheaper than SSD for this)

6. **Firewall:**
   - Allow HTTP traffic: ✅
   - Allow HTTPS traffic: ✅

7. **Advanced Options → Networking:**
   - Network tags: Add `gpu-vm` (for easy SSH access)

8. Click **"Create"**

**⚠️ Important:** Spot instances can be terminated. Save your work frequently!

**Option B: Using gcloud CLI (Faster, if you have it installed)**

```bash
gcloud compute instances create fever-gpu-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --boot-disk-size=50GB \
    --metadata=install-nvidia-driver=True
```

### Step 4: Install CUDA & Drivers (on the VM)

1. **SSH into your VM:**
   - In GCP Console, click "SSH" next to your VM
   - Or use: `gcloud compute ssh fever-gpu-vm --zone=us-central1-a`

2. **Update system:**
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

3. **Install CUDA (if not auto-installed):**
   ```bash
   # Check if NVIDIA driver is installed
   nvidia-smi
   
   # If command not found, install:
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-4
   ```

4. **Verify GPU:**
   ```bash
   nvidia-smi
   # Should show your L4 GPU
   ```

---

## Part 2: Python Environment Setup

### Step 1: Install Python & pip

```bash
# Check Python version (should be 3.10+)
python3 --version

# Install pip if needed
sudo apt install python3-pip -y

# Install build tools (needed for some packages)
sudo apt install build-essential -y
```

### Step 2: Create Virtual Environment

```bash
# Install venv
sudo apt install python3-venv -y

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your prompt
```

### Step 3: Install PyTorch with CUDA

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Install Project Dependencies

```bash
# Upload your requirements.txt to the VM, or create it there
# Then install:
pip install -r requirements.txt

# This will take 5-10 minutes
```

**Alternative: Install packages one by one if requirements.txt fails:**
```bash
pip install transformers datasets accelerate peft bitsandbytes trl scikit-learn numpy pandas wandb tqdm
```

### Step 5: Verify Installation

```bash
# Test PyTorch can see GPU
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Should output:
# True
# NVIDIA L4 (or your GPU name)
```

---

## Part 3: HuggingFace Setup

### Step 1: Create HuggingFace Account

1. Go to https://huggingface.co/join
2. Sign up (free)
3. Verify your email

### Step 2: Request Llama-3.1 Access

1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
2. Click **"Agree and access repository"**
3. Fill out the form (select "Research" or "Education")
4. Wait for approval (usually hours, sometimes instant)

### Step 3: Generate Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `fever-project`
4. Type: **Read** (sufficient for downloading models)
5. Click **"Generate token"**
6. **Copy the token immediately** (you won't see it again!)

### Step 4: Login on VM

**On your GCP VM:**

```bash
# Install huggingface-hub CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Paste your token when prompted
# Should see: "Login successful"
```

**Verify login:**
```bash
huggingface-cli whoami
# Should show your username
```

---

## Part 4: Upload Project to VM

### Option A: Using gcloud (if on Windows/Mac with gcloud installed)

```bash
# From your local machine (in project directory)
gcloud compute scp --recurse . fever-gpu-vm:~/fever-project --zone=us-central1-a
```

### Option B: Using Git (Recommended)

**On your VM:**
```bash
# Install git
sudo apt install git -y

# Clone your repo (if you pushed to GitHub)
git clone <your-repo-url>
cd DL-PROJECT

# Or create the files manually
```

### Option C: Manual File Transfer

1. Use GCP Console's built-in file browser
2. Or use `scp` from local machine:
   ```bash
   scp -r . username@VM_IP:~/fever-project
   ```

---

## Part 5: Final Verification

**On your VM, in the project directory:**

```bash
# Activate venv
source venv/bin/activate

# Run verification script
python verify_setup.py
```

**Expected output:**
```
============================================================
Setup Verification
============================================================
Checking GPU...
✅ GPU available: NVIDIA L4
   VRAM: 24.0 GB

Checking HuggingFace authentication...
✅ Logged in as: your_username

Checking model access: meta-llama/Meta-Llama-3.1-8B-Instruct...
✅ Model accessible!

Checking FEVER data loading...
✅ Data loaded successfully!
   Train: 20000 examples
   Eval: 2000 examples

============================================================
✅ All checks passed! You're ready to train.
```

---

## Part 6: Cost Management

### Monitor Spending

1. Go to GCP Console → "Billing"
2. Set up budget alerts:
   - "Budgets & alerts" → "Create budget"
   - Set limit: $80 (leaves buffer)
   - Alert at 50%, 90%, 100%

### Stop VM When Not Using

```bash
# Stop VM (saves money, keeps disk)
gcloud compute instances stop fever-gpu-vm --zone=us-central1-a

# Start VM when needed
gcloud compute instances start fever-gpu-vm --zone=us-central1-a
```

**Cost estimate:**
- L4 spot: ~$0.80/hr
- With $100 budget: ~125 hours of GPU time
- More than enough for your project!

---

## Troubleshooting

### "GPU not found"
- Check: `nvidia-smi` works
- Verify: `torch.cuda.is_available()` returns True
- Reinstall: CUDA drivers if needed

### "Model access denied"
- Verify: You accepted Llama-3.1 license on HuggingFace
- Check: `huggingface-cli whoami` shows your username
- Try: `huggingface-cli login` again

### "Out of memory"
- Reduce `BATCH_SIZE` in config.py
- Reduce `MAX_SEQ_LEN` in config.py
- Make sure `USE_4BIT = True` in config.py

### "VM terminated" (spot instance)
- Spot instances can be preempted
- Save checkpoints frequently (code does this automatically)
- Consider regular (non-spot) instance for final training runs

---

## Next Steps

Once verification passes:
1. Edit `config.py`: Set `TRAIN_SAMPLES = 1000` for sanity check
2. Run: `python train.py`
3. Watch it train! 🚀
