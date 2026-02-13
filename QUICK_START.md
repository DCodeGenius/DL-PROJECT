# Quick Start Checklist

Follow these steps in order. Check off each as you complete it.

## Phase 1: HuggingFace Setup (Do this FIRST - can do on your local machine)

- [ ] **Step 1.1:** Go to https://huggingface.co/join and create account
- [ ] **Step 1.2:** Verify your email
- [ ] **Step 1.3:** Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- [ ] **Step 1.4:** Click "Agree and access repository"
- [ ] **Step 1.5:** Fill form (select "Research" or "Education")
- [ ] **Step 1.6:** Wait for approval (usually instant to a few hours)
- [ ] **Step 1.7:** Go to https://huggingface.co/settings/tokens
- [ ] **Step 1.8:** Create new token (name: `fever-project`, type: Read)
- [ ] **Step 1.9:** Copy token and save it somewhere safe

## Phase 2: Google Cloud Setup

- [ ] **Step 2.1:** Go to https://console.cloud.google.com
- [ ] **Step 2.2:** Create new project (name: `fever-finetuning`)
- [ ] **Step 2.3:** Enable billing (link your $50 credit)
- [ ] **Step 2.4:** Go to Compute Engine → VM instances
- [ ] **Step 2.5:** Click "Create Instance"
- [ ] **Step 2.6:** Fill in VM settings (see details below)
- [ ] **Step 2.7:** Click "Create" and wait for VM to start

## Phase 3: VM Environment Setup

- [ ] **Step 3.1:** SSH into VM (click "SSH" button in GCP console)
- [ ] **Step 3.2:** Run setup script: `bash setup_vm.sh`
- [ ] **Step 3.3:** Or follow manual setup steps
- [ ] **Step 3.4:** Verify GPU: `nvidia-smi`

## Phase 4: Upload Project & Final Setup

- [ ] **Step 4.1:** Upload project files to VM (see methods below)
- [ ] **Step 4.2:** Login to HuggingFace: `huggingface-cli login`
- [ ] **Step 4.3:** Paste your token when prompted
- [ ] **Step 4.4:** Run verification: `python verify_setup.py`
- [ ] **Step 4.5:** If all checks pass, you're ready!

---

## Detailed Instructions

### VM Settings (Step 2.6)

When creating VM, use these exact settings:

**Basic:**
- Name: `fever-gpu-vm`
- Region: `us-central1` (or closest to you)
- Zone: `us-central1-a` (or any with L4 available)

**Machine Configuration:**
- Machine family: **GPU**
- Machine type: `n1-standard-4`
- GPU: **NVIDIA L4**, Count: **1**
- GPU availability: **Spot** (saves money!)

**Boot Disk:**
- OS: **Ubuntu 22.04 LTS**
- Size: **50 GB**
- Type: **Standard persistent disk**

**Firewall:**
- ✅ Allow HTTP traffic
- ✅ Allow HTTPS traffic

Click **Create**!

### Upload Project Files (Step 4.1)

**Method 1: Using Git (Recommended)**
```bash
# On VM
git clone <your-repo-url>
cd DL-PROJECT
```

**Method 2: Using gcloud (if installed locally)**
```bash
# On your local Windows machine (in project directory)
gcloud compute scp --recurse . fever-gpu-vm:~/fever-project --zone=us-central1-a
```

**Method 3: Manual (copy-paste files)**
- Use GCP Console's built-in file browser
- Or use VS Code Remote SSH extension

---

## Cost Estimate

- L4 Spot GPU: ~$0.80/hour
- With $100 total budget: ~125 hours
- Your project needs: ~20-30 hours
- **You have plenty of budget!** ✅

---

## Troubleshooting

**"No GPU quota"**
→ Go to IAM & Admin → Quotas → Request L4 quota

**"VM creation failed"**
→ Try different zone (us-east1, us-west1, etc.)

**"nvidia-smi not found"**
→ Run: `sudo apt install nvidia-driver-535 -y` then reboot

**"HuggingFace login fails"**
→ Make sure you accepted Llama-3.1 license first

---

## Next Steps After Setup

Once `verify_setup.py` passes:

1. Edit `config.py`: Set `TRAIN_SAMPLES = 1000` for test run
2. Run: `python train.py`
3. Watch it train! 🚀
