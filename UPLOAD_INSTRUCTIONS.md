# How to Upload Files to GCP VM (Without gcloud)

## Option 1: Using Git (Easiest)

### Step 1: Push to GitHub (if not already)

On your local machine (in project directory):
```bash
# Initialize git if not already
git init
git add .
git commit -m "Initial commit"

# Create repo on GitHub, then:
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Clone on VM

On the VM (after SSH'ing in):
```bash
sudo apt install git -y
git clone <your-github-repo-url>
cd DL-PROJECT
```

---

## Option 2: Using GCP Console File Browser

1. SSH into VM (click "SSH" button in GCP console)
2. In the SSH window, click the gear icon (⚙️) → "Upload file"
3. Select your project files one by one
4. Or use the file browser to create files manually

---

## Option 3: Copy-Paste Files (For Small Projects)

1. SSH into VM
2. Create project directory:
   ```bash
   mkdir -p ~/fever-project
   cd ~/fever-project
   ```
3. Create files using `nano` or `vi`:
   ```bash
   nano config.py
   # Paste content, Ctrl+X to save
   ```
4. Repeat for each file

---

## Option 4: Install gcloud SDK (If you want to use it)

### On Windows:

1. Download: https://cloud.google.com/sdk/docs/install
2. Run installer
3. Restart PowerShell
4. Run: `gcloud init` to authenticate
5. Then use: `gcloud compute scp --recurse . fever-gpu-vm:~/fever-project --zone=us-central1-a`

---

## Recommended: Use Git!

It's the fastest and most reliable method.
