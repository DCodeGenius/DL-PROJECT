"""
Training script that automatically stops the VM after completion.
This saves money by stopping the VM when not in use.
"""

import subprocess
import sys
import os
from train import main as train_main

def stop_vm():
    """Stop the GCP VM to save money."""
    print("\n" + "=" * 60)
    print("Stopping VM to save money...")
    print("=" * 60)
    
    # Get VM name and zone from metadata (GCP provides this)
    try:
        # Try to get instance name and zone from metadata
        import urllib.request
        instance_name = urllib.request.urlopen(
            'http://metadata.google.internal/computeMetadata/v1/instance/name',
            headers={'Metadata-Flavor': 'Google'}
        ).read().decode()
        
        zone = urllib.request.urlopen(
            'http://metadata.google.internal/computeMetadata/v1/instance/zone',
            headers={'Metadata-Flavor': 'Google'}
        ).read().decode().split('/')[-1]
        
        print(f"VM: {instance_name}, Zone: {zone}")
        print("Stopping VM via gcloud...")
        
        # Stop the VM
        result = subprocess.run(
            ['gcloud', 'compute', 'instances', 'stop', instance_name, '--zone', zone],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ VM stop command sent successfully!")
        else:
            print(f"⚠️  Could not stop via gcloud: {result.stderr}")
            print("You may need to stop it manually from GCP Console")
            
    except Exception as e:
        print(f"⚠️  Could not auto-stop VM: {e}")
        print("\n⚠️  IMPORTANT: Please stop the VM manually from GCP Console!")
        print("   Go to: Compute Engine → VM instances → Click 'Stop'")
        print(f"   This will save you ~$0.80/hour while the VM is stopped")

def main():
    """Run training and stop VM when done."""
    print("=" * 60)
    print("Starting Training (Test Run)")
    print("=" * 60)
    print("\n⚠️  Note: VM will automatically stop after training completes")
    print("   Results will be saved to: ./results/")
    print("   You can start the VM again anytime to continue work")
    print("=" * 60 + "\n")
    
    try:
        # Run training
        train_main()
        
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)
        print(f"\nResults saved to: {os.path.abspath('results')}")
        print("\nStopping VM in 10 seconds...")
        print("(Press Ctrl+C to cancel and keep VM running)")
        
        import time
        time.sleep(10)
        
        # Stop VM
        stop_vm()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("VM will NOT be stopped automatically")
        print("Please stop it manually if you're done")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        print("VM will NOT be stopped automatically")
        print("Please check the error and stop VM manually when done")
        raise

if __name__ == "__main__":
    main()
