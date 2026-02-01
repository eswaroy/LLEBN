
import os
import subprocess
import sys

def run_step(command, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD: {command}")
    print(f"{'='*60}")
    
    try:
        subprocess.check_call(command, shell=True)
        print("✓ Success")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with error code {e.returncode}")
        print("Stopping execution.")
        sys.exit(1)

def main():
    # Ensure CWD is set to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working Directory: {os.getcwd()}")
    
    print("HUMAN VERIFICATION PROJECT - EXECUTION SCRIPT")
    print("---------------------------------------------")
    
    # 1. Download Data
    if not os.path.exists("data/balabit_raw") and not os.path.exists("data/Mouse-Dynamics-Challenge-master"):
        print("Data not found. Attempting download...")
        run_step("python data/download_dataset.py", "Download Balabit Dataset")
    else:
        print("✓ Data appears to be present.")

    # 2. Run Integration Tests
    run_step("python tests/test_integration.py", "Run Integration Tests")
    
    # 3. Train Model
    run_step("python training/train_siamese.py", "Train Siamese Network (Optimized)")
    
    # 4. Run Verification
    run_step("python test_verification.py", "Run Verification System")
    
    print("\n" + "="*60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
