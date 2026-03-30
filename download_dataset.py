"""
Script to download the RSNA Pediatric Bone Age Dataset from Kaggle.
Note: You need a Kaggle account and a kaggle.json API key to use this.
"""

import os
import zipfile

def setup_kaggle():
    print("Checking for Kaggle API...")
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        print("⚠️ kaggle.json not found!")
        print("Please download it from Kaggle -> Settings -> Create New API Token")
        print("And place it in C:\\Users\\YourName\\.kaggle\\kaggle.json")
        return False
    return True

def download_dataset():
    try:
        import kaggle
        print("📥 Downloading RSNA Bone Age Dataset...")
        kaggle.api.dataset_download_files('kmader/rsna-bone-age', path='./dataset', unzip=True)
        print("✅ Download complete and extracted!")
    except ImportError:
        print("❌ Kaggle library not installed. Run: pip install kaggle")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if setup_kaggle():
        download_dataset()
    else:
        print("\nStructure set up at ./dataset/ but real images need manual download or Kaggle API.")
