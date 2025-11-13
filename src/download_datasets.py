import os
import urllib.request
from pathlib import Path
import zipfile
import tarfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_ffhq():
    """Download FFHQ dataset (sample images)"""
    print("Downloading FFHQ dataset...")
    base_url = "https://drive.google.com/uc?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL"
    output_dir = Path("data/ffhq")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: For full dataset, use official FFHQ repository
    # This is a placeholder - you'll need to manually download from:
    # https://github.com/NVlabs/ffhq-dataset
    print("Please download FFHQ dataset manually from:")
    print("https://github.com/NVlabs/ffhq-dataset")
    print(f"Extract to: {output_dir}")

def download_lsun_bedrooms():
    """Download LSUN Bedrooms dataset"""
    print("Downloading LSUN Bedrooms dataset...")
    output_dir = Path("data/bedrooms")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Please download LSUN Bedrooms from:")
    print("https://www.innovatiana.com/en/datasets/lsun-bedrooms")
    print(f"Extract to: {output_dir}")

def download_lsun_churches():
    """Download LSUN Churches dataset"""
    print("Downloading LSUN Churches dataset...")
    output_dir = Path("data/churches")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Please download LSUN Churches from:")
    print("https://huggingface.co/datasets/tglcourse/lsun_church_train")
    print(f"Extract to: {output_dir}")

def create_sample_dataset():
    """Create sample images for testing"""
    from PIL import Image
    import numpy as np
    
    print("Creating sample dataset for testing...")
    
    for dataset_name in ['bedrooms', 'churches', 'ffhq']:
        output_dir = Path(f"data/{dataset_name}/samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 100 sample images
        for i in range(100):
            # Generate random image
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(output_dir / f"sample_{i:04d}.jpg")
    
    print("Sample dataset created successfully!")

if __name__ == "__main__":
    print("=" * 60)
    print("Image Steganography Dataset Download")
    print("=" * 60)
    
    # Create sample dataset for immediate testing
    create_sample_dataset()
    
    print("\n" + "=" * 60)
    print("For production use, please download full datasets:")
    print("=" * 60)
    download_ffhq()
    print()
    download_lsun_bedrooms()
    print()
    download_lsun_churches()