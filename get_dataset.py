import argparse
import os
import gdown
import zipfile

def download_floodvn():
    if os.path.exists('floodvn'):
        print("✓ FloodVN already exists. Skipping.")
        return
    file_id = '1tQYUVtSdYJ3cGn1oftmb9MeWrmu4ez7P'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'floodvn.zip'
    print("Downloading FloodVN...")
    gdown.download(url, output, quiet=False)
    print("Extracting FloodVN...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('floodvn')
    os.remove(output)
    print("✓ FloodVN ready.\n")

def download_floodkaggle():
    if os.path.exists('floodkaggle'):
        print("✓ FloodKaggle already exists. Skipping.")
        return
    file_id = '1hue0azsiVoCrFRQ7FIn6xpKViw5ZEKlO'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'floodkaggle.zip'
    print("Downloading FloodKaggle...")
    gdown.download(url, output, quiet=False)
    print("Extracting FloodKaggle...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('floodkaggle')
    os.remove(output)
    print("✓ FloodKaggle ready.\n")

def download_floodnet():
    if os.path.exists('floodnet'):
        print("✓ FloodNet already exists. Skipping.")
        return
    file_id = '1IbbI5iomI7elrvERrGlgGB0V32KgOdIj'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'floodnet.zip'
    print("Downloading FloodNet...")
    gdown.download(url, output, quiet=False)
    print("Extracting FloodNet...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(output)
    print("✓ FloodNet ready.\n")

def main():
    parser = argparse.ArgumentParser(description='Download Flood Detection Datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['floodvn', 'floodkaggle', 'floodnet', 'all'],
                        help='Dataset to download: floodvn, floodkaggle, floodnet, or all')
    args = parser.parse_args()
    
    print("="*70)
    print("DATASET DOWNLOADER")
    print("="*70)
    print(f"Selected: {args.dataset.upper()}")
    print("="*70 + "\n")
    
    if args.dataset == 'floodvn':
        download_floodvn()
    elif args.dataset == 'floodkaggle':
        download_floodkaggle()
    elif args.dataset == 'floodnet':
        download_floodnet()
    elif args.dataset == 'all':
        print("Downloading all datasets...\n")
        download_floodvn()
        download_floodkaggle()
        download_floodnet()
    
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()