"""
Data Download Script

Downloads:
1. MAESTRO dataset (for VQ-VAE pretraining)
2. PiJAMA dataset (for Brad Mehldau data)
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_maestro(data_dir: str = "./data/maestro"):
    """
    Download MAESTRO v3.0.0 dataset

    Total size: ~120GB (full) or ~10GB (MIDI only)
    We'll download MIDI only for efficiency
    """
    print("=" * 50)
    print("Downloading MAESTRO dataset (MIDI only)...")
    print("=" * 50)

    os.makedirs(data_dir, exist_ok=True)

    maestro_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
    zip_path = os.path.join(data_dir, "maestro-v3.0.0-midi.zip")

    # Download
    if not os.path.exists(zip_path):
        print(f"\nDownloading from {maestro_url}...")
        download_url(maestro_url, zip_path)
        print("✅ Download complete!")
    else:
        print(f"✅ {zip_path} already exists, skipping download")

    # Extract
    extract_dir = os.path.join(data_dir, "maestro-v3.0.0")
    if not os.path.exists(extract_dir):
        print(f"\nExtracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✅ Extraction complete!")
    else:
        print(f"✅ {extract_dir} already exists, skipping extraction")

    # Count MIDI files
    midi_files = list(Path(extract_dir).rglob("*.mid*"))
    print(f"\n✅ MAESTRO dataset ready: {len(midi_files)} MIDI files")

    return extract_dir


def download_pijama(data_dir: str = "./data/pijama"):
    """
    Download PiJAMA dataset from Zenodo

    Contains Brad Mehldau performances
    """
    print("=" * 50)
    print("Downloading PiJAMA dataset...")
    print("=" * 50)

    os.makedirs(data_dir, exist_ok=True)

    # PiJAMA Zenodo record
    zenodo_record = "10.5281/zenodo.7631170"

    print(f"\nPiJAMA dataset needs to be downloaded manually from:")
    print(f"https://zenodo.org/record/7631170")
    print(f"\nAlternatively, install zenodo_get and run:")
    print(f"  pip install zenodo-get")
    print(f"  zenodo_get {zenodo_record} -o {data_dir}")

    # Try automatic download with zenodo_get
    try:
        import zenodo_get
        print("\n✅ zenodo_get found, attempting automatic download...")

        original_dir = os.getcwd()
        os.chdir(data_dir)

        subprocess.run(["zenodo_get", zenodo_record], check=True)

        os.chdir(original_dir)
        print("✅ PiJAMA download complete!")

        return data_dir

    except ImportError:
        print("\n⚠️  zenodo_get not installed")
        print("Install with: pip install zenodo-get")
        return None

    except Exception as e:
        print(f"\n❌ Error downloading PiJAMA: {e}")
        return None


def filter_brad_mehldau(pijama_dir: str, output_dir: str = "./data/brad_mehldau"):
    """
    Filter Brad Mehldau performances from PiJAMA dataset

    PiJAMA contains metadata about artists
    """
    print("=" * 50)
    print("Filtering Brad Mehldau performances...")
    print("=" * 50)

    os.makedirs(output_dir, exist_ok=True)

    # This is a placeholder - actual implementation depends on PiJAMA structure
    print("\n⚠️  Manual filtering required:")
    print("1. Locate PiJAMA metadata file")
    print("2. Filter for artist='Brad Mehldau'")
    print("3. Copy corresponding MIDI files to:", output_dir)

    return output_dir


def download_test_data(data_dir: str = "./data/test"):
    """
    Download small test dataset for development

    Uses a subset of MAESTRO
    """
    print("=" * 50)
    print("Setting up test dataset...")
    print("=" * 50)

    maestro_dir = download_maestro()

    os.makedirs(data_dir, exist_ok=True)

    # Copy first 10 MIDI files for testing
    midi_files = list(Path(maestro_dir).rglob("*.mid*"))[:10]

    import shutil
    for midi_file in midi_files:
        shutil.copy(midi_file, data_dir)

    print(f"\n✅ Test dataset ready: {len(midi_files)} files in {data_dir}")

    return data_dir


def main():
    """Main download script"""
    import argparse

    parser = argparse.ArgumentParser(description="Download training data")
    parser.add_argument(
        "--dataset",
        choices=["maestro", "pijama", "test", "all"],
        default="test",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Base data directory"
    )

    args = parser.parse_args()

    if args.dataset == "maestro" or args.dataset == "all":
        download_maestro(os.path.join(args.data_dir, "maestro"))

    if args.dataset == "pijama" or args.dataset == "all":
        pijama_dir = download_pijama(os.path.join(args.data_dir, "pijama"))
        if pijama_dir:
            filter_brad_mehldau(pijama_dir, os.path.join(args.data_dir, "brad_mehldau"))

    if args.dataset == "test":
        download_test_data(os.path.join(args.data_dir, "test"))

    print("\n" + "=" * 50)
    print("✅ Data download complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
