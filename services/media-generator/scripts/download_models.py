#!/usr/bin/env python3
"""
Download models for local avatar processing.
Downloads Wav2Lip and FOMM models if not already present.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --wav2lip-only
    python scripts/download_models.py --fomm-only
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import shutil


# Model configurations
MODELS = {
    "wav2lip": {
        "dir": "/app/models/wav2lip",
        "files": [
            {
                "name": "wav2lip_gan.pth",
                "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth",
                "size_mb": 150,
                "description": "Wav2Lip GAN model for high-quality lip-sync"
            }
        ]
    },
    "fomm": {
        "dir": "/app/models/fomm",
        "files": [
            {
                "name": "vox-cpk.pth.tar",
                "url": "https://github.com/AliaksandrSiarohin/first-order-model/releases/download/v1.0.0/vox-cpk.pth.tar",
                "size_mb": 700,
                "description": "FOMM checkpoint trained on VoxCeleb"
            }
        ],
        "configs": [
            {
                "name": "vox-256.yaml",
                "url": "https://raw.githubusercontent.com/AliaksandrSiarohin/first-order-model/master/config/vox-256.yaml",
                "description": "FOMM configuration for 256x256 faces"
            }
        ]
    }
}


def download_with_progress(url: str, dest: str, desc: str = ""):
    """Download file with progress indicator."""
    print(f"  Downloading: {desc or url}")
    print(f"  Destination: {dest}")

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    try:
        urlretrieve(url, dest, reporthook=progress_hook)
        print()  # Newline after progress
        return True
    except URLError as e:
        print(f"\n  Error downloading: {e}")
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_wav2lip_models(model_dir: str = None) -> bool:
    """Download Wav2Lip model."""
    config = MODELS["wav2lip"]
    model_dir = Path(model_dir or config["dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Downloading Wav2Lip Models")
    print("="*60)

    success = True
    for file_info in config["files"]:
        dest = model_dir / file_info["name"]
        if dest.exists():
            print(f"\n  [SKIP] {file_info['name']} already exists")
            continue

        print(f"\n  Model: {file_info['name']} (~{file_info['size_mb']} MB)")
        print(f"  {file_info['description']}")

        if not download_with_progress(file_info["url"], str(dest), file_info["name"]):
            success = False

    return success


def download_fomm_models(model_dir: str = None) -> bool:
    """Download FOMM model and config."""
    config = MODELS["fomm"]
    model_dir = Path(model_dir or config["dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Downloading FOMM Models")
    print("="*60)

    success = True

    # Download model files
    for file_info in config["files"]:
        dest = model_dir / file_info["name"]
        if dest.exists():
            print(f"\n  [SKIP] {file_info['name']} already exists")
            continue

        print(f"\n  Model: {file_info['name']} (~{file_info['size_mb']} MB)")
        print(f"  {file_info['description']}")

        if not download_with_progress(file_info["url"], str(dest), file_info["name"]):
            success = False

    # Download config files
    for config_info in config.get("configs", []):
        dest = model_dir / config_info["name"]
        if dest.exists():
            print(f"\n  [SKIP] {config_info['name']} already exists")
            continue

        print(f"\n  Config: {config_info['name']}")
        print(f"  {config_info['description']}")

        if not download_with_progress(config_info["url"], str(dest), config_info["name"]):
            success = False

    return success


def create_driving_videos_placeholder(model_dir: str = None):
    """Create placeholder for driving videos directory."""
    config = MODELS["fomm"]
    model_dir = Path(model_dir or config["dir"])
    driving_dir = model_dir / "driving_videos"
    driving_dir.mkdir(parents=True, exist_ok=True)

    readme_path = driving_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# FOMM Driving Videos

Place pre-recorded driving videos here for gesture animation:

- `driving_talking.mp4` - Natural talking head movements
- `driving_presenting.mp4` - Presenter-style gestures
- `driving_nodding.mp4` - Nodding agreement movements
- `driving_gesturing.mp4` - Hand gesture animations
- `driving_neutral.mp4` - Subtle neutral movements

## Requirements:
- Resolution: 256x256 or higher
- Duration: 5-15 seconds
- Format: MP4 (H.264)
- Content: Single person with clear face

## Sources:
You can create these from stock footage or record your own.
The movements will be transferred to the avatar image.
"""
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"\n  Created driving videos README: {readme_path}")


def verify_models() -> dict:
    """Verify which models are available."""
    status = {}

    print("\n" + "="*60)
    print("Model Status")
    print("="*60)

    for model_name, config in MODELS.items():
        model_dir = Path(config["dir"])
        files_present = []
        files_missing = []

        for file_info in config["files"]:
            path = model_dir / file_info["name"]
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                files_present.append(f"{file_info['name']} ({size_mb:.1f} MB)")
            else:
                files_missing.append(file_info["name"])

        status[model_name] = {
            "available": len(files_missing) == 0,
            "present": files_present,
            "missing": files_missing
        }

        print(f"\n  {model_name.upper()}:")
        print(f"    Directory: {model_dir}")
        print(f"    Status: {'READY' if status[model_name]['available'] else 'INCOMPLETE'}")
        if files_present:
            for f in files_present:
                print(f"    [OK] {f}")
        if files_missing:
            for f in files_missing:
                print(f"    [MISSING] {f}")

    return status


def main():
    parser = argparse.ArgumentParser(description="Download models for local avatar processing")
    parser.add_argument("--wav2lip-only", action="store_true", help="Only download Wav2Lip model")
    parser.add_argument("--fomm-only", action="store_true", help="Only download FOMM model")
    parser.add_argument("--verify", action="store_true", help="Only verify model status")
    parser.add_argument("--model-dir", type=str, help="Custom model directory base path")
    args = parser.parse_args()

    if args.model_dir:
        MODELS["wav2lip"]["dir"] = f"{args.model_dir}/wav2lip"
        MODELS["fomm"]["dir"] = f"{args.model_dir}/fomm"

    if args.verify:
        verify_models()
        return 0

    success = True

    if args.wav2lip_only:
        success = download_wav2lip_models()
    elif args.fomm_only:
        success = download_fomm_models()
        create_driving_videos_placeholder()
    else:
        # Download all models
        if not download_wav2lip_models():
            success = False
        if not download_fomm_models():
            success = False
        create_driving_videos_placeholder()

    # Verify final status
    verify_models()

    print("\n" + "="*60)
    if success:
        print("All downloads completed successfully!")
    else:
        print("Some downloads failed. Check the errors above.")
    print("="*60 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
