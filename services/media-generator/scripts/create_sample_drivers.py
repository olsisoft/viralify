#!/usr/bin/env python3
"""
Create sample driving videos for FOMM testing.
These are synthetic videos that provide basic motion patterns.
For best results, replace with real pre-recorded driving videos.

Usage:
    python scripts/create_sample_drivers.py
"""

import os
import subprocess
import math
from pathlib import Path


OUTPUT_DIR = Path("/app/models/fomm/driving_videos")


def create_synthetic_driver(
    name: str,
    duration: float = 5.0,
    motion_type: str = "oscillate",
    fps: int = 25
):
    """
    Create a synthetic driving video using FFmpeg.

    These are placeholder videos that create simple motion patterns.
    Replace with real driving videos for production use.
    """
    output_path = OUTPUT_DIR / f"driving_{name}.mp4"

    if output_path.exists():
        print(f"  [SKIP] {name} already exists")
        return True

    print(f"  Creating {name} driver ({duration}s, {motion_type})...")

    # Different motion patterns using FFmpeg filters
    if motion_type == "oscillate":
        # Gentle horizontal oscillation (good for talking)
        filter_complex = (
            f"color=black:s=256x256:d={duration},"
            f"drawbox=x='128+sin(t*3)*20':y=128:w=10:h=10:c=white:t=fill,"
            f"drawbox=x='100+sin(t*2.5)*15':y=100:w=8:h=8:c=gray:t=fill,"
            f"drawbox=x='156+sin(t*2.5)*15':y=100:w=8:h=8:c=gray:t=fill"
        )
    elif motion_type == "nod":
        # Vertical nodding motion
        filter_complex = (
            f"color=black:s=256x256:d={duration},"
            f"drawbox=x=128:y='128+sin(t*2)*30':w=10:h=10:c=white:t=fill"
        )
    elif motion_type == "present":
        # Wider movements for presenting
        filter_complex = (
            f"color=black:s=256x256:d={duration},"
            f"drawbox=x='128+sin(t*1.5)*40':y='128+cos(t*1.5)*20':w=12:h=12:c=white:t=fill"
        )
    elif motion_type == "gesture":
        # Multiple points moving (simulating gestures)
        filter_complex = (
            f"color=black:s=256x256:d={duration},"
            f"drawbox=x='128+sin(t*2)*30':y=100:w=10:h=10:c=white:t=fill,"
            f"drawbox=x='80+sin(t*3+1)*40':y='180+sin(t*2)*30':w=8:h=8:c=gray:t=fill,"
            f"drawbox=x='176+sin(t*3-1)*40':y='180+sin(t*2+0.5)*30':w=8:h=8:c=gray:t=fill"
        )
    else:  # neutral
        # Very subtle movement
        filter_complex = (
            f"color=black:s=256x256:d={duration},"
            f"drawbox=x='128+sin(t*0.5)*5':y='128+sin(t*0.3)*3':w=10:h=10:c=white:t=fill"
        )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", filter_complex,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-t", str(duration),
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0 and output_path.exists():
            print(f"    Created: {output_path}")
            return True
        else:
            print(f"    Error: {result.stderr[:200] if result.stderr else 'Unknown'}")
            return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    print("="*60)
    print("Creating Sample FOMM Driving Videos")
    print("="*60)
    print()
    print("NOTE: These are synthetic placeholders.")
    print("For best quality, replace with real pre-recorded videos.")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create driving videos for each gesture type
    drivers = [
        ("talking", 8.0, "oscillate"),
        ("nodding", 5.0, "nod"),
        ("presenting", 10.0, "present"),
        ("gesturing", 8.0, "gesture"),
        ("neutral", 5.0, "neutral"),
    ]

    success_count = 0
    for name, duration, motion in drivers:
        if create_synthetic_driver(name, duration, motion):
            success_count += 1

    print()
    print("="*60)
    print(f"Created {success_count}/{len(drivers)} driving videos")
    print(f"Location: {OUTPUT_DIR}")
    print("="*60)

    # Create info file
    info_path = OUTPUT_DIR / "SYNTHETIC_DRIVERS.txt"
    with open(info_path, "w") as f:
        f.write("These are synthetic driving videos for testing.\n")
        f.write("For production quality, replace with real recordings.\n")
        f.write("\nRecommended sources:\n")
        f.write("- Record yourself with 256x256 face crop\n")
        f.write("- Use stock footage of talking heads\n")
        f.write("- VoxCeleb dataset samples\n")

    return 0 if success_count == len(drivers) else 1


if __name__ == "__main__":
    exit(main())
