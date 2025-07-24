#!/usr/bin/env python3
"""
Simple test runner for the Spotify scraper
"""

import subprocess
import sys
import pathlib

def main():
    print("Running Spotify Liked Songs scraper...")
    print("=" * 50)
    
    # Check if cookies file exists
    cookies_file = pathlib.Path("spotify_cookies.json")
    if cookies_file.exists():
        print(f"✓ Found cookies file: {cookies_file}")
    else:
        print("⚠ No cookies file found - script will capture cookies first")
    
    # Run the main scraper
    try:
        result = subprocess.run([sys.executable, "spotify_liked_songs.py"], 
                              capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✓ Script completed successfully!")
            
            # Check output file
            output_file = pathlib.Path("my_liked_songs.json")
            if output_file.exists():
                print(f"✓ Output file created: {output_file} ({output_file.stat().st_size} bytes)")
            else:
                print("⚠ No output file found")
        else:
            print(f"✗ Script failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("✗ Script timed out after 5 minutes")
    except Exception as e:
        print(f"✗ Error running script: {e}")

if __name__ == "__main__":
    main()
