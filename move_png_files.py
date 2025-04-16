#!/usr/bin/env python3

import os
import sys
import glob
import shutil
from pathlib import Path


def move_png_files(source_dir, files_per_subdir):
    """
    Lists PNG files from a directory and moves them into numbered subdirectories.
    
    Args:
        source_dir (str): Path to the source directory containing PNG files
        files_per_subdir (int): Number of files per subdirectory
    """
    # Convert to Path object for easier path manipulation
    src_path = Path(source_dir).resolve()
    
    # Check if the source directory exists
    if not src_path.is_dir():
        print(f"Error: Directory '{source_dir}' does not exist")
        sys.exit(1)
    
    # Change to the source directory
    os.chdir(src_path)
    
    # Find all PNG files in the directory (case insensitive)
    png_files = []
    for ext in ['*.png', '*.PNG']:
        png_files.extend(glob.glob(ext))
    
    # Check if PNG files were found
    if not png_files:
        print(f"No PNG files found in '{source_dir}'")
        sys.exit(0)
    
    # Sort files by their numeric value (assuming format %08d.png)
    def extract_number(filename):
        try:
            # Extract the numeric part from the filename
            return int(os.path.splitext(filename)[0])
        except ValueError:
            # If the filename doesn't match the expected format, return a large number
            # so non-conforming files appear at the end
            return float('inf')
    
    png_files.sort(key=extract_number)
    
    # Print the number of PNG files found
    print(f"Found {len(png_files)} PNG files in '{source_dir}'")
    
    # Calculate the number of subdirectories needed
    num_subdirs = (len(png_files) + files_per_subdir - 1) // files_per_subdir
    print(f"Will create {num_subdirs} subdirectories")
    
    # Create subdirectories and move files
    for i in range(len(png_files)):
        # Determine which subdirectory this file goes into
        subdir_index = i // files_per_subdir + 1
        subdir_name = f"{subdir_index:03d}"
        subdir_path = src_path / subdir_name
        
        # Create subdirectory if it doesn't exist
        if not subdir_path.exists():
            subdir_path.mkdir()
            print(f"Created subdirectory: {subdir_name}")
        
        # Get the current file
        file = png_files[i]
        
        # Move the file to the subdirectory
        print(f"Moving '{file}' to '{subdir_name}/'")
        shutil.move(file, subdir_path / file)
    
    print("Operation completed successfully")


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <directory_path> <files_per_subdirectory>")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    
    # Check if files_per_subdir is a positive integer
    try:
        files_per_subdir = int(sys.argv[2])
        if files_per_subdir <= 0:
            raise ValueError("Files per subdirectory must be a positive integer")
    except ValueError:
        print("Error: Files per subdirectory must be a positive integer")
        sys.exit(1)
    
    # Run the main function
    move_png_files(source_dir, files_per_subdir)