
import os

folders = [r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\HerbaceousVegetation', 
           r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\River', 
           r'E:\Earth Observation\Machine Learning\Rwanda Space Challenge\dataset\EuroSAT\Residential']

# Check the directory contents
for folder in folders:
    print(f"Checking folder: {folder}")
    for subdir, dirs, files in os.walk(folder):
        print(f"Subdirectory: {subdir}")
        print(f"Number of images: {len(files)}")
        for file in files:
            print(f"Found file: {file}")
