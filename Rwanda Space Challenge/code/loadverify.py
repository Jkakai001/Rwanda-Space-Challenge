
import os

folders = [
    r'C:\Users\STUDENT\Downloads\EuroSAT\River',
    r'C:\Users\STUDENT\Downloads\EuroSAT\HerbaceousVegetation',
    r'C:\Users\STUDENT\Downloads\EuroSAT\Residential'
]

# Verify image loading
for folder in folders:
    print(f"Checking folder: {folder}")
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for common image extensions
                print(f"Found image: {filename}")
            else:
                print(f"Not an image: {filename}")
    else:
        print(f"Error: Directory {folder} does not exist.")
