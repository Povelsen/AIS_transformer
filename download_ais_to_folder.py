import requests
import zipfile
import os

# User inputs
year = input("Enter the year (e.g., 2025): ")
month = input("Enter the month (e.g., 02): ")
start_day = int(input("Enter the start day (e.g., 12): "))
end_day = int(input("Enter the end day (e.g., 18): "))
target_folder = "/Users/mathiaspovelsen/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/Kandidat/02456 - Deep Learning/AIS_Transformer_Gemini_and_Meta/raw_ais_data"

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Base URL pattern
base_url = f'http://aisdata.ais.dk/{year}/aisdk-{year}-{month}-{{day}}.zip'

# Days to download
days = range(start_day, end_day + 1)

for day in days:
    url = base_url.format(day=f"{day:02d}")
    zip_filename = f'aisdk-{year}-{month}-{day:02d}.zip'
    
    # Download the zip file
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    # Save and extract immediately
    with open(zip_filename, 'wb') as f:
        f.write(response.content)
    
    print(f"Extracting {zip_filename} to {target_folder}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    
    # Remove the temporary zip file
    os.remove(zip_filename)
    
    print(f"Completed for day {day}")

print(f"All files have been downloaded and extracted to '{target_folder}'.")