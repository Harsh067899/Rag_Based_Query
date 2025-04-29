import os
import requests
from zipfile import ZipFile
import io
import shutil

def download_sample_data():
    """
    Download sample lab report data from Google Drive
    
    The dataset link provided in the problem statement:
    https://drive.google.com/file/d/1LzG7oJ-cqGHK9KbwXnWfkWgnQ3xi8Cr9/view?usp=sharing
    
    This script will:
    1. Download the zip file from Google Drive
    2. Extract it to a 'data' directory
    3. Clean up temporary files
    """
    # Google Drive file ID from the link
    file_id = "1LzG7oJ-cqGHK9KbwXnWfkWgnQ3xi8Cr9"
    
    # Direct download URL for Google Drive
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Alternative download method if the above doesn't work
    # download_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    
    print("Downloading sample lab report data...")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    try:
        # Download the file
        response = requests.get(download_url, stream=True)
        
        # Check if the response is valid
        if response.status_code == 200:
            # Save the downloaded file temporarily
            with open("temp_data.zip", "wb") as f:
                f.write(response.content)
            
            # Extract the zip file
            print("Extracting files...")
            with ZipFile("temp_data.zip", "r") as zip_ref:
                zip_ref.extractall("data")
            
            # Clean up the temporary zip file
            os.remove("temp_data.zip")
            
            print("Sample data downloaded and extracted successfully to 'data' directory!")
        else:
            print(f"Failed to download data: HTTP status code {response.status_code}")
            print("Please download manually from: https://drive.google.com/file/d/1LzG7oJ-cqGHK9KbwXnWfkWgnQ3xi8Cr9/view?usp=sharing")
    
    except Exception as e:
        print(f"Error occurred during download: {str(e)}")
        print("Please download manually from: https://drive.google.com/file/d/1LzG7oJ-cqGHK9KbwXnWfkWgnQ3xi8Cr9/view?usp=sharing")

if __name__ == "__main__":
    download_sample_data() 