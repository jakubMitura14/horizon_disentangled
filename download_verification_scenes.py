import xnat
import os
import sys
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Connection Setup ---
server_url = "https://imaging-platform.diz-ag.med.ovgu.de"
username = "1823514a-11f5-48e8-a56b-e2d8f0fe546d" 
password = "2oxa2hiv8YXXuOQIkFNCk449CtXOrwkEiTmVUSFZdey1FMF68EeJ8zuT5COLrArB"
project_id = "Prostata_bimodal_PIPELINE"

# Define the local base directory for downloads
base_output_directory = "./data"

# Target resource label and filename
TARGET_RESOURCE_LABEL = "slicer_2.2"
TARGET_FILENAME = "verification_scene.mrb"

def download_file(file_obj, dest_path):
    """Download a file from XNAT, skipping if already exists."""
    if os.path.exists(dest_path):
        print(f"  > Skipping (already exists): {dest_path}")
        return False  # Indicate skip, not a new download
        
    print(f"  > Downloading to: {dest_path}")
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        file_obj.download(dest_path)
        print("  > Download complete.")
        return True
    except Exception as e:
        print(f"  > ERROR downloading: {e}")
        return False

try:
    print(f"Connecting to {server_url}...")
    with xnat.connect(server_url, user=username, password=password, verify=False) as session:
        print(f"Successfully connected.")
        
        if project_id in session.projects:
            project = session.projects[project_id]
            print(f"Found project: {project.name}")
            print(f"Searching for '{TARGET_FILENAME}' in '{TARGET_RESOURCE_LABEL}' subject-level resources...\n")

            count_downloaded = 0
            count_skipped = 0
            count_not_found = 0

            for subject in project.subjects.values():
                subject_label = subject.label
                # Use a shorter name for the local folder if the label is very long
                # e.g., extract just the PatXX part
                short_name = subject_label.split("__")[-1] if "__" in subject_label else subject_label
                
                print(f"Checking Subject: {short_name}")
                
                file_found = False
                
                # Search in subject-level resources
                for resource in subject.resources.values():
                    res_label = resource.label if resource.label else ""
                    
                    if res_label.lower() == TARGET_RESOURCE_LABEL.lower():
                        # Found the correct resource, now look for the file
                        for file in resource.files.values():
                            if file.name == TARGET_FILENAME:
                                dest_path = os.path.join(base_output_directory, short_name, TARGET_FILENAME)
                                if download_file(file, dest_path):
                                    count_downloaded += 1
                                else:
                                    count_skipped += 1
                                file_found = True
                                break
                    if file_found:
                        break
                
                if not file_found:
                    print(f"  > '{TARGET_FILENAME}' not found in '{TARGET_RESOURCE_LABEL}' for {short_name}")
                    count_not_found += 1

            print(f"\n=== Summary ===")
            print(f"Downloaded: {count_downloaded}")
            print(f"Skipped (already exist): {count_skipped}")
            print(f"Not found: {count_not_found}")
            
        else:
            print(f"Project {project_id} not found.")

except Exception as e:
    print(f"Connection failed or error occurred: {e}")
