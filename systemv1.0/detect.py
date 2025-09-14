import shutil
import os
import glob
from ultralytics import YOLO

def move_detection_results(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        if os.path.exists(target_file):
            if os.path.isdir(target_file):
                shutil.rmtree(target_file)
            else:
                os.remove(target_file)
        shutil.move(source_file, target_dir)
    shutil.rmtree(source_dir)

def detect_images(images_folder, model_path, callback=None):
    model = YOLO(model_path)
    results = model.predict(images_folder, save=True, save_txt=True, imgsz=640, conf=0.5)
    latest_run_dir = max(glob.glob(os.path.join('runs', 'detect', '*')), key=os.path.getmtime)
    results_dir = os.path.join(images_folder, 'results')
    move_detection_results(latest_run_dir, results_dir)
    if callback:
        callback(results_dir)

# import os
# import shutil
# from pathlib import Path
# from ultralytics import YOLO

# def move_detection_results(source_dir, target_dir):
#     """
#     Moves files from source directory to target directory.
#     If files with the same name already exist in the target directory, they are removed first.
#     """
#     # Ensure target directory exists
#     Path(target_dir).mkdir(parents=True, exist_ok=True)

#     # Iterate through files in the source directory
#     for file_name in os.listdir(source_dir):
#         source_file = os.path.join(source_dir, file_name)
#         target_file = os.path.join(target_dir, file_name)
        
#         # Remove target file if it exists
#         if os.path.exists(target_file):
#             if os.path.isdir(target_file):
#                 shutil.rmtree(target_file)
#             else:
#                 os.remove(target_file)

#         # Move the file from source to target
#         shutil.move(source_file, target_dir)

#     # Remove the source directory after moving files
#     shutil.rmtree(source_dir)

# def detect_images(images_folder, model_path, callback=None):
#     """
#     Detects objects in images using a pre-trained YOLO model and moves the results to a target folder.
#     Optionally invokes a callback with the path to the results directory.
#     """
#     model = YOLO(model_path)
    
#     # Perform prediction on the image folder
#     results = model.predict(images_folder, save=True, save_txt=True, imgsz=640, conf=0.5)

#     # Define the path to the 'runs/detect' folder
#     runs_detect_dir = os.path.join('runs', 'detect')
    
#     # Ensure the directory exists before proceeding
#     if not os.path.exists(runs_detect_dir):
#         os.makedirs(runs_detect_dir)
#         print(f"Created directory: {runs_detect_dir}")

#     # Get the most recent directory by modification time
#     latest_run_dir = None
#     latest_time = 0  # Start with an arbitrary low time

#     # Loop through each subdirectory in 'runs/detect'
#     for subdir in os.listdir(runs_detect_dir):
#         subdir_path = os.path.join(runs_detect_dir, subdir)
        
#         # Only consider directories
#         if os.path.isdir(subdir_path):
#             subdir_mtime = os.path.getmtime(subdir_path)  # Get the modification time of the subdirectory
#             if subdir_mtime > latest_time:
#                 latest_time = subdir_mtime
#                 latest_run_dir = subdir_path
    
#     # If no valid subdirectory is found, raise an error
#     if latest_run_dir is None:
#         raise ValueError(f"No valid detection subdirectories found in '{runs_detect_dir}'.")

#     # Define the target results directory within the images folder
#     results_dir = os.path.join(images_folder, 'results')
    
#     # Move the detection results to the results directory
#     move_detection_results(latest_run_dir, results_dir)
    
#     # If a callback is provided, call it with the results directory
#     if callback:
#         callback(results_dir)
