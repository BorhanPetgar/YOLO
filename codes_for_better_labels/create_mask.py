import os
import json
import numpy as np
from PIL import Image, ImageDraw

def create_mask_from_json(json_path, image_path, output_mask_folder, labels, label_to_id):
    # Step 1: Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Step 2: Read the corresponding image
    image = Image.open(image_path)
    width, height = image.size
    
    # Step 3: Extract the points from the JSON data
    shapes = data['shapes']
    yolo_annotations = []
    for shape in shapes:
        if shape['label'] in labels:
            points = shape['points']
            points = [(int(x), int(y)) for x, y in points]  # Ensure points are tuples of integers
            
            # Step 4: Create a mask from the points
            mask = Image.new('L', (width, height), 0)
            ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
            mask = np.array(mask)
            
            # Save the mask as an image
            mask_image = Image.fromarray(mask * 255)
            mask_output_path = os.path.join(output_mask_folder, f"{os.path.splitext(os.path.basename(json_path))[0]}_{shape['label']}.png")
            mask_image.save(mask_output_path)
            
            # Step 5: Convert polygon to bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            center_x = x_min + bbox_width / 2
            center_y = y_min + bbox_height / 2
            
            # Normalize coordinates
            center_x /= width
            center_y /= height
            bbox_width /= width
            bbox_height /= height
            
            # Create YOLO annotation
            class_id = label_to_id[shape['label']]
            yolo_annotations.append(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}")
    
    # Save YOLO annotations to a text file
    yolo_output_path = os.path.join(output_mask_folder, f"{os.path.splitext(os.path.basename(json_path))[0]}.txt")
    with open(yolo_output_path, 'w') as f:
        f.write("\n".join(yolo_annotations))

def process_folders(image_folder, json_folder, output_mask_folder, labels):
    # Ensure the output folder exists
    os.makedirs(output_mask_folder, exist_ok=True)
    
    # Get list of image and JSON files
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder) if f.endswith(('.jpeg', '.jpg', '.png'))}
    json_files = {os.path.splitext(f)[0]: f for f in os.listdir(json_folder) if f.endswith('.json')}
    
    # Create a mapping from labels to class IDs
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    
    # Process each pair of image and JSON files with matching names
    for name in image_files.keys() & json_files.keys():
        image_path = os.path.join(image_folder, image_files[name])
        json_path = os.path.join(json_folder, json_files[name])
        create_mask_from_json(json_path, image_path, output_mask_folder, labels, label_to_id)

# Example usage
image_folder = '/home/borhan/Desktop/test'
json_folder = '/home/borhan/Desktop/lab'
output_mask_folder = '/home/borhan/Desktop/out'
labels = ['human', 'dog', 'cat']  # List of labels to create masks for

process_folders(image_folder, json_folder, output_mask_folder, labels)
