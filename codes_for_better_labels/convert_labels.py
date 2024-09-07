import os
import cv2
import numpy as np

def load_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        labels = file.readlines()
    boxes = []
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes

def save_yolo_labels(label_path, boxes):
    with open(label_path, 'w') as file:
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def adjust_boxes(boxes, roi, img_width, img_height):
    x_min, y_min, x_max, y_max = roi
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    adjusted_boxes = []
    for box in boxes:
        class_id, x_center, y_center, width, height = box
        # Convert YOLO format to absolute coordinates
        abs_x_center = x_center * img_width
        abs_y_center = y_center * img_height
        abs_width = width * img_width
        abs_height = height * img_height
        # Adjust coordinates based on the ROI
        new_x_center = (abs_x_center - x_min) / roi_width
        new_y_center = (abs_y_center - y_min) / roi_height
        new_width = abs_width / roi_width
        new_height = abs_height / roi_height
        # Only keep boxes that are within the ROI
        if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1:
            adjusted_boxes.append((class_id, new_x_center, new_y_center, new_width, new_height))
    return adjusted_boxes

def process_images_and_labels(image_folder, label_folder, output_image_folder, output_label_folder, roi):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    for image_name in os.listdir(image_folder):
        if image_name.endswith('.jpg'):
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(image_folder, image_name)
            label_path = os.path.join(label_folder, base_name + '.txt')

            if os.path.exists(label_path):
                # Load image and labels
                image = cv2.imread(image_path)
                img_height, img_width = image.shape[:2]
                boxes = load_yolo_labels(label_path)

                # Crop the image
                x_min, y_min, x_max, y_max = roi
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Adjust the bounding boxes
                adjusted_boxes = adjust_boxes(boxes, roi, img_width, img_height)

                # Save the cropped image and adjusted labels
                cropped_image_path = os.path.join(output_image_folder, image_name)
                cropped_label_path = os.path.join(output_label_folder, base_name + '.txt')
                cv2.imwrite(cropped_image_path, cropped_image)
                save_yolo_labels(cropped_label_path, adjusted_boxes)

                print(f"Processed {image_name} and {base_name}.txt")

# Define the folders
image_folder = '/home/borhan/Desktop/for_convert_images'
label_folder = '/home/borhan/Desktop/for_convert_labels'
output_image_folder = '/home/borhan/Desktop/crops_images'
output_label_folder = '/home/borhan/Desktop/crops_labels'

# Define the ROI (top-left and bottom-right corners)
roi = (600, 150, 905, 360)  # Example ROI, you can change these values

# Process the images and labels
process_images_and_labels(image_folder, label_folder, output_image_folder, output_label_folder, roi)