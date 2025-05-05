import os
import shutil
import yaml
import random
from pathlib import Path

def create_directory_structure(base_dir):
    """Create the YOLO dataset directory structure"""
    # Main directories
    os.makedirs(os.path.join(base_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", "test"), exist_ok=True)
    
    print(f"Created directory structure at {base_dir}")
    
def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """Split dataset into train, validation and test sets, ensuring matching names"""
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory {image_dir} does not exist")
    
    if not os.path.exists(label_dir):
        raise ValueError(f"Label directory {label_dir} does not exist")
    
    # Create the directory structure
    create_directory_structure(output_dir)
    
    # Get all image files and their basenames without extensions
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_basenames = [os.path.splitext(f)[0] for f in image_files]
    
    # Get all label files and their basenames without extensions
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
    label_basenames = [os.path.splitext(f)[0] for f in label_files]
    
    # Find common basenames (images that have corresponding labels)
    common_basenames = set(image_basenames).intersection(set(label_basenames))
    
    if not common_basenames:
        raise ValueError("No matching image-label pairs found")
    
    # Get the actual files with matching names
    valid_image_files = [f"{name}{os.path.splitext(img)[1]}" for name in common_basenames 
                        for img in image_files if os.path.splitext(img)[0] == name]
    
    # Check ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Train, validation and test ratios must sum to 1")
    
    # Shuffle files
    random.seed(random_seed)
    valid_basenames = list(common_basenames)
    random.shuffle(valid_basenames)
    
    # Calculate split indices
    train_end = int(len(valid_basenames) * train_ratio)
    val_end = train_end + int(len(valid_basenames) * val_ratio)
    
    # Split the files
    train_basenames = valid_basenames[:train_end]
    val_basenames = valid_basenames[train_end:val_end]
    test_basenames = valid_basenames[val_end:]
    
    # Function to copy files by basename
    def copy_files_by_basename(basenames, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
        count = 0
        for basename in basenames:
            # Find the corresponding image file (could have different extensions)
            image_file = next((f for f in os.listdir(src_img_dir) 
                              if os.path.splitext(f)[0] == basename), None)
            
            if not image_file:
                continue
                
            label_file = f"{basename}.txt"
            
            # Copy image
            src_img_path = os.path.join(src_img_dir, image_file)
            dst_img_path = os.path.join(dst_img_dir, image_file)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy label
            src_label_path = os.path.join(src_label_dir, label_file)
            dst_label_path = os.path.join(dst_label_dir, label_file)
            
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
                count += 1
            else:
                print(f"Warning: Label file {label_file} not found")
        
        return count
    
    # Copy files to respective directories
    train_count = copy_files_by_basename(
        train_basenames, image_dir, label_dir, 
        os.path.join(output_dir, "images", "train"), 
        os.path.join(output_dir, "labels", "train")
    )
    
    val_count = copy_files_by_basename(
        val_basenames, image_dir, label_dir, 
        os.path.join(output_dir, "images", "val"), 
        os.path.join(output_dir, "labels", "val")
    )
    
    test_count = copy_files_by_basename(
        test_basenames, image_dir, label_dir, 
        os.path.join(output_dir, "images", "test"), 
        os.path.join(output_dir, "labels", "test")
    )
    
    print(f"Dataset split complete: {train_count} train, {val_count} validation, {test_count} test")
    
    return train_count, val_count, test_count

def create_yaml_file(output_dir, class_names, train_count, val_count, test_count=0):
    """Create the data.yaml file"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    
    data = {
        "path": output_dir,  # dataset root dir
        "train": "images/train",  # train images (relative to 'path')
        "val": "images/val",      # val images (relative to 'path')
        "test": "images/test",    # test images (optional)
        "nc": len(class_names),   # number of classes
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created data.yaml file at {yaml_path}")
    
    # Also create classes.names file for compatibility
    classes_path = os.path.join(output_dir, "classes.names")
    with open(classes_path, 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    
    print(f"Created classes.names file at {classes_path}")

def main():
    # Read configuration from YAML
    config_path = "dataset_config.yaml"
    
    if not os.path.exists(config_path):
        # Create a default config file if it doesn't exist
        default_config = {
            "output_dir": "YOLO_dataset",
            "image_dir": "/path/to/source/images",
            "label_dir": "/path/to/source/labels",
            "classes": ["obj1", "obj2", "obj3"],
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "random_seed": 42
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default configuration file at {config_path}")
        print("Please edit the configuration file and run the script again.")
        return
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create absolute paths
    output_dir = os.path.abspath(config.get("output_dir", "YOLO_dataset"))
    image_dir = os.path.abspath(config.get("image_dir", ""))
    label_dir = os.path.abspath(config.get("label_dir", ""))
    classes = config.get("classes", ["unknown"])
    
    # Split dataset
    train_count, val_count, test_count = split_dataset(
        image_dir, 
        label_dir, 
        output_dir, 
        train_ratio=config.get("train_ratio", 0.7), 
        val_ratio=config.get("val_ratio", 0.2), 
        test_ratio=config.get("test_ratio", 0.1),
        random_seed=config.get("random_seed", 42)
    )
    
    # Create YAML file
    create_yaml_file(output_dir, classes, train_count, val_count, test_count)
    
    print("YOLO dataset structure created successfully.")

if __name__ == "__main__":
    main()