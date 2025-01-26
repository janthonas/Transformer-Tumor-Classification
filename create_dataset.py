import os
import shutil

from sklearn.model_selection import train_test_split

# Removing Pesky Files

# Traverse the directory and remove all .DS_Store files
def clean_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            if file == ".DS_Store":
                os.remove(os.path.join(root, file))
                print(f"Removed: {os.path.join(root, file)}")
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):  # Check if folder is empty
                os.rmdir(dir_path)
                print(f"Removed empty folder: {dir_path}")

# Used to split dataset into Train, Test and Validation  
def split_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio):
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # List all image files
        images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]

        # Split into train, val, and test sets
        train_val_images, test_images = train_test_split(images, test_size=test_ratio, random_state=42)
        train_images, val_images = train_test_split(train_val_images, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

        # Move files to corresponding folders
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'train', class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'val', class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'test', class_name, img))

# Creating Train, Val, Test Dataset              
input_dir = "mri-tumor"
output_dir = "mri-tumor-org"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

for split in ['train', 'val', 'test']:
    for class_name in os.listdir(input_dir):
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
        
                
clean_directory(input_dir)
print("Directory cleaned.")

split_dataset(input_dir, output_dir, train_ratio, val_ratio, test_ratio)
clean_directory(output_dir)
print("Directory cleaned.")
print("Dataset Split")

