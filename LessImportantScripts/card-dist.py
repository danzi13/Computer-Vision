import os
import shutil
import random

def create_folders(base_folder):
    """Create train, validation, and test folders."""
    train_folder = os.path.join(base_folder, 'train')
    val_folder = os.path.join(base_folder, 'validation')
    test_folder = os.path.join(base_folder, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    return train_folder, val_folder, test_folder

def distribute_images(card_name, image_files, folder, train_folder, val_folder, test_folder):
    """Randomly distribute images per card into train, validation, and test folders."""
    # Shuffle the image file list randomly
    random.shuffle(image_files)

    # Move 3 random images to training
    for img in image_files[:3]:
        src_path = os.path.join(folder, img)
        dst_path = os.path.join(train_folder, img)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"Moved {img} to training folder")

    # Move 1 random image to validation
    val_img = image_files[3]
    src_path = os.path.join(folder, val_img)
    dst_path = os.path.join(val_folder, val_img)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved {val_img} to validation folder")

    # Move 1 random image to test
    test_img = image_files[4]
    src_path = os.path.join(folder, test_img)
    dst_path = os.path.join(test_folder, test_img)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved {test_img} to test folder")

def main():
    # Base folder where your card images are located
    base_folder = 'photos'

    # Create train, validation, and test folders
    train_folder, val_folder, test_folder = create_folders(base_folder)

    # Automatically detect all the card names based on the file names in the folder
    card_dict = {}
    
    # List all files in the base folder
    for filename in os.listdir(base_folder):
        if filename.endswith(".png"):
            # Extract card name (e.g., KS from KS-01.png)
            card_name = filename.split('-')[0]
            if card_name not in card_dict:
                card_dict[card_name] = []
            card_dict[card_name].append(filename)

    # For each card, distribute the images randomly
    for card_name, image_files in card_dict.items():
        if len(image_files) >= 5:
            distribute_images(card_name, image_files, base_folder, train_folder, val_folder, test_folder)
        else:
            print(f"Not enough images for card: {card_name}. Skipping.")

    print("All images distributed.")

if __name__ == "__main__":
    main()
