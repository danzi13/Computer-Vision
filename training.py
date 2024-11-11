import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import numpy as np
import cv2
import os
from SSDtesting import crop_card, classify_card_by_center, bounding
import re

# Define the collate function to handle batch loading
def collate_fn(batch):
    images, targets = zip(*batch)  # Separate images and targets (bounding boxes + labels)
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    return list(images), list(targets)

# Example folder path
folder = 'photos/train'

# Label mapping for Value and Suit (you should adjust based on your dataset)
label_mapping_value = {
    "A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "J": 10, "Q": 11, "K": 12  # Update this based on your actual card values
}

label_mapping_suit = {
    "C": 0, "D": 1, "H": 2, "S": 3  # Clubs, Diamonds, Hearts, Spades
}

def extract_value_and_suit_from_filename(filename):
    """Extract both the Value and Suit from the filename."""
    # Ensure the filename is a string (if it's bytes, decode it)
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8')

    # Adjusted regex pattern to capture the value and suit (handles single and double digit values)
    match = re.match(r'(\d{1,2})([A-Z])-(\d+)', filename)  # e.g., "10C-02" or "AC-02"
    
    if match:
        value = match.group(1)  # The value (e.g., "10" or "A")
        suit = match.group(2)   # The suit (e.g., "C", "H", etc.)
        return value, suit
    else:
        # If the match is not found, let's handle these cases where value and suit might not follow the expected pattern
        match = re.match(r'([A-Z])([A-Z])-.*', filename)  # e.g., for Joker-type cards (J, Q, K)
        if match:
            value = match.group(1)  # The value letter
            suit = match.group(2)   # The suit letter
            return value, suit
    return None, None  # Default return if not matched

def load_images_from_folder(folder):
    """
    Load images, crop them, and extract bounding boxes and class labels.
    """
    images = []
    labels = []
    bounding_boxes = []
    original_images = []

    file_list = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]
    
    print(f"Total files found: {len(file_list)}")

    for filename in file_list:
        img_path = os.path.join(folder, filename)

        # Step 1: Crop the card image
        cropped_card_image, card_contour = crop_card(img_path)
        
        if cropped_card_image is not None:
            # Step 2: Classify card features (Value, Suit, etc.)
            card_features = classify_card_by_center(cropped_card_image, bounding(cropped_card_image, card_contour))

            value_boxes = card_features.get("Value")
            suit_boxes = card_features.get("Suit")
            middle_box = card_features.get("Middle")
            color = card_features.get("Color")
            cropped_image = card_features.get("Image (Crop)")

            # Extract Value and Suit from filename
            value, suit = extract_value_and_suit_from_filename(filename)
            #print(f"Extracted Value: {value}, Suit: {suit}")

            # Convert the extracted value and suit to scalar using mappings
            if value in label_mapping_value and suit in label_mapping_suit:
                value_index = label_mapping_value[value]
                suit_index = label_mapping_suit[suit]
            else:
                print(f"Skipping image {filename} due to invalid value or suit: {value}, {suit}")
                continue

            # Step 3: Process each bounding box individually
            # Process Value bounding boxes
            if value_boxes:
                for box in value_boxes:
                    if box is not None:
                        box_tensor = torch.as_tensor(box, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                        labels_tensor = torch.as_tensor([value_index], dtype=torch.int64)  # Only value label
                        images.append(cropped_image)
                        bounding_boxes.append(box_tensor)
                        labels.append(labels_tensor)

            # Process Suit bounding boxes
            if suit_boxes:
                for box in suit_boxes:
                    if box is not None:
                        box_tensor = torch.as_tensor(box, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                        labels_tensor = torch.as_tensor([suit_index], dtype=torch.int64)  # Only suit label
                        images.append(cropped_image)
                        bounding_boxes.append(box_tensor)
                        labels.append(labels_tensor)

            # Store the original image for visualization purposes
            original_image = cv2.imread(img_path)
            original_images.append(original_image)

    return images, labels, bounding_boxes, original_images

# Prepare the data
images, labels, bounding_boxes, original_images = load_images_from_folder(folder)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 320))  # Resize to match SSD input size
])
# Check how many images were loaded
print(f"Loaded {len(images)} images.")
# Prepare the train_data list from images, labels, and bounding_boxes
train_data = []
for i in range(len(images)):
    image = images[i]
    boxes = bounding_boxes[i]
    
    for box in boxes:
        if box is not None:
            box_tensor = torch.as_tensor(box, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            labels_tensor = torch.as_tensor(labels[i], dtype=torch.int64)  # Convert labels into tensor

            if isinstance(image, np.ndarray):  # If the image is not already a tensor
                image = transform(image)  # Convert the image to a tensor using the defined transform
            
            target = {"boxes": box_tensor, "labels": labels_tensor}
            train_data.append((image, target))

# Now you can use train_data with DataLoader
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)



# Load the pre-trained SSD model
model = ssdlite320_mobilenet_v3_large(weights='SSDLite320_MobileNet_V3_Large_Weights.COCO_V1')
model.train()  # Set the model to training mode

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 10  # Adjust this based on your needs

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for images_batch, targets_batch in train_loader:
        # Move data to the device (GPU if available)
        images_batch = [image for image in images_batch]  # Move to CPU or GPU as needed
        targets_batch = [{k: v for k, v in t.items()} for t in targets_batch]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images_batch, targets_batch)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'ssd_card_model.pth')

print("Model training complete and saved.")
