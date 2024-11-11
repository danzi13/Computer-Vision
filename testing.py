import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import os
import cv2
from SSDtesting import crop_card, classify_card_by_center, bounding
import re
import numpy as np

# Define the collate function to handle batch loading
def collate_fn(batch):
    images, targets = zip(*batch)  # Separate images and targets (bounding boxes + labels)
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    return list(images), list(targets)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 320))  # Resize to match SSD input size
])
# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained SSD model and its weights
model = ssdlite320_mobilenet_v3_large(weights='SSDLite320_MobileNet_V3_Large_Weights.COCO_V1')
model.load_state_dict(torch.load('ssd_card_model.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Example folder path for test data
test_folder = 'photos/test'

# Label mappings
label_mapping_value = {
    "A": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "J": 10, "Q": 11, "K": 12
}

label_mapping_suit = {
    "C": 0, "D": 1, "H": 2, "S": 3
}

# Helper function to extract Value and Suit from filename
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

# Load test images and corresponding labels
test_data = []
file_list = [f for f in os.listdir(test_folder) if f.endswith('.png') or f.endswith('.jpg')]

for filename in file_list:
    img_path = os.path.join(test_folder, filename)
    # Crop the card image
    cropped_card_image, card_contour = crop_card(img_path)

    if cropped_card_image is not None:
        # Classify card features (Value, Suit, etc.)
        card_features = classify_card_by_center(cropped_card_image, bounding(cropped_card_image, card_contour))
        value_boxes = card_features.get("Value")
        suit_boxes = card_features.get("Suit")
        cropped_image = card_features.get("Image (Crop)")

        # Extract Value and Suit from filename
        value, suit = extract_value_and_suit_from_filename(filename)

        # Convert extracted value and suit to scalar labels
        if value in label_mapping_value and suit in label_mapping_suit:
            value_index = label_mapping_value[value]
            suit_index = label_mapping_suit[suit]
        else:
            print(f"Skipping image {filename} due to invalid value or suit: {value}, {suit}")
            continue

        # Process Value bounding boxes
        if value_boxes:
            for box in value_boxes:
                if box is not None:
                    box_tensor = torch.as_tensor(box, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                    labels_tensor = torch.as_tensor([value_index], dtype=torch.int64)  # Only value label
                    test_data.append((transform(cropped_image), {"boxes": box_tensor, "labels": labels_tensor}))

        # Process Suit bounding boxes
        if suit_boxes:
            for box in suit_boxes:
                if box is not None:
                    box_tensor = torch.as_tensor(box, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
                    labels_tensor = torch.as_tensor([suit_index], dtype=torch.int64)  # Only suit label
                    test_data.append((transform(cropped_image), {"boxes": box_tensor, "labels": labels_tensor}))

# Create DataLoader for test data
test_loader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Run inference on test data
total_correct = 0
total_images = 0

with torch.no_grad():
    for images_batch, targets_batch in test_loader:
        images_batch = [image.to(device) for image in images_batch]
        targets_batch = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

        # Make predictions
        outputs = model(images_batch)

        for output, target in zip(outputs, targets_batch):
            predicted_labels = output['labels']
            true_labels = target['labels']
            correct = (predicted_labels == true_labels).sum().item()
            total_correct += correct
            total_images += len(true_labels)

accuracy = total_correct / total_images
print(f"Accuracy on test data: {accuracy:.4f}")
