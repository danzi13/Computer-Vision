import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2


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
model.load_state_dict(torch.load('ssd_card_model.pth', map_location=device))
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
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8')

    # Match common card patterns like `10C-02` or `QD-02`
    match = re.match(r'(\d{1,2}|[AJQK])([CDHS])-\d+', filename)
    if match:
        value = match.group(1)  # Value (e.g., "10", "A", "J")
        suit = match.group(2)   # Suit (e.g., "C", "D", "H", "S")
        return value, suit

    print(f"Invalid filename format: {filename}")
    return None, None

# Load test images and corresponding labels
test_data = []
file_list = [f for f in os.listdir(test_folder) if f.endswith('.png') or f.endswith('.jpg')]

for filename in file_list:
    value, suit = extract_value_and_suit_from_filename(filename)
    if value is not None and suit is not None:
        if value in label_mapping_value and suit in label_mapping_suit:
            test_data.append((filename, label_mapping_value[value], label_mapping_suit[suit]))
        else:
            print(f"Skipping image {filename} due to invalid mapping.")

# Initialize variables for metrics
all_true_labels = []
all_predicted_labels = []

with torch.no_grad():
    for filename, true_value, true_suit in test_data:
        # Load image
        img_path = os.path.join(test_folder, filename)
        image = transform(cv2.imread(img_path)).to(device).unsqueeze(0)

        # Run inference
        outputs = model(image)

        if len(outputs) > 0 and 'labels' in outputs[0]:
            predicted_labels = outputs[0]['labels'].tolist()
            if predicted_labels:
                all_true_labels.append(true_value)  # Append true value label
                all_predicted_labels.append(predicted_labels[0])  # Append predicted label
            else:
                print(f"No labels detected for {filename}")
        else:
            print(f"Invalid output for {filename}")

# Calculate performance metrics
precision = precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
recall = recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
