import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np

def crop_card(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(image_path)  # Keep the original color image for displaying results

    # Step 1: Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 2: Apply Sobel operator to find edges
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the Sobel x and y results
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = np.uint8(sobel_combined)

    # Step 3: Threshold the Sobel result to create a binary image
    _, binary_image = cv2.threshold(sobel_combined, 50, 255, cv2.THRESH_BINARY)

    # Step 4: Find contours based on the binary Sobel edge map
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Identify the largest contour, assuming it's the card
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Step 6: Calculate the bounding rectangle for cropping
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Step 7: Crop the card from the original image
        cropped_card = original_image[y:y+h, x:x+w]
        plt.imshow(cv2.cvtColor(cropped_card, cv2.COLOR_BGR2RGB))
        plt.title(f"{label} Boxes")
        plt.axis('off')
        plt.show()

        return cropped_card, largest_contour

    else:
        print("No contours detected.")
        return None, None

def classify_card_by_center(cropped_card_image, bounding_boxes):
    """
    Classify card features (Value, Suit, Middle) based on bounding boxes and their distance from the center.

    Parameters:
        cropped_card_image: The cropped card image.
        bounding_boxes: List of bounding boxes (x_min, y_min, x_max, y_max).

    Returns:
        A dictionary with the classified boxes and visualizations.
    """
    if bounding_boxes:
        largest_area = 0
        largest_index = -1

        for i, box in enumerate(bounding_boxes):
            x_min, y_min, x_max, y_max = box
            area = (x_max - x_min) * (y_max - y_min)
            if area > largest_area:
                largest_area = area
                largest_index = i

        # Remove the largest bounding box
        if largest_index != -1:
            bounding_boxes.pop(largest_index)
    # Calculate the center of the card
    card_center_x = cropped_card_image.shape[1] // 2
    card_center_y = cropped_card_image.shape[0] // 2

    # Calculate distances from the center for each box
    def calculate_box_center(box):
        x_min, y_min, x_max, y_max = box
        return ((x_min + x_max) // 2, (y_min + y_max) // 2)

    def calculate_distance_from_center(center):
        return np.sqrt((center[0] - card_center_x)**2 + (center[1] - card_center_y)**2)

    box_distances = []
    for box in bounding_boxes:
        center = calculate_box_center(box)
        distance = calculate_distance_from_center(center)
        box_distances.append((box, center, distance))

    #print(box_distances)
    # Sort boxes by distance from the center (closest to farthest)
    box_distances.sort(key=lambda x: x[2])

    #print(box_distances)

    # Middle box is the closest to the center
    middle_box = box_distances[0][0]
    remaining_boxes = box_distances[1:]  # Exclude the middle box

    # Group boxes into pairs by distance similarity
    # Group boxes into pairs by distance similarity and opposite directions

    def find_adjacent_pairs(boxes, distance_tolerance=30, angle_tolerance=0.05):
        
        pairs = []
        used_boxes = set()

        def direction_vector(center):
            # Compute direction vector from card center to box center
            return np.array([center[0] - card_center_x, center[1] - card_center_y])

        for i, (box1, center1, dist1) in enumerate(boxes):
            if box1 in used_boxes:
                continue

            best_pair = None
            min_distance_diff = float("inf")

            vec1 = direction_vector(center1)
            vec1_norm = vec1 / np.linalg.norm(vec1)

            for j, (box2, center2, dist2) in enumerate(boxes):
                if i == j or box2 in used_boxes:
                    continue

                # Compute direction vector and normalize
                vec2 = direction_vector(center2)
                vec2_norm = vec2 / np.linalg.norm(vec2)

                # Check if directions are opposite
                cosine_similarity = np.dot(vec1_norm, vec2_norm)
                is_opposite = cosine_similarity < -1 + angle_tolerance

                # Check if distances are similar
                distance_diff = abs(dist1 - dist2)

                if is_opposite and distance_diff <= distance_tolerance and distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    best_pair = box2

            if best_pair:
                pairs.append((box1, best_pair))
                used_boxes.add(box1)
                used_boxes.add(best_pair)

        pairs = list(set(tuple(sorted(pair)) for pair in pairs))

        return pairs



    adjacent_pairs = find_adjacent_pairs(remaining_boxes)
    sorted_pairs = sorted(
            adjacent_pairs,
            key=lambda pair: (
                ((pair[0][0] + pair[0][2]) // 2 - card_center_x) ** 2 +
                ((pair[0][1] + pair[0][3]) // 2 - card_center_y) ** 2 +
                ((pair[1][0] + pair[1][2]) // 2 - card_center_x) ** 2 +
                ((pair[1][1] + pair[1][3]) // 2 - card_center_y) ** 2
            ) ** 0.5
        )


    # Classify the largest pairs as Value, the second-largest as Suit
    value_boxes = adjacent_pairs[0] if len(adjacent_pairs) > 0 else None
    suit_boxes = adjacent_pairs[1] if len(adjacent_pairs) > 1 else None

    #for item in suit_boxes:
    #    print(calculate_distance_from_center(item))
    
    #for item in value_boxes:
    #    print(calculate_distance_from_center(item))

    # Visualize bounding boxes
    def visualize_boxes(cropped_card_image, label, boxes):
        """
        Visualize bounding boxes on the card with labels.
        """
        display_image = cropped_card_image.copy()

        if not boxes:
            return

        for i, box in enumerate(boxes):
            if isinstance(box, tuple) and len(box) == 2:
                # Draw a pair of boxes
                for single_box in box:
                    x_min, y_min, x_max, y_max = single_box
                    cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.putText(display_image, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                # Draw a single box
                x_min, y_min, x_max, y_max = box
                cv2.rectangle(display_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(display_image, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        plt.title(f"{label} Boxes")
        plt.axis('off')
        plt.show()

    #for item in adjacent_pairs:
    #    visualize_boxes(cropped_card_image, "3", item)

    #visualize_boxes(cropped_card_image, "Middle", [middle_box])
    #visualize_boxes(cropped_card_image, "Value", value_boxes)
    #visualize_boxes(cropped_card_image, "Suit", suit_boxes)

    def detect_red_or_black(image, bounding_box):
        
        x_min, y_min, x_max, y_max = bounding_box
        cropped_region = image[y_min:y_max, x_min:x_max]

        red_color = np.array([70, 70, 255])  # BGR format
        black_color = np.array([0, 0, 0])


        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        center_pixel = image[center_y, center_x]

        red_color = np.array([70, 70, 255])  # BGR format
        black_color = np.array([0, 0, 0])

        red_distance = np.linalg.norm(center_pixel - red_color)
        black_distance = np.linalg.norm(center_pixel - black_color)

        return "Red" if red_distance < black_distance else "Black"

    color = detect_red_or_black(cropped_card_image, middle_box)
    #print(color)
    # Return the classified boxes
    return {
        "Value": value_boxes,
        "Suit": suit_boxes,
        "Middle": middle_box,
        "Color": color,
        "Image (Crop)": cropped_card_image
    }

def bounding(cropped_card_image, contour):
    """
    Detect bounding boxes, remove noise, merge close boxes, and display the results.
    
    Parameters:
        cropped_card_image: Cropped card image.
        
    Returns:
        - List of valid combined bounding boxes.
    """
    # Convert to grayscale
    cropped_card_gray = cv2.cvtColor(cropped_card_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, binary_image = cv2.threshold(cropped_card_gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Label the regions
    labels = measure.label(binary_image, connectivity=2)
    properties = measure.regionprops(labels)

    # Step 1: Extract valid bounding boxes
    bounding_boxes = []
    for prop in properties:
        minr, minc, maxr, maxc = prop.bbox
        area = prop.area

        # Skip small/noisy regions
        if not isinstance(area, (int, float)) or area < 40:  # Ensure valid scalar and minimum size
            continue

        bounding_boxes.append((int(minc), int(minr), int(maxc), int(maxr)))
    

    # Step 2: Merge close bounding boxes manually
    merged_boxes = []
    for box in bounding_boxes:
        minc, minr, maxc, maxr = box
        merged = False

        for i, existing_box in enumerate(merged_boxes):
            e_minc, e_minr, e_maxc, e_maxr = existing_box

            # Check if boxes are close in both horizontal and vertical directions
            if (abs(minc - e_minc) < 30 and abs(minr - e_minr) < 30) or \
               (abs(maxc - e_maxc) < 30 and abs(maxr - e_maxr) < 30):
                # Merge the boxes
                new_minc = min(minc, e_minc)
                new_minr = min(minr, e_minr)
                new_maxc = max(maxc, e_maxc)
                new_maxr = max(maxr, e_maxr)
                merged_boxes[i] = (new_minc, new_minr, new_maxc, new_maxr)
                merged = True
                break

        if not merged:
            merged_boxes.append((minc, minr, maxc, maxr))

    

    return merged_boxes


# Example usage
cropped_card_image, card_contour = crop_card('photos/train/3S-02.png')
#cropped_card_image, card_contour = crop_card('photos/train/7C-04.png')
#cropped_card_image, card_contour = crop_card('photos/test/AC-02.png')

if cropped_card_image is not None:
    item = classify_card_by_center(cropped_card_image, bounding(cropped_card_image,card_contour))
    
