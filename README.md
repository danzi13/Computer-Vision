# Computer-Vision

### Part 3
NOTE: Git push origin main: did not let me push because of large files so this is uploaded through github.com, please contact if all the files aren’t there to run the code

**Methods**: 

**1. Preprocessing and Justification**

- The python script SSDtesting.py contains 3 preprocessing methods, that are imported when training and testing: 
-crop_card(image_path),
-classify_card_by_center(cropped_card_image, bounding_boxes), 
-bounding(cropped_card_image, contour),
I will go into detail of each method, but what these 3 functions do is crop the card, bound boxes on the card (while removing noise and joining close bondings) and
-returns: {"Value": value_boxes,"Suit": suit_boxes,"Middle": middle_box, "Color": color, "Image (Crop)": cropped_card_image}
This is the original image:
![Card Original](readme_images/original_card.png)
**Crop_Card(image_path):**
This function loads the image in grayscale which simplifies the edge detection on the table for our card. We then apply Gaussian blur to reduce noise ensuring edge detection isn’t influenced by small details. We then used Sobel for actual edge detection which was useful for identifying straight edges crucial for our horizontal and vertical gradients. After edge detection the image is thresholded to create a binary image (edges = white, non-edges = black). This step simplifies the edges, making it easier to detect contours and, ultimately, the card itself. We then findContours to identify the largest one (which will be the card), we use this largest contour to isolate the card itself from the image to crop and only focus on feature extraction from this card.Blurring is essential to smooth out noise, which could lead to false edges or small irrelevant contours. It ensures that the edge detection only highlights the meaningful boundaries. Tried to replicate what I learned from the cereal Sobel Operator: Sobel was a standard edge detection operator that computes intensity gradients, making it a suitable method for detecting the sharp, well-defined edges of a card. Sobel’s simple approach is ideal for detecting straight edges in relatively clean images. Thresholding: This is just the binary format we used in class so I continued from this baseline Contour Detection: Again is just the contour format we used in class so I continued from this baseline Bounding Box and Cropping: These are also basic methods learned from class just used cv_image show to display image, not even really sure if cropping can be more “efficient” 
**This is the cropped image:**
![Card Crop](readme_images/cropped_card.png)


**Bounding(cropped_image, contour):**
This function is basically to clean up the card itself. We detect boxes, remove noise (of small irrelevant detections), merge boxes that are close together and display results. Parameters are a cropped_image and it returns a list of bounding boxes. We do the following steps to achieve this: grayscale conversion, image into grayscale -> convert image to binary based on intensity values, region labeling connection regions in the binary image, bounding box extraction with filtering of ones that are too small, then merge bounding boxes. The function isolates features by thresholding, labeling regions, and extracting bounding boxes. Nearby bounding boxes are merged to prevent misclassification of adjacent features. Not going to repeat with justifications of methods used above (grayscale conversion, thresholding, regional labeling). I will start from bounding boxes, extracts using properties = measure.regionprops(labels), I used this method because it was effective with the cereal and I was familiar with it using it to calculate stuff like area from properties which was helpful for merging based on overlapping pixels or removing if I think the object is too small. I merge objects nearby boxes to improving the accuracy of feature detection and reducing false positives. This is particularly useful when objects are close to each other but detected as separate regions.
**This is the object detection on card**
![Card with Box](readme_images/bounding_boxes.png)



**classify_card_by_center(cropped_card_image, bounding_boxes):**

This method was the most interesting to code up. This method uses a combination of geometric, spatial, and color features to classify and segment card elements. By analyzing bounding boxes, measuring distances, and pairing related boxes, the algorithm is able to accurately extract key features like the card’s value and suit. Removing the Largest Bounding Box, This largest bounding box is the card itself so we no longer need this for identification of the card. Knowing the card itself is in the crop I find the center of the card. This center is found with image dimensions and then is used to measure relative distance of bounding boxes from this center (which I will explain why). We then use distance calculation between bounding boxes and card center to identify the boxes that are closest to center which is the middle of the card, then the furthest pair away from the center is the cards value and then next furthest is the cards suit as pictured below. There is a function to look for pairs, some problems I ran into was off the card detection furthest from middle so I made sure that when finding outer edges there is something equal distant and the opposite direction. I also implemented red or black detection on the middle object (since its uniformity center is the color). I then Return: The function returns a dictionary containing the bounding boxes for the value, suit, and middle features, the color detected, and the cropped image. This is the final output, useful for downstream tasks like classification or visualization.
**Here is the card boxes:**
![Card Middle](readme_images/middle.png)

![Card Original](readme_images/values.png)

![Card Original](readme_images/suit.png)


## Part 2: Data Collection and Preparation

### 1. Source of Data
- **Collected Data**: The dataset consists of images of playing cards, captured manually using a phone camera. The images were taken under controlled indoor lighting conditions to ensure consistency across the dataset. Each card was photographed five times, resulting in a diverse dataset suitable for training, validation, and testing. I collected data using the camera.py script allowing me to take photos automatically speeding up the process and allowing 
  
  - The images are named in the format `XX-01.png`, where `XX` represents the card identifier (e.g., `KS` for King of Spades) and the number represents the sample.
  
### 2. Differences Between Training and Validation Subsets, these sets can be found in photos within each respective folder.
- **Training Set**: 60% (3) of the images were randomly selected for training. This set was collected with a fixed background, consistent lighting, and slight variations in positioning.
  
- **Validation Set**: 20% (1) of the images were used for validation. The validation images include minor variations in card angles, and slight changes in lighting or background, which helps evaluate the model's generalization.
  
- **Test Set (Unknown)**: 20% (1) of the images were set aside for final testing. These images will remain untouched until the final evaluation to avoid bias.

- These were chosen randomly in each resepctive folder using a script (card-dist.py) to speed up the process.

### 3. Number of Distinct Objects/Subjects
- The dataset contains **52 distinct card types** (e.g., King of Spades, Ace of Hearts, etc.). +2 different sets of Jokers and 5 of the back of the cards. Each card has **5 samples**, leading to a total of 275 images.

### 4. Characterization of Samples
- **Resolution**: The images are captured at a resolution of 1080x1920 pixels.
  
- **Sensors Used**: The images were captured using a smartphone camera (iPhone) with an automatic focus sensor.
  
- **Lighting and Ambient Conditions**: The dataset was collected under controlled indoor lighting, ensuring uniform exposure across the images. Variations

-End of Part2


# Poker Hand Identification Using Computer Vision
## Problem Overview
The goal of this project is to use computer vision to identify poker hands from images. This would involve detecting playing cards, then recognizing suits, ranks and value within the game and evaluating the hand based on the card combinations. Some challenges would include lighting, card orientation and different card designs. 
## Part 2: Approach
### Card Detection
The first step is to locate the playing cards within the image. Since cards are uniformly shaped rectangles, traditional object detection techniques such as edge detection or bounding boxes can be applied to find the card in the frame. 
### 2. Card Recognition
There are already concrete datasets of identifying playing cards for us from the website already provided. ​​http://www2.imse-cnm.csic.es/caviar/POKERDVS.html. The first part would be identifying traditional card designs, first the color, then suit, then rank. 
### 3. Poker Hand Evaluation
I think for this part we can use APIs to pull hand odds calculations. For specifically Texas Hold ‘Em depending on how many cards are shown we can even have pre flop odds, middle game odds and ending game odds to win. Then we could also give the cards on the board into either a library or API and give the classifier for pair, two pair, and more complex hands along with odds to win. 
##: Dataset Requirements
### 1. Training Dataset
The training data should have all playing cards. Maybe even some jokers to identify cards that maybe aren’t cards or maybe they're not being shown. Single cards would be used for classification and the backend will do hand calculations. We can have diverse conditions like angle for identifying cards, different lighting etc. The main concept is playing cards being the dataset but we might need additional data. 


### 2. Validation Dataset
We can have datasets of different playing cards that were not included in the training deck either to further prove our model with identifying stuff like suit and numbers or use to help tune and avoid overfitting. 
### 3. Testing Dataset
We will have certain cards not entered into the training dataset because if we can identify numbers and suit then we could get every card without all of them being in training. We can have different card variations, lighting and angles. 
##: Conclusion
This project will involve a complete computer vision pipeline to detect, recognize and evaluate poker hands. We will break the problem down into detection of cards, recognition of the actual value and then evaluation stages. I think given enough time and granted success I would like to implement the option of seeing which stage of the game we are in by seeing the amount of face down cards. Seeing 2 cards (your own hand) with 5 backs of cards is “preflop”, then “post flop” then “turn” then “river.” Challenges such as different card designs, lighting, angles will all come into effect. I think with enough training we can preserve. This is the high-level plan that outlines the direction for the project and we will adjust with new techniques throughout the semester. 

