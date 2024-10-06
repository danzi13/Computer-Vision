# Computer-Vision

## Part 2: Data Collection and Preparation

### 1. Source of Data
- **Collected Data**: The dataset consists of images of playing cards, captured manually using a phone camera. The images were taken under controlled indoor lighting conditions to ensure consistency across the dataset. Each card was photographed five times, resulting in a diverse dataset suitable for training, validation, and testing. I collected data using the camera.py script allowing me to take photos automatically speeding up the process and allowing 
  
  - The images are named in the format `XX-01.png`, where `XX` represents the card identifier (e.g., `KS` for King of Spades) and the number represents the sample.
  
### 2. Differences Between Training and Validation Subsets
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

