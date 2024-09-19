# Computer-Vision

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

