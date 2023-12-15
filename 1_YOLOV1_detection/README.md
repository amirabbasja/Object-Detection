This directory contains a simple object detection using YOLOv1 algorithm.

# Notes
1. The input image for training is resized to 448x448x3 tensors.
2. The image is divided into SxS grids (S = 7 in the original paper). 
3. Each grid cell is consisted of 64x64 pixels and is responsible with one prediction.
4. Each grid cell is responsible for predicting one object. The cell in which the object's center is located is responsible for its prediction. In other words, each grid cell can predict no more than 1 object. So the maximum number of objects that can be detected in a n image is 49 for YOLOv1. 
5. Each grid cell outputs target values which define the object's location (x,y,w,h). These parameters are called targets and should be close to the ground truth values.
6. Predicting the object location is done with relative absolute values instead of absolute values to make the training process easier (also called the Label encoding process)
7. The relative target values for the object from its absolute values are calculated below:
    Taking (x_a,y_a) to be the absolute coordinates of the anchor top-left corner (the cell's) we have
    x_rel = (x-x_a)/64
    y_rel = (y-y_a)/64
    w_rel = w/448
    h_rel = h/448
    Where (x,y,w,h) are he absolute target values of the bounding box.
    Note that the reason why we do this, is to convert the target values smaller numbers in the (0,1) range. which helps with the training process.
    Note that the absolute coordinate axes are locates at the picture's top-left corner (i.e. As we move to the right x increases and as we move to the bottom of the image y increases) 
8. The target values for grid cells not containing any objects are all zeros.
9. The prediction vector:
    - Each grid cell (anchor box) predicts two bounding boxes (B=2). Each bounding box is represented by a vector (x_rel,y_rel,w_rel,h_rel,c) where the first 2 are relative to the top-left corner of the bounding box, the height and width of the bounding box are relative to the entire image and c, represents the probability of the box containing an object (Referred to as the confidence score).
    - Each class is represented by one-hot vectors git status
    - YOLOv1 supports 20 classes so the prediction vector for each cell contains (2*5 + 20) = 30 parameters. Considering we have 7x7 cells, the entire output of the network will be a 7x7x30 matrix.
    - The output vector is in the following order : (x_rel1,y_rel1,w_rel1,h_rel1,c1) | (x_rel2,y_rel2,w_rel2,h_rel2,c2) | (p1,p2,p3,..,20)
    - The object class is calculated by: argmax(p1,p2,..,p20). Note that the property scores are numbers between 0 and 1 and may not exactly be equal to 0 or 1.
10. For post processing the output layer and getting the absolute bounding box values, we have the following calculations:
    x = x_rel*64 + x_a
    y = y_rel*64 + y_a
    w = w_rel * 448
    h = h_rel * 448
11. Conditional confidence score is calculated for both bounding boxes: c_hat_1 = c1 * p. Where p = max(p1,p2,...,p20) < 1. This parameter is based on the confidence of an object existing.
12. When analyzing the prediction matrix for each cell, only the bounding box with the higher confidence score is kept.
12. YOLO does not predict a class for every box, it predicts a class for each cell. But each cell is associated with two boxes, so those boxes will have the same predicted class, even though they may have different shapes and positions. 

# The architecture
13. The YOLOv1 architecture is inspired by the GoogleNet model.
14. The network model is consisted of two parts. The first part is called the Back bone which is 24 convolution layers which is used for generating the feature maps. The further layers are used to get the predicting which in yoloV1 has the shape 7x7x1024 = 50176.
15. In the second part, the back bone output is flattened and the passed to two fully connected layers. with their output is reshaped to a 7x7x30 matrix. The final fully connected layer has 1470 nodes (7*7*30=1470)
16. [Add architecture image]

# Loss calculation
17. The loss for each image is the sum of loss for every grid cell
18. Loss = Objectless loss + 
18. Loss function: Add the formula from 23:00 in the video 

Ref: https://www.youtube.com/watch?v=zgbPj4lSc58&list=PL1u-h-YIOL0sZJsku-vq7cUGbqDEeDK0a

Other detailed explanations on the YOLOv1 structure and its implementation are referenced below.
https://towardsdatascience.com/yolo-made-simple-interpreting-the-you-only-look-once-paper-55f72886ab73
https://wikidocs.net/167699