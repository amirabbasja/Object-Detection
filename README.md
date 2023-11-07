Object detection is a technique used for identification and localization of objects in an image. Bu localizing an object in an image, we surround the said object in bounding boxes, 

One of the very important CNN architectures used in object detection is called as YOLO (You Only Look Once). Below are some informations about the YOLO structure:
1. verall, it has 24 convolutional layers, 4 max-pooling layers and 2 fully connected layers.
2. Resizez image to 448*448 before feeding it to layers.
3. The RELU activation function is usde throughout the model.
4. A 1*1 convolution is applied to reduce the channels in this model.
