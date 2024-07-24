# Object detection

Object detection is a critical task in computer vision that involves identifying and localizing objects within an image. Unlike image classification, which assigns a single label to an entire image, object detection goes a step further by predicting bounding boxes around each object and classifying them. This dual capability allows for more detailed analysis and interaction with visual data, making object detection essential for various applications such as autonomous driving, video surveillance, and medical imaging. Modern object detection systems often employ deep learning techniques, leveraging convolutional neural networks (CNNs) to achieve high accuracy and efficiency in detecting multiple objects within a single image.

The process of object detection can be divided into two main approaches: one-stage and two-stage detectors. One-stage detectors, such as YOLO (You Only Look Once) and SSD (Single Shot MultiBox Detector), perform object localization and classification in a single pass through the network, making them faster but sometimes less accurate. Two-stage detectors, like R-CNN (Region-based Convolutional Neural Networks) and its variants, first generate region proposals and then classify these regions, offering higher accuracy at the cost of increased computational complexity. Both approaches have their unique advantages and are chosen based on the specific requirements of the application, such as the need for real-time processing or higher precision.

## This repo

The current repository is an on-going side-project of mine. In this repo I will try to implement Object detection algorithms in full detail, so hopefully it could be used as a future reference for the people who are trying to develope object detection projects or, gain a deeper knowledge in this area.

## The files

Each implemented algorithm is added in a separate directory. There is a directory that contains various *training/test/validation* data. You might see references to this directory in my code; however, I have avoided uploading it to the git repository, because it contains gigabytes of data. Alternatively, i have added the method of acquiring these data in **dataDownloader.ipynb** file.

The file **dataHandler.py** contains general methods for working with datasets and have been used extensively in lower-level directories.
