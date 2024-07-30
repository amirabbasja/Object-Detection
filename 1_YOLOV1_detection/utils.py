# For importing datahandler methods from the parent directory
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from  PIL import Image
import pandas as pd
import tensorflow as tf
import dataHandler as handler

def customSQRT(tensor):
    """
    Returns element-wise square root of the tensor. Only can be used for tensors that all its 
    elements are smaller than 1. We use taylor series of sqrt(1-x) = 1 - x/2 - x^2/8 - x^3/16
    to do the calculations
    
    Args: tensor: tf.Tensor: A tensorflow tensor (All elements should be smaller than 1)
    """
    __x = tf.subtract(1., tensor)
    
    return  1. - tf.divide(__x, 2.) - tf.divide(tf.pow(__x, 2),8.) - tf.divide(tf.pow(__x, 3),16.) \
        - tf.multiply(tf.divide(tf.pow(__x, 4),128.),5) - tf.multiply(tf.divide(tf.pow(__x, 5),256.),7)\
        - tf.multiply(tf.divide(tf.pow(__x, 6),1024.),21)

def customSQRT2(tensor):
    """
    Returns element-wise square root of the tensor. Only can be used for tensors that all its 
    elements are smaller than 1. We use taylor series of sqrt(0.5-x) which has been acquired
    using Wolfram Alpha through the following link:
    https://www.wolframalpha.com/widgets/view.jsp?id=f9476968629e1163bd4a3ba839d60925
    
    Args: tensor: tf.Tensor: A tensorflow tensor (All elements should be smaller than 1)
    """
    __x = tf.subtract(0.5, tensor)
    
    return  0.707107 - tf.multiply(0.707107, __x) - tf.multiply(0.353553, tf.pow(__x, 2)) - tf.multiply(0.353553, tf.pow(__x, 3)) \
        - tf.multiply(0.441942, tf.pow(__x, 4)) - tf.multiply(0.618718, tf.pow(__x, 5)) + tf.multiply(0.928078, tf.pow(__x, 6)) \
        - tf.multiply(1.45841, tf.pow(__x, 7)) - tf.multiply(2.36991, tf.pow(__x, 8)) - tf.multiply(3.94985, tf.pow(__x, 9))   

def iouUtils(boxParams, gridRatio = tf.constant(7, tf.float32)):
    """
    Given bounding box centers and its width and height, calculates top-left and bottom-right coordinates of the box.
    Note that calculations in this function are done with teh assumption of w and h being a float number, between 0 and 1
    with respect to the entire image's size. However, x and y of the bounding box's center are assumed to be a float 
    between 0 and 1, with respect to the upper-left point of the grid cell.

    Args:
        boxParams: tf.Tensor: A tensor with following information (Box center X, Box center Y, Box width, Box height) for all
            boxes in a tensor.
        gridRatio: int: The number of evenly distributed grid cells in each image axis. Use 7 for YOLOv1.
    
    Returns:
        Two tensors, one indicating top-left pint of the bBox and, the other one denoting bottom-right edge.
    """
    boxXY = boxParams[...,0:2]
    halfWH = tf.divide(boxParams[...,2:], tf.constant([2.]))

    # Top-left (X, Y) and bottom-right (X, Y)
    return tf.subtract(boxXY, halfWH * gridRatio), tf.add(boxXY, halfWH * gridRatio)

def calcIOU(predict_topLeft, predict_bottomRight, truth_topLeft, truth_bottomRight):
    """
    Calculates intersection over union for two bounding boxes.

    Args:
        predict_topLeft, predict_bottomRight: tf.Tensor: Top-left and bottom-right coordinates of the predicted box, acquired 
            by iouUtils.
        truth_topLeft, truth_bottomRight: tf.Tensor: Top-left and bottom-right coordinates of the ground truth box, acquired 
            by iouUtils.
    
    Returns:
        Intersection over union of two boxes
    """

    intersectEdgeLeft = tf.maximum(predict_topLeft, truth_topLeft)
    intersectEdgeRight = tf.minimum(predict_bottomRight, truth_bottomRight)
    
    intersectWH = tf.abs(tf.subtract(intersectEdgeLeft, intersectEdgeRight))
    intersectArea = tf.reduce_prod(intersectWH, axis = -1)

    # Get area of predicted and ground truth bounding boxes
    predArea = tf.reduce_prod(tf.abs(tf.subtract(predict_topLeft, predict_bottomRight)), axis = -1)
    truthArea = tf.reduce_prod(tf.abs(tf.subtract(truth_topLeft, truth_bottomRight)), axis = -1)

    
    # Return IOU
    return tf.divide(intersectArea, predArea + truthArea - intersectArea)

def lrScheduler(epoch, schedule, currentLR):
    """
    Returns a learning rate value with respect to epoch number.

    Args: 
        epoch: int: the current epoch number.
        schedule: list: A list of tuples of epoch number and its respective learning rate value. 
            If the epoch number of the fitting process doesn't reach the specified epoch number,
            the learning rate will remail unchanged. The entries have to be in order of epoch 
            numbers.
        currentLR: float: The learning rate of the model before starting the most recent epoch.

    Returns: float: learning rate.
    """
    
    newLR = currentLR

    for entry in schedule:
        if entry[0] == epoch:
            newLR = float(entry[1])
    
    return newLR


# #  Testing IOU code
# predict = tf.random.uniform((3,4))
# truth = tf.random.uniform((3,4))
# p1, p2 = iouUtils(predict)
# t1, t2 = iouUtils(truth)
# print(calcIOU(p1, p2, t1, t2))