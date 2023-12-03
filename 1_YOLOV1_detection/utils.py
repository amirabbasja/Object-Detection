# For importing datahandler methods from the parent directory
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from  PIL import Image
import pandas as pd
import path
import tensorflow as tf
import dataHandler as handler

def outTensor_YOLOv1(imgPath, imgId, annots:pd.DataFrame, C, S = 7, B = 1, imgSize = (448,448)):
    """
    Segments the input image and outputs the matrix used for training the network.

    Args:
        imgPath: str: The path for the image
        imgId: str: The target image's id that also exists in the annotations dataframe.
        annots: pd.Dataframe: The annotations containing the bounding boxes and.
            labels. It is encouraged that the dataframe only contain the required image's
            annotations.
        C: int: Number of classes.
        S: int: Number of the parts that each side of the image should be divided to.
            S = 7 for YOPath1.
        B: int: Number of the bounding boxes per grid. B = 2 for YOLOv1 predictions and 
            B = 1 for generating target tensors for learning.
        imgSize: tuple: The width and height of the image. Pass "None" to avoid 
            resizing the input image. The images in YOLOv1 have the shape 448x448 
    
    Returns:
        The image as a numpy array and a numpy array with the shape S x S x (B * 5 + C).
    """

    # Open the image
    img = Image.open(imgPath)

    # Resize the image if necessary
    if imgSize != None:
        img = img.resize(imgSize)

    # Instantiate the output tensor
    out = np.zeros((S,S,B*5+C))

    # In case if the provided dataframe contains more than required data
    annots = annots[annots.id == imgId]
    
    for _, row in annots.iterrows():
        # Get the absolute values for x,y,w and h
        x = row.xCenter * 448
        y = row.yCenter * 448
        w = row.width * 448
        h = row.height * 448

        # Get the x and y indexes of the grid cell
        cell_idx_i = int(x / 64) + 1
        cell_idx_j = int(y / 64) + 1

        # The top-left corner of the gridCell
        x_a = (cell_idx_i-1) * 64
        y_a = (cell_idx_j-1) * 64

        # The relative coordinates of the bounding box to the gridcell's top-left
        # corner except w and h which are relative to the entire image.
        xRel = (x-x_a) / 64
        yRel = (y-y_a) / 64
        wRel = w / 448
        hRel = h / 448

        # Change the output matrix accordingly. The target tensor/matrix should have 
        # the following properties for each grid cell: (confidenceScore|xRel|yRel|w|h|classNo)
        # where the classNo is a one-hot encoded vector.
        print(cell_idx_i-1,cell_idx_j-1, "added")
        out[cell_idx_i-1,cell_idx_j-1,:] = np.array([1, xRel, yRel, wRel, hRel,1])
    
    return np.array(img), np.array(out)


df = handler.annotationsToDataframe("Object-Detection/data/labels/train","txt")
imgPredTensor = outTensor_YOLOv1("Object-Detection/data/images/train/00a5820192213a93.jpg", "00a5820192213a93", df, 1, B = 1)
# print(imageTensor[3,3,:])
handler.dispBBox("Object-Detection/data/images/train", "00a5820192213a93", df, ["person"], gridCells=7, newSize=(448,448))
