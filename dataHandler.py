"""
Contains the necessary function for handeling the train, test and cross-validation datasets.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Necessary for readig an image
import matplotlib.patches as patches # Necessary for drawing bounding boxes
import glob
import json
from pathlib import Path
import pandas as pd

def dispBBox(picDir, picName, labelDetail, labelsNames):
    """
    Displays the detection bounding box and the label text on an image.

    Args: 
        picDir: str: The directory where the image is saved
        picName: str: The name of the picture. We assume the file's extension is "jpg"
        labelDetail: list: A list containing the bounding box elements and other details.
            The bounding box is formatted as follows: [<top-left-x>, <top-left-y>, <width>, <height>]
            where the bounding box coordinates are expressed as relative values in [0, 1] x [0, 1].
        labelDetail: str: The location of the textfile containing the bounding box and labels for each
            detection. Each detection should be written in a seperate line containig 5 numbers with 
            the following numbers [labelNo. boxCenterX boxCenterY boxWidth boxHeight]
        labelsNames: list: A list of the labels used in detections process

    Returns: None
    """

    # Show the image
    fig, ax = plt.subplots()
    img = mpimg.imread(f"{picDir}/{picName}.jpg")
    ax.imshow(img)
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = abs(ax.get_xlim()[0]-ax.get_xlim()[1]), abs(ax.get_ylim()[0]-ax.get_ylim()[1])

    if type(labelDetail) == str:
        with open(labelDetail) as file:
            temp = file.readlines()
            for detection in temp:
                detection = detection.replace("\n","")
                detection = [float(x) for x in detection.split(" ")]

                # Note that the bounding box parameters when importing from a text file are different 
                # than that of the fiftyone library's exported json file. The bounding boxes are extracted
                # from the fiftyone's exported JSON file in a way to be compatible with YOLO-v8  algorithm
                bBox = [detection[1] - detection[3]/2, detection[2] - detection[4]/2, detection[3], detection[4]]
                ax.add_patch(patches.Rectangle((bBox[0]*width,bBox[1]*height+10),width*bBox[2],height*bBox[3], fill = None, color = "red"))
                ax.text(
                    bBox[0]*width,bBox[1]*height, labelsNames[int(detection[0])],
                    bbox = dict(facecolor='white', edgecolor='red', pad = 1), size = 7, backgroundcolor = "red"
                )
    else:
        # Display the bounding boxes
        for i in range(len(labelDetail["labels"][picName])):
            bBox = labelDetail["labels"][picName][i]["bounding_box"]
            # print(labelDetail["labels"][picName][i])
            ax.add_patch(patches.Rectangle((bBox[0]*width,bBox[1]*height+10),width*bBox[2],height*bBox[3], fill = None, color = "red"))
            ax.text(
                bBox[0]*width,bBox[1]*height, labelsNames[labelDetail["labels"][picName][i]["label"]],
                bbox = dict(facecolor='white', edgecolor='red', pad = 1), size = 7, backgroundcolor = "red"
            )
    
    return None

def annotationsToDataframe(annotDir, annotExt, annotId = None):
    """
    Reads the annotations from a directory and returns a dataframe. Annotations can be saved
    in two formats: TXT or XLS. 
        TXT: The annotations saved in text files should contain a single detection in each 
        line. Every line should be in the following order: [className rel_x rel_y width height].
        Where rel_x and rel_y are the coordinates of the center of the bounding box relative to 
        the entire picture and width and height of the box are also relative the entre picture.
        XLS: ----TODO----
    Note that it is assumed that the ID of each annotation is the file's name and there should 
    be and image file with the same exact name (And different extension) in the data directory.  

    Args:
        annotDir: str: The directory containing annotations.
        annotExt: str: The extensions of the annotations.
        annotId: str: The id of the specific image. If you want the returned dataframe to contain only 
            the annotations for a specific image. If None, the entire annotation directory will 
            be read and returned as a dataframe.

     Returns: 
        A pandas dataframe with columns: [id, boxCenterX, boxCenterY, boxWidth, boxHeight, class]       
    """
    # For performance purposes, we wont use append/concat row methods for each new entry. We append 
    # new data to lists as we iterate through the annotations. At the end we make a dataframe with 
    # the lists in hand.Temprorary lists for appending the new data.
    __lstID = []
    __lstBoxCenterX = []
    __lstBoxCenterY = []
    __lstBoxWidth = []
    __lstBoxHeight = []
    __lstClass = []

    if annotExt.lower() == "txt":
        # Read the files in the directory
        files = glob.glob(f"{annotDir}/*.txt")
        for file in files:

            # GEt the annotation ID which is the file name
            __fName = Path(file).stem
            
            # Only get a specific annotation. IF else, get all the annotations.
            if annotId != None:
                if __fName != annotId:
                    continue

            with open(file) as f:
                
                for annot in f.readlines():
                    annot = annot.replace("\n","") # Replace newline character
                    annot = annot.split(" ")

                    # Append the new data
                    __lstID.append(__fName)
                    __lstBoxCenterX.append(float(annot[1]))
                    __lstBoxCenterY.append(float(annot[2]))
                    __lstBoxWidth.append(float(annot[3]))
                    __lstBoxHeight.append(float(annot[4]))
                    __lstClass.append(int(annot[0]))
        
    elif annotExt.lower() == "txt":
        pass
    else:
        print("Invalid extension type. Only text and XML files are acceptable.")

    
    # Merge the lists to make a dataframe
    df = pd.DataFrame(
        list(zip(__lstID, __lstBoxCenterX, __lstBoxCenterY, __lstBoxWidth, __lstBoxHeight, __lstClass)),
        columns = ["id", "boxCenterX", "boxCenterY", "boxWidth", "boxHeight", "class"]   
    )

    return df