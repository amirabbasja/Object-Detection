"""
Contains the necessary function for handeling the train, test and cross-validation datasets.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Necessary for readig an image
import matplotlib.patches as patches # Necessary for drawing bounding boxes
import json


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
        # Dispaly the bounding boxes
        for i in range(len(labelDetail["labels"][picName])):
            bBox = labelDetail["labels"][picName][i]["bounding_box"]
            # print(labelDetail["labels"][picName][i])
            ax.add_patch(patches.Rectangle((bBox[0]*width,bBox[1]*height+10),width*bBox[2],height*bBox[3], fill = None, color = "red"))
            ax.text(
                bBox[0]*width,bBox[1]*height, labelsNames[labelDetail["labels"][picName][i]["label"]],
                bbox = dict(facecolor='white', edgecolor='red', pad = 1), size = 7, backgroundcolor = "red"
            )
    
    return None
