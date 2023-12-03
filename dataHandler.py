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
from PIL import Image

def dispBBox(picDir, picName, annotation, labelsNames, newSize = None, gridCells = None):
    """
    Displays the detection bounding box and the label text on an image.

    Args: 
        picDir: str: The directory where the image is saved
        picName: str: The name of the picture without extension. We assume the file's extension is "jpg".
        annotation: list: A list containing the bounding box elements and other details.
            The bounding box is formatted as follows: [<top-left-x>, <top-left-y>, <width>, <height>]
            where the bounding box coordinates are expressed as relative values in [0, 1] x [0, 1].
        annotation: str: The location of the textfile containing the bounding box and labels for each
            detection. Each detection should be written in a seperate line containig 5 numbers with 
            the following numbers [labelNo. boxCenterX boxCenterY boxWidth boxHeight]
        annotation: pd.Dataframe: A pandas dataframe containing an annotation in each row. Preferably 
            it should be returned from annotationsToDataframe method.
        labelsNames: list: A list of the labels used in detections process
        newSize: tuple: Weather to resize the image. A tuple (newWidth,newHeight) in pixels.
        gridCells: tuple: Show the gridcells on the image. Example: (grid count x, grid count y). Added 
            for debugging YOLO algorithms.

    Returns: 
        None
    """

    # Show the image
    fig, ax = plt.subplots()
    img = Image.open(f"{picDir}/{picName}.jpg")

    # Resize the image if necessary
    if newSize != None:
        img = img.resize(newSize)

    ax.imshow(img)
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = abs(ax.get_xlim()[0]-ax.get_xlim()[1]), abs(ax.get_ylim()[0]-ax.get_ylim()[1])

    if type(annotation) == str:
        # Direct annotation file
        # Note that the bounding box parameters when importing from a text file are relative to the 
        # entire image and the x and y should represent the coordinates of the center of the bounding box.

        with open(annotation) as file:
            temp = file.readlines()
            for detection in temp:
                detection = detection.replace("\n","")
                detection = [float(x) for x in detection.split(" ")]

                # matplotlib needs the bottom-left point of the bounding box to visualize it 
                bBox = [detection[1] - detection[3]/2, detection[2] - detection[4]/2, detection[3], detection[4]]
                ax.add_patch(patches.Rectangle((bBox[0]*width,bBox[1]*height+10),width*bBox[2],height*bBox[3], fill = None, color = "red"))
                ax.text(
                    bBox[0]*width,bBox[1]*height, labelsNames[int(detection[0])],
                    bbox = dict(facecolor='white', edgecolor='red', pad = 1), size = 7, backgroundcolor = "red"
                )
    elif type(annotation) == dict:
        # JSON files
        # Note that the bounding box parameters when importing from a JSON file are relative to the 
        # entire image and the x and y should represent the coordinates of the top-left of the bounding box.
        # The JSON file is output of fiftyone library.

        for i in range(len(annotation["labels"][picName])):
            bBox = annotation["labels"][picName][i]["bounding_box"]

            # matplotlib needs the bottom-left point of the bounding box to visualize it 
            ax.add_patch(patches.Rectangle((bBox[0]*width,bBox[1]*height+10),width*bBox[2],height*bBox[3], fill = None, color = "red"))
            ax.text(
                bBox[0]*width,bBox[1]*height, labelsNames[annotation["labels"][picName][i]["label"]],
                bbox = dict(facecolor='white', edgecolor='red', pad = 1), size = 7, backgroundcolor = "red"
            )
    elif type(annotation) == pd.DataFrame:
        # Pandas dataframe
        # Note that the bounding box parameters when importing from a pandas dataframe are relative to the 
        # entire image and the x and y should represent the coordinates of the center of the bounding box
        
        df = annotation[annotation.id == picName]
        for _, row in df.iterrows():

            # matplotlib needs the bottom-left point of the bounding box to visualize it 
            bBox = [row.boxCenterX - row.boxWidth/2, row.boxCenterY - row.boxHeight/2, row.boxWidth, row.boxHeight]
            ax.add_patch(patches.Rectangle((bBox[0]*width,bBox[1]*height+10),width*bBox[2],height*bBox[3], fill = None, color = "red"))
            ax.text(
                bBox[0]*width,bBox[1]*height, labelsNames[int(row.objClass)],
                bbox = dict(facecolor='white', edgecolor='red', pad = 1), size = 7, backgroundcolor = "red"
            )
    
    # Add the gridcells
    if gridCells != None:
        # Adding the horizental lines
        for i in range(gridCells[1]):
            ax.plot([0, width], [height*(i+1)/gridCells[1],height*(i+1)/gridCells[1]], color = "blue", linewidth = 2)

        # Adding the vertical lines
        for i in range(gridCells[0]):
            ax.plot([width*(i+1)/gridCells[0],width*(i+1)/gridCells[0]], [0, height], color = "blue", linewidth = 2)

        # Adding the center of the bounding box
        if type(annotation) == pd.DataFrame:
            for _, row in df.iterrows():

                # matplotlib needs the bottom-left point of the bounding box to visualize it 
                ax.scatter(row.boxCenterX * width, row.boxCenterY * height, s = 50, c = "blue")

    plt.show()
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
        A pandas dataframe with columns: [id, boxCenterX, boxCenterY, boxWidth, boxHeight, objClass]       
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
        # ----TODO----
        pass
    else:
        print("Invalid extension type. Only text and XML files are acceptable.")

    
    # Merge the lists to make a dataframe
    df = pd.DataFrame(
        list(zip(__lstID, __lstBoxCenterX, __lstBoxCenterY, __lstBoxWidth, __lstBoxHeight, __lstClass)),
        columns = ["id", "boxCenterX", "boxCenterY", "boxWidth", "boxHeight", "objClass"]   
    )

    return df