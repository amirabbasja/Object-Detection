"""
Contains the necessary function for handeling the train, test and cross-validation datasets.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Necessary for readig an image
import matplotlib.patches as patches # Necessary for drawing bounding boxes
import glob
import os
import keras
from pathlib import Path
import pandas as pd
import numpy as np
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
        
    elif annotExt.lower() == "xml":
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

class dataGenerator_YOLOv1(keras.utils.Sequence):
    """
    The dataGenerator class is used to help with the loading of training data to tensorflow model. 
    Loading the entire dataset can be memory intensive. To solve this issue, only at the beginning of
    each epoch the training data is loaded in predefined batches. As stated in tensorflow documentation, 
    using this method guarantees that the network will only train once on each sample per epoch.

    Also note that: Every Sequence must implement the __getitem__ and the __len__ methods. If you want 
    to modify your dataset between epochs you may implement on_epoch_end. The method __getitem__ should
    return a complete batch.

    Ref: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, trainImgDir, batchSize, imgSize, annotDf, nClass, shuffle):
        """
        Initializes the object.

        Args:
            trainImgDir: str: The directory which contains the training data. Each file should be saved with
            jpg extension. Also the imageId of each training sample should be the same as its file name. 
                e.g. {imageId}.gpg
            batchSize: int: The size of training samples in each batch. Preferably powers of two.
            annotDf: pd.Dataframe: A pandas directory containing all of the annotations.
            imgSize: tuple: A tuple containing training image size (width,height) in pixels. (448,448) for
                YOLOv1.
            nClass = int: Number of classes that are to be detected.
            shuffle: bool: Weather to shuffle the data at the end of each epoch. 
        """
        self.trainDir = trainImgDir
        self.imgSize = imgSize
        self.batchSize = batchSize
        self.annots = annotDf
        self.nClass = nClass
        self.shuffle = shuffle
        self.indexes = np.array([])
        self.lstImageId = [] # A list of entire training image ids

        # Search the trainDir to acquire all the image IDs
        __lst = os.listdir(trainImgDir)
        __lst = [item for item in __lst if item.endswith(".jpg")]
        __lst = [tmp.replace(".jpg", "") for tmp in __lst]
        self.lstImageId = __lst

        # Run once when object is created.
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Updates the indexes after each epoch. If self.shuffle == True, the training indexes will be shuffled.
        """
        self.indexes = np.arange(len(self.lstImageId))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """
        As stated in tensorflow documentation, should return the number of batches per epoch
        """
        return int(len(self.lstImageId) / self.batchSize)
        
    def __getitem__(self, idx):
        """
        This method generates batches. The right batch should be chosen by using the index argument.
        The logic: self.indexes contains indexes from 0 to len(self.lstImageId) at the end of each 
        epoch the index list may be shuffled. Using the index argument, tensorflow iterates through 
        batches. We select the proper chunk of self.indexes using the index argument then we fill 
        lstIDs with self.lstImageId items using the indexes we acquired.   
        """
        __indexes = self.indexes[self.batchSize * idx : self.batchSize * (idx+1)]
        lstIDs = [self.lstImageId[i] for i in __indexes]

        # Generate the batch
        x,y = self.__generateBatch(lstIDs)
        return x,y

    def __generateBatch(self, lstImg):
        """
        Generates a batch by iterating through a list of image IDs.

        Args:
            lstImg: list: A list of strings, containing image IDs.
        
        Returns:
            A batch of training and ground truth data.
        """
        x = []
        y = []

        for id in lstImg:
            img, tensor = self.__read(id)

            x.append(img)
            y.append(tensor)
        
        return np.array(x), np.array(y)


    def __read(self, ID):
        """
        Read the images and generate the ground truth tensor from annotations.
        First the image is read, resized and normalized, Then the annotations from the previously 
        acquired dataframe is used to generate the ground truth tensor.
        For YOLOv1 each image is divided to 7*7 grids and each grid cell has the following parameter
        in (order is important): [confScore, relX, relY, width, height, <classes one-hot vector>].
        where relX and relY define the center of the bounding box relative to the grid cell. width 
        and height parameters define the width and height of the bounding box relative to the 
        entire image (They are NOT relative to the bounding box to avoid acquiring numbers bigger 
        than 1). 

        #TODO Generalize it for multiple classes.

        Args: 
            ID: str: ID of the image to read

        Returns: 
            Two numpy arrays, The normalized image and it's ground truth tensor compatible with YOLOv1 
            architecture. 
        """
        imgDir =  f"{self.trainDir}/{ID}.jpg"

        # Read, resize and normalize the image
        img = Image.open(imgDir)
        img = img.resize(self.imgSize)
        img = np.array(img)/255.

        # Generate the ground truth tensor
        outTensor = np.zeros((7,7,1*5 + self.nClass))

        # Get the relevant annotations
        df = self.annots[self.annots.id == ID]

        for _, row in df.iterrows():
            # Get the absolute values for x,y,w and h
            x = row.boxCenterX * 448
            y = row.boxCenterY * 448
            w = row.boxWidth * 448
            h = row.boxHeight * 448

            # Get the x and y indexes of the grid cell
            cell_idx_i = int(x / 64) + 1
            cell_idx_j = int(y / 64) + 1

            # The top-left corner of the gridCell
            x_a = (cell_idx_i-1) * 64
            y_a = (cell_idx_j-1) * 64

            # The relative coordinates of the bounding box to the grid cell's top-left
            # corner except w and h which are relative to the entire image.
            xRel = (x-x_a) / 64
            yRel = (y-y_a) / 64
            wRel = w / 448
            hRel = h / 448

            # Change the output matrix accordingly. The target tensor/matrix should have 
            # the following properties for each grid cell: (confidenceScore|xRel|yRel|w|h|classNo)
            # where the classNo is a one-hot encoded vector.
            outTensor[cell_idx_i-1,cell_idx_j-1,:] = np.array([1, 1, xRel, yRel, wRel, hRel])
        
        return img, outTensor
        
"""
df = annotationsToDataframe(f"{os.getcwd()}/data/labels/train", "txt")
a = dataGenerator_YOLOv1(f"{os.getcwd()}/data/images/train", 1, (448,448), df, 1, True)
x,y = a.__getitem__(0)
u = np.int16(x[0,:]*255)
v = y[0,:]

fig, ax = plt.subplots()
ax.imshow(u)

gridCells = (7,7)
# Add the grid cells
if gridCells != None:
    # Adding the horizontal lines
    for i in range(gridCells[1]):
        ax.plot([0, 448], [448*(i+1)/gridCells[1],448*(i+1)/gridCells[1]], color = "blue", linewidth = 2)

    # Adding the vertical lines
    for i in range(gridCells[0]):
        ax.plot([448*(i+1)/gridCells[0],448*(i+1)/gridCells[0]], [0, 448], color = "blue", linewidth = 2)

for i in range(7):
    for j in range(7):
        print(v[i,j,:])
        if v[i,j,0] != 0:
            ax.scatter(i * 64 + v[i,j,1]*64, j * 64 + v[i,j,2]*64, c="red")
            xC = i * 64 + v[i,j,1]*64
            yC = j * 64 + v[i,j,2]*64
            ww = 448 *  v[i,j,3]
            hh = 448 *  v[i,j,4]
            ax.add_patch(patches.Rectangle((xC - ww/2,yC - hh/2),ww,hh, fill = None, color = "red"))


plt.show()
"""