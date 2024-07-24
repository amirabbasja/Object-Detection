# training the network
# To check the GPU status use (if you have a NVIDIA GPU): watch -n 1 nvidia-smi

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from utils import lrScheduler
import sys, os

# Import YOLOv1-specific methods and classes
from YOLOv1_Model import YOLOV1_Model
from YOLOv1_learning_Rate import customLearningRate
from YOLOv1_Reshape_Layer import YOLOv1_LastLayer_Reshape
from YOLOv1_Loss import YOLOv1_loss

here = os.path.dirname(".")
sys.path.append(os.path.join(here, '..'))
from dataHandler import *

# Start the training process
if __name__ == "__main__":

    # See if there are any GPUs
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # See if the directory to save the checkpoints exists
    if not os.path.isdir(f"{os.getcwd()}/model_data"):
        os.mkdir(f"{os.getcwd()}/model_data")

    # Instantiate the checkpoint object
    chkPoint = ModelCheckpoint(filepath='./model_data/model_{epoch:02d}-{val_loss:.2f}.keras',save_best_only=True,monitor='val_loss',mode='min',verbose=1)

    # YOLOv1 specific parameters
    imageShape = (448,448)
    numClasses = 1

    # Training variables
    numEpochs = 135
    batch_size = 1
    LR_schedule = [(0, 0.01),(75, 0.001),(105, 0.0001),]

    dfTrain = annotationsToDataframe(f"../data/labels/train", "txt")
    trainingBatchGenerator = dataGenerator_YOLOv1(f"../data/images/train", batch_size, imageShape, dfTrain, numClasses, True)

    dfTest = annotationsToDataframe(f"../data/labels/test", "txt")
    testingBatchGenerator = dataGenerator_YOLOv1(f"../data/images/test", batch_size, imageShape, dfTrain, numClasses, True)

    model = YOLOV1_Model().getModel()

    model.fit(x=trainingBatchGenerator,
            # steps_per_epoch = int(dfTrain.shape[0] // batch_size),
            epochs = numEpochs,
            verbose = 1,
            validation_data = testingBatchGenerator,
            # validation_steps = int(len(dfTest) // batch_size),
            callbacks = [customLearningRate(lrScheduler, LR_schedule),chkPoint]
    )