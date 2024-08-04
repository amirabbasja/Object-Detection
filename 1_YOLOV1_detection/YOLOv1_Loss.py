# YOLOv1 Loss
import tensorflow as tf
from utils import iouUtils, calcIOU, customSQRT3

def YOLOv1_loss(yTrue, yPred):
    """
    Runs in the even of loss function calculations
    
    Args:
        yTrue, yPred: tf.Tensor: The ground truth value and the predicted value, respectively

    Returns:
        The calculated loss.
    """
    lambdaNoObj = tf.constant(.5)
    lambdaCoord = tf.constant(5.)

    # Split the predictions and ground truth vectors to coordinates, confidence and class matrices
    # 1. Ground truth 
    idx1, idx2 = 1, 1 + 1
    targetClass = yTrue[...,:idx1]
    targetConf = yTrue[...,idx1:idx2]
    targetCoords = yTrue[...,idx2:]

    # 2. Prediction
    idx1, idx2 = 1, 1 + 2
    predClass = yPred[...,:idx1]
    predConf = yPred[...,idx1:idx2]
    predCoords = yPred[...,idx2:]

    # Get the best bounding boxes by calculating the IOUs
    # Note: To to do this process for the confidence scores as well, we concat each box's confidence
    # score to its bounding box coordinates and analyze them as a whole.
    predBox1 = tf.concat([tf.expand_dims(predConf[...,0],-1),predCoords[...,:4]], axis = -1)
    predBox2 = tf.concat([tf.expand_dims(predConf[...,1],-1),predCoords[...,4:]], axis = -1)

    # Get the corners of bounding boxes to calculate IOUs
    # Note, iouUtils is not coded to accept confidence scores. So we only pass the coordinates into 
    # it. 
    p1_left, p1_right = iouUtils(predBox1[...,1:]) 
    p2_left, p2_right = iouUtils(predBox2[...,1:])
    t_left, t_right = iouUtils(targetCoords) 

    # Calculate IOUs for first and second predicted bounding box
    p1_IOU = calcIOU(p1_left, p1_right, t_left, t_right)
    p2_IOU = calcIOU(p2_left, p2_right, t_left, t_right)

    # Get the cells that have objects
    maskObj = tf.cast(0 < targetConf, tf.float32)
    maskNoObj = tf.cast(0 == targetConf, tf.float32)
    
    # Get mask tensors for boxes that their first prediction has higher IOU
    mask_p1Bigger = tf.expand_dims(tf.cast(p2_IOU < p1_IOU, tf.float32),-1)
    
    # Get mask tensors for boxes that their second prediction has higher IOU
    mask_p2Bigger = tf.expand_dims(tf.cast(p1_IOU <= p2_IOU, tf.float32),-1)

    # Getting the responsible bounding box for loss calculation. 
    # Output is of shape [...,5] because we disregard the bounding box with smaller IOU
    # The first element is the confidence score of that box.
    respBox = maskObj*(mask_p1Bigger * predBox1 + mask_p2Bigger * predBox2)
    
    # Calculating the losses
    # 1. Classification loss
    classificationLoss =  tf.math.reduce_sum(tf.math.square(maskObj * tf.subtract(targetClass, predClass)))

    # 2. Confidence loss
    # Bear in mind, for the boxes with no objects, we account for the confidence loss as well. 
    # To penalize the network for high confidence scores of the cells containing no objects. The 
    # cells that have no objects, have a confidence score of 0 in the target ground truth matrix.
    # Thus, the loss is calculated as follows: SUM_All_Cells_No_OBJ((C1-0)^2 + (C2-0)^2)
    confidenceLossObj = tf.math.reduce_sum(tf.math.square(maskObj * tf.subtract(targetConf, tf.expand_dims(respBox[...,0],-1))))
    confidenceLossNoObj =  lambdaNoObj * tf.reduce_sum(maskNoObj * tf.reduce_sum(tf.square(predConf), axis = -1, keepdims = True))
    
    # 3. Localization loss
    # Bear in mind that respBox is of the shape (...,5) and targetCoords dimension is (...,4) 
    xyLoss = tf.reduce_sum(tf.square(tf.subtract(respBox[...,1:3], targetCoords[...,0:2])),-1,True)
    whLoss = tf.reduce_sum(tf.square(tf.subtract(customSQRT3(respBox[...,3:5], respBox[...,3:5], 10), customSQRT3(targetCoords[...,2:4], targetCoords[...,2:4], 10))),-1,True)
    localizationLoss = tf.reduce_sum(lambdaCoord * (xyLoss + whLoss) )
    
    # Sum all the tree types of the errors
    return classificationLoss + confidenceLossNoObj + confidenceLossObj + localizationLoss


# # Define a simple model using the custom reshaper layer to test it
# input = tf.keras.layers.Input(shape=(539,))
# x = YOLOv1_LastLayer_Reshape((7,7,11))(input)
# model = tf.keras.Model(inputs = input, outputs = x, name = "dummy")
# model.compile(optimizer='adam',  loss=testLoss, metrics=['accuracy'])

# xTest = np.random.randint(10, size = (1,539))
# pred = model.predict(xTest)
# # model.evaluate(xTest,np.expand_dims(yTruth, 0),)
# model.fit(xTest, np.expand_dims(yTruth, 0), epochs = 1)
# # metrics = model.evaluate(xTest)
# # print(pred.shape)