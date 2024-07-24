import tensorflow as tf

class YOLOv1_LastLayer_Reshape(tf.keras.layers.Layer):
    """
    Defines a costume layer for reshaping the last layer to YOLOv1 compatible layer.
    Note, No build function is needed.
    """
    def __init__(self, targetShape):
        """
        Initializes the layer.
        """
        super().__init__()
        self.targetShape = tuple(targetShape)
    
    def get_config(self):
        """
        Helps in serializing the layer data
        """
        config = super().get_config()
        config.update({"target_shape": self.targetShape})
        return config

    def call(self, layerInput):
        """
        Forward computations. We take the first Sx * Sy * C indexes of each the input 
        vector to resemble the class probabilities of each grid cell. The rest of the 
        Sx * Sy * B indexes resemble the confidence scores of each grid cell And the 
        rest resemble the bounding box parameters <boxCenterX, boxCenterY, width, height>  

        Args:
            layerInput: tensor: The output from a dense (fully connected) layer.
        """
        
        Sx, Sy = self.targetShape[0], self.targetShape[1] # Number of parts that each axis is divided to
        C = 1 # Number of classes
        B = 2 # Number of predicted bounding boxes per grid cell


        # Get the batch size
        batchSize = tf.keras.backend.shape(layerInput)[0]

        # Class probabilities
        classProbs = tf.keras.backend.reshape(layerInput[:,:Sx*Sy*C], (batchSize,) + (Sx,Sy,C))
        classProbs = tf.keras.backend.softmax(classProbs) # Run a softmax to choose the right class with highest prob

        # Confidence scores
        confScores = tf.keras.backend.reshape(layerInput[:,Sx*Sy*C:Sx*Sy*(C+B)], (batchSize,) + (Sx,Sy,B))
        confScores = tf.keras.backend.sigmoid(confScores) # Confidence scores should be between 0 and 1

        # Bounding boxes
        bBox = tf.keras.backend.reshape(layerInput[:,Sx*Sy*(C+B):], (batchSize,) + (Sx,Sy,B*4))
        bBox = tf.keras.backend.sigmoid(bBox) # All of the bounding box parameters are relative (Between 0 and 1)


        return tf.keras.backend.concatenate([classProbs, confScores, bBox])

# # # Define a simple model using the custom reshaper layer to test it
# # input = tf.keras.layers.Input(shape=(539,))
# # x = YOLOv1_LastLayer_Reshape((7,7,11))(input)
# # model = tf.keras.Model(inputs = input, outputs = x, name = "dummy")
# # model.compile(optimizer='adam',  loss='mse', metrics=['accuracy'])

# # xTest = np.random.randint(10, size = (1,539))
# # pred = model.predict(xTest)
# # print(pred.shape)