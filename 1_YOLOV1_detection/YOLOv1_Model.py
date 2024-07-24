import tensorflow as tf
from YOLOv1_Reshape_Layer import YOLOv1_LastLayer_Reshape
from keras.regularizers import l2 # type: ignore
from YOLOv1_Loss import YOLOv1_loss

class YOLOV1_Model():
    def __init__(self):
        pass

    def getModel(self):
        # YOLOv1 structure
        YOLOv1_inputShape = (448,448,3) # Shape of the input image 
        classNo = 1 # Number of classes we are trying to detect
        input = tf.keras.layers.Input(shape=YOLOv1_inputShape)
        leakyReLu = tf.keras.layers.LeakyReLU(negative_slope = .1)


        # The backbone, Acts ads a feature extractor
        # L1
        x = tf.keras.layers.Conv2D(filters = 64, kernel_size=7, strides = 2, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(input)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding = "same")(x)

        # L2
        x = tf.keras.layers.Conv2D(filters = 192, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding = "same")(x)

        # L3
        x = tf.keras.layers.Conv2D(filters = 128, kernel_size=1, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 256, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 256, kernel_size=1, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 512, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding = "same")(x)

        # L4
        for _ in range(4):
            x = tf.keras.layers.Conv2D(filters = 256, kernel_size=1, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
            x = tf.keras.layers.Conv2D(filters = 512, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 512, kernel_size=1, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 1024, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding = "same")(x)

        # L5
        x = tf.keras.layers.Conv2D(filters = 512, kernel_size=1, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 1024, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 512, kernel_size=1, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 1024, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 1024, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 1024, kernel_size=3, strides = 2, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)

        # L6
        x = tf.keras.layers.Conv2D(filters = 1024, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)
        x = tf.keras.layers.Conv2D(filters = 1024, kernel_size=3, strides = 1, padding = "same", activation= leakyReLu, kernel_regularizer=l2(1e-5))(x)

        # Neck
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(4096)(x)
        x = tf.keras.layers.Dense(7*7*(5*2+classNo), activation="sigmoid")(x)
        x = tf.keras.layers.Dropout(.5)(x) # Dropout layer for avoiding overfitting
        x = YOLOv1_LastLayer_Reshape((7,7,5*2+classNo))(x)
        model = tf.keras.Model(inputs = input, outputs = x, name = "YOLOv1")

        model.compile(loss = YOLOv1_loss ,optimizer = 'adam')
        
        return model