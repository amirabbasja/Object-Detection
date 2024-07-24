import keras
import tensorflow as tf


class customLearningRate(keras.callbacks.Callback):
    """
    Sets the learning rate of the fitting process with respect to the epoch number.

    Args:
        schedule: method: Using the epoch number, returns the suitable learning rate
        LR_schedule: list: A list of tuples of epoch number and its respective learning rate value. 
            If the epoch number of the fitting process doesn't reach the specified epoch number,
            the learning rate will remail unchanged. The entries have to be in order of epoch 
            numbers.
    """
    def __init__(self, scheduleFCN, LR_schedule):
        """
        Initialized the class

        Args: 
            scheduleFCN: method: A method that returns new learning rate
            LR_schedule: list: 
        """
        super(customLearningRate, self).__init__()
        self.LR_schedule = LR_schedule
        self.scheduleFCN = scheduleFCN

    def on_epoch_begin(self, epoch, logs=None):
        """
        Runs on the epoch start.

        Args:
            epoch: int: The current epoch number.
        """

        # # Check to see of the model has defined a learning rate
        # if hasattr(self.model.optimizer, "lr"):
        #     raise Exception("custom learning rate generator: First define a learning rate for the model.")
        
        # Get current learning rate
        learningRate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # Get the new learning rate
        newLearningRate = self.scheduleFCN(epoch, self.LR_schedule, learningRate)

        # Set the new learning rate as the model's learning rate
        
        self.model.optimizer.learning_rate.assign(newLearningRate)

        # Notify the user
        if learningRate != newLearningRate:
            tf.print(f"Updated the learning rate at epoch NO. {epoch}. New learning rate: {newLearningRate}")
