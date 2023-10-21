import numpy as np
import Dataset

class LogisticRegression:

    def __init__(self):
        # Class constructor

    def _cross_entropy_loss(self, y_prediction, y_labels):
        loss = 0

        for y,yh in zip(y_prediction, y_labels):
            loss += -yh*np.log(y)

        return loss/y_prediction.shape[0]

    def _sigmoide(self):


    def _initialize_weights(self, variance):
        # Initially, gaussian distribution with mean = 0

    def train(self):
        # Training loop
        # Compute probabilities 
        # Compute gradient
        # Update weights
        
    def inference(self):
        # 

    def _gradient(self):
        # Compute loss function gradient

    def _update_weights(self, learning_rate=0.005):
        # Update weights according to learning rate


