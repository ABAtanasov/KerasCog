
import numpy as np
import theano

from keras.models import Sequential
from keras.layers import TimeDistributed, SimpleRNN, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.layers.noise import GaussianNoise


# We want to import the neural network that we will be using from a global
# python script with information about all the network types
from networks import noise_recurrent, leak_recurrent, newGaussianNoise

# Perhaps we just want to make each model a class? 

# For now users. This way we print all the information relating to the parameters of the model
def print_params():
    pass

# This sets the parameters explicitly, by giving arguments like N_rec, etc.
def set_params():
    pass

# This will generate a set of ideal behaviour for the network under various conditions
def generate_trials(params):
    pass
    
# This will be the method used to train the network
def train():
    pass