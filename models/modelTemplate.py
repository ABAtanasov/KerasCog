
import numpy as np
import theano

from keras.models import Sequential
from keras.layers import TimeDistributed, SimpleRNN, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise


# We want to import the neural network that we will be using from a global
# python script with information about all the network types
from networks import noise_recurrent, leak_recurrent, newGaussianNoise

# Worth considering:
# Perhaps we just want to make each model a class?


# ------------------------------------------------------------------
# Print model parameters:
# ------------------------------------------------------------------
def print_params():
    pass
# ------------------------------------------------------------------
# Set model parameters:
# ------------------------------------------------------------------
def set_params():
    pass

# ------------------------------------------------------------------
# Generate training data:
# ------------------------------------------------------------------
def generate_trials(params):
    pass

# ------------------------------------------------------------------
# Train!
# ------------------------------------------------------------------
def train():
    pass
