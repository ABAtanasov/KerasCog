# A task file should define a task for a network,
# then train the network


# ------------------------------------------------------------------
# Set task parameters:
# - return parameters in a dict
# ------------------------------------------------------------------
def set_params():
    pass

# ------------------------------------------------------------------
# Generate training data:
# - return task input (x_train), task output (y_train),
#   objective function mask (mask), params
# ------------------------------------------------------------------
def generate_trials(params):
    pass

# ------------------------------------------------------------------
# Train using a model from models
# ------------------------------------------------------------------
def train():
    pass

# ------------------------------------------------------------------
# define parameters, generate trials, and train
# ------------------------------------------------------------------
if __name__ == '__main__':
    pass
