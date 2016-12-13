# model from Alex's flipflop file

from keras.layers import TimeDistributed, Dense, Activation
from keras.models import Sequential
from keras.constraints import maxnorm

from backend.Networks import leak_recurrent, dense_output_with_mask


def model(params):
    model = Sequential()

    # Incorporating leakiness in the neurons
    model.add(leak_recurrent(input_dim=2, output_dim=params['N_rec'], return_sequences=True, activation='relu',
                             noise=params['rec_noise'], consume_less='mem', tau=params['tau'], dale_ratio=params['dale_ratio']))

    # Before going directly to the output, we apply a relu to the signal FIRST and THEN sum THOSE signals
    # So this is the difference between W * [x]_+ (what we want) and [W * x]_+ (what we would have gotten)
    model.add(Activation('relu'))
    
    # Output neuron
    model.add(TimeDistributed(dense_output_with_mask(output_dim=1, activation='linear', dale_ratio=.8,
                                                     input_dim=params['N_rec'])))

    # Using mse, like in Daniel's example. Training is slow, for some reason when using binary_crossentropy
    model.compile(loss = 'mse', optimizer='Adam', sample_weight_mode="temporal")

    return model
