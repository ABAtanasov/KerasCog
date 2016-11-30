
import numpy as np
import theano

from keras.models import Sequential
from keras.layers import TimeDistributed, SimpleRNN, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.layers.noise import GaussianNoise

from new_layers import noise_recurrent, leak_recurrent, newGaussianNoise

def set_params(nturns = 3, input_wait = 3, quiet_gap = 4, stim_dur = 3, 
                    var_delay_length = 0, stim_noise = 0, rec_noise = .1, 
                    sample_size = 256, epochs = 40, N_rec = 50):
    params = dict()
    params['nturns']          = nturns
    params['input_wait']       = input_wait
    params['quiet_gap']        = quiet_gap
    params['stim_dur']         = stim_dur
    params['var_delay_length'] = var_delay_length
    params['stim_noise']       = stim_noise
    params['rec_noise']        = rec_noise
    params['sample_size']      = sample_size
    params['epochs']           = epochs
    params['N_rec']            = N_rec

    return params


# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def generate_trials(params):
    nturns           = params['nturns']
    input_wait       = params['input_wait']
    quiet_gap        = params['quet_gap']
    stim_dur         = params['stim_dur']
    var_delay_length = params['var_delay_length']
    stim_noise       = params['stim_noise']
    sample_size      = params['sample_size']
    
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    
    turn_times   = np.zeros(sample_size)
    input_times  = np.zeros([sample_size, nturns])
    output_times = np.zeros([sample_size, nturns])
    for sample in np.arange(sample_size):
        turn_time[sample] =  stim_dur + quiet_gap + var_delay[sample]
        for i in np.arange(nturns): 
            input_times[sample, t]  = input_wait + i * turn_time[sample]
            output_times[sample, t] = input_wait + i * turn_time[sample] + stim_dur
    
    x_train = np.zeros([sample_size, seq_dur, 2])
    y_train = 0.5 * np.ones([sample_size, seq_dur, 1])
    for sample in np.arange(sample_size):
        for turn in np.arange(nturns):
            firing_neuron = np.random.randint(2)                # 0 or 1
            x_train[sample, 
                    input_times[turn] :(input_times[turn] + stim_dur), 
                    firing_neuron] 
                                        = 1
            y_train[sample, 
                    output_times[turn]:(input_times[turn] + turn_time[sample]),
                    firing_neuron]
                                        = 1

    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 2)
    params['input_times']   = input_times
    params['output_times']  = output_times
    return (x_train, y_train, params)

# This is the train function, using the Adam modified SGD method
# NOT FINISHED: Consult Dave
def train(x_train, y_train, params):
    epochs      = params['epochs']
    sample_size = params['sample_size']
    N_rec       = params['N_rec']
    rec_noise   = params['rec_noise']
    
    model = Sequential()
    
    
    # Fix this to make it recurrent:
    model.add(input_dim = 2, output_dim = N_rec, return_sequences = True, activation = 'relu', noise = 0.1)
    #model.add(leak_recurrent(input_dim=2, output_dim=N_rec, return_sequences=True, activation='relu',noise=0.1))

    # We want to add rec_noise to our network: 
    # Is this a problem??
    model.add(newGaussianNoise(rec_noise)) 
    
    model.add(TimeDistributed(Dense(output_dim=1, activation='linear')))
    
    # Note I'm not using mse, unlike Daniel's example. Try changing this if training is slow
    model.compile(loss = 'binary_crossentropy', optimizer='Adam')
    model.fit(x_train, y_train, nb_epoch=epochs, batch_size=32)
    return (model, params, x_train)







