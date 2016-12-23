import matplotlib.pyplot as plt
import numpy as np

from models.SimpleRecurrent import SimpleRecurrentModel
from keras.callbacks import ModelCheckpoint


def set_params(nturns = 3, input_wait = 3, quiet_gap = 4, stim_dur = 3,
                    var_delay_length = 0, stim_noise = 0, rec_noise = .1,
                    sample_size = 512, epochs = 100, N_rec = 50, dale_ratio=0.8, tau=100):
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
    params['dale_ratio']       = dale_ratio
    params['tau'] = tau

    return params


# This generates the training data for our network
# It will be a set of input_times and output_times for when we expect input
# and when the corresponding output is expected
def generate_trials(params):
    nturns           = params['nturns']
    input_wait       = params['input_wait']
    quiet_gap        = params['quiet_gap']
    stim_dur         = params['stim_dur']
    var_delay_length = params['var_delay_length']
    stim_noise       = params['stim_noise']
    sample_size      = int(params['sample_size'])
    
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size, dtype=int)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1

    input_times  = np.zeros([sample_size, nturns],dtype=np.int)
    output_times = np.zeros([sample_size, nturns],dtype=np.int)
    
    turn_time = np.zeros(sample_size, dtype=np.int)
    
    for sample in np.arange(sample_size):
        turn_time[sample] =  stim_dur + quiet_gap + var_delay[sample]
        for i in np.arange(nturns): 
            input_times[sample, i]  = input_wait + i * turn_time[sample]
            output_times[sample, i] = input_wait + i * turn_time[sample] + stim_dur
    
    seq_dur = int(max([output_times[sample, nturns-1] + quiet_gap, sample in np.arange(sample_size)]))
    
    x_train = np.zeros([sample_size, seq_dur, 2])
    y_train = 0.5 * np.ones([sample_size, seq_dur, 1])
    for sample in np.arange(sample_size):
        for turn in np.arange(nturns):
            firing_neuron = np.random.randint(2)                # 0 or 1
            x_train[sample, 
                    input_times[sample, turn]:(input_times[sample, turn] + stim_dur),
                    firing_neuron] = 1
            y_train[sample, 
                    output_times[sample, turn]:(input_times[sample, turn] + turn_time[sample]),
                    0] = firing_neuron

    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [0 if x == .5 else 1 for x in y_train[sample,:,:]]

    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 2)
    params['input_times']   = input_times
    params['output_times']  = output_times
    return (x_train, y_train, params, mask)

# This is the train function, using the Adam modified SGD method
def train(x_train, y_train, params, mask):
    epochs = params['epochs']
    
    model = SimpleRecurrentModel(params)
    
    checkpoint = ModelCheckpoint('../weights/flipflop_weights-{epoch:02d}.h5')
    
    model.fit(x_train, y_train, nb_epoch=epochs, batch_size=64, callbacks = [checkpoint], sample_weight=mask)
    return (model, params, x_train)

def run_flipflop(model, params, x_train):


    x_pred = x_train[0:4,:,:]
    y_pred = model.predict(x_train)
    
    plt.plot(x_pred[0, :, 0])
    plt.plot(x_pred[0, :, 1])
    plt.plot(y_pred[0, :, 0])
    plt.show()


if __name__ == '__main__':
    params = set_params(epochs=20, sample_size= 512, input_wait=50, stim_dur=50, quiet_gap=100, nturns=5, N_rec=50, rec_noise=0,
                        stim_noise=0.1, dale_ratio=None, tau=100)

    trial_info = generate_trials(params)

    train_info = train(trial_info[0], trial_info[1], trial_info[2], trial_info[3])

    run_flipflop(train_info[0], train_info[1], train_info[2])

