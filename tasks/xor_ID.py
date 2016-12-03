import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint

from models import danielsModel.model


def set_params(seq_dur = 30, mem_gap = 4, out_gap = 3, stim_dur = 3,
                    first_in = 3, var_delay_length = 0, stim_noise = 0, rec_noise = .1, 
                    sample_size = 512, epochs = 80, nb_rec = 50):
    params = dict()
    params['first_input'] = first_in
    params['stim_dur'] = stim_dur
    params['seq_dur'] = seq_dur
    params['mem_gap'] = mem_gap
    params['out_gap'] = out_gap
    params['var_delay_length'] = var_delay_length
    params['nb_rec_neurons'] = nb_rec
    params['epochs'] = epochs
    params['sample_size'] = sample_size
    params['stim_noise'] = stim_noise
    params['rec_noise'] =rec_noise
    assert params['first_input'] + params['stim_dur'] + params['mem_gap'] + params['stim_dur'] + params['out_gap'] < params['seq_dur'], 'malos parameteres'
    return params


def generate_trials(params):
    seq_dur = params['seq_dur']
    mem_gap = params['mem_gap']
    out_gap = params['out_gap']
    stim_dur = params['stim_dur']
    first_in = params['first_input']
    var_delay_length = params['var_delay_length']
    sample_size = params['sample_size']
    stim_noise = params['stim_noise']
    xor_seed = np.array([[1, 1],[0, 1],[1, 0],[0, 0]])
    xor_y = np.array([0,1,1,0])
    if var_delay_length == 0:
        var_delay = np.zeros(sample_size)
    else:
        var_delay = np.random.randint(var_delay_length, size=sample_size) + 1
    second_in = first_in + stim_dur + mem_gap
    out_t = second_in + stim_dur + out_gap
    trial_types = np.random.randint(4, size=sample_size)
    x_train = np.zeros([sample_size, seq_dur, 2])
    y_train = 0.5 * np.ones([sample_size, seq_dur, 1])
    for ii in np.arange(sample_size):
        x_train[ii, first_in:first_in + stim_dur, 0] = xor_seed[trial_types[ii], 0]
        x_train[ii, second_in + var_delay[ii]:second_in + var_delay[ii] + stim_dur, 1] = xor_seed[trial_types[ii], 1]
        y_train[ii, out_t + var_delay[ii]:, 0] = xor_y[trial_types[ii]]

    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 2)
    params['second_input'] = second_in
    params['output_time'] = out_t
    return (x_train, y_train, params)


def train(x_train, y_train, params):
    epochs = params['epochs']
    
    model = danielsModel.model(params)
    
    checkpoint = ModelCheckpoint('../weights/xor_weights-{epoch:02d}.h5')
    
    # TODO talk to Dave if we should replace this by fit_generator
    
    
    model.fit(x_train, y_train, nb_epoch=epochs, batch_size=32, callbacks = [checkpoint])
    return (model, params, x_train)

def run_xor(model, params):
    seq_dur = params['seq_dur']
    mem_gap = params['mem_gap']
    stim_dur = params['stim_dur']
    first_in = params['first_input']
    
    stim_dur = params['stim_dur']
    stim_noise = params['stim_noise']
    
    second_in = first_in + stim_dur + mem_gap
    
    xor_seed = np.array([[1, 1],[0, 1],[1, 0],[0, 0]])
    second_in = first_in + stim_dur + mem_gap
    x_pred = np.zeros([4, seq_dur, 2])
    for jj in np.arange(4):
        x_pred[jj, first_in:first_in + stim_dur, 0] = xor_seed[jj, 0]
        x_pred[jj, first_in:first_in + stim_dur, 1] = 1-xor_seed[jj, 0]
        x_pred[jj, second_in:second_in + stim_dur, 0] = xor_seed[jj, 1]
        x_pred[jj, second_in:second_in + stim_dur, 1] = 1-xor_seed[jj, 1]

    x_pred = x_pred + stim_noise * np.random.randn(4, seq_dur, 2)
    y_pred = model.predict(x_pred)
    plt.figure(121)
    for ii in np.arange(4):
        plt.subplot(2, 2, ii + 1)
        plt.plot(y_pred[ii, :, 0])
        plt.plot(x_pred[ii, :, :])
        plt.ylim([-0.1, 1.1])
        plt.title(str(xor_seed[ii, :]))

    plt.show()
    return (x_pred, y_pred)


if __name__ == '__main__':
    params = set_params()

    trial_info = generate_trials(params)

    train_info = train(trial_info[0], trial_info[1], trial_info[2])

    run_xor(train_info[0], train_info[1])

