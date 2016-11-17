
import numpy as np
from keras.models import Sequential
from keras.layers import TimeDistributed, SimpleRNN, Dense
import theano
import scipy.linalg as la
from keras.layers.core import Dense
from keras.layers.recurrent import Recurrent, time_distributed_dense
from keras import backend as K
from keras import activations, initializations, regularizers
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed
from keras.engine.topology import Layer, InputSpec
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

def set_params(seq_dur = 30, mem_gap = 4, out_gap = 3, stim_dur = 3, first_in = 3, var_delay_length = 0, stim_noise = 0, sample_size = 256, epochs = 40, nb_rec = 50):
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
    assert params['first_input'] + params['stim_dur'] + params['mem_gap'] + params['stim_dur'] + params['out_gap'] < params['seq_dur'], 'malos parameteres'
    return params


def gen_xor_data(params):
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


def train_xor(x_train, y_train, params):
    epochs = params['epochs']
    sample_size = params['sample_size']
    nb_rec = params['nb_rec_neurons']
    model = Sequential()
    model.add(myRNN(input_dim=2, output_dim=nb_rec, return_sequences=True, activation='relu'))
    model.add(TimeDistributed(Dense(output_dim=1, activation='sigmoid')))
    model.compile(loss='mse', optimizer='Adam')
    model.fit(x_train, y_train, nb_epoch=epochs, batch_size=32)
    return (model, params, x_train)


def run_xor(model, params):
    seq_dur = params['seq_dur']
    mem_gap = params['mem_gap']
    stim_dur = params['stim_dur']
    first_in = params['first_input']
    second_in = params['second_input']
    stim_dur = params['stim_dur']
    stim_noise = params['stim_noise']
    xor_seed = np.array([[1, 1],[0, 1],[1, 0],[0, 0]])
    second_in = first_in + stim_dur + mem_gap
    x_pred = np.zeros([4, seq_dur, 2])
    for jj in np.arange(4):
        x_pred[jj, first_in:first_in + stim_dur, 0] = xor_seed[jj, 0]
        x_pred[jj, second_in:second_in + stim_dur, 1] = xor_seed[jj, 1]

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


def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].output, allow_input_downcast=True)
    activations = get_activations(X_batch)
    return activations


def get_weights(model):
    return model.layers[0].get_weights()


def get_maps(model, X_batch, layer = 0):
    activations = get_activations(model, layer, X_batch)
    return activations == 0


def plot_eig(model, maps, t, condition = 0, color = [0, 0, 1]):
    w = get_weights(model)
    wrec = w[1]
    bin_maps = maps.astype('int')
    bin_t = bin_maps[condition, t, :].reshape(np.shape(bin_maps)[2], 1)
    bin_mask = bin_t.dot(bin_t.T)
    eva = la.eig(wrec * bin_mask)
    plt.scatter(eva[0].real, eva[0].imag, 12, color)
    plt.xlabel('real')
    plt.ylabel('imaginary')


class myRNN(Recurrent):
    """Fully-connected RNN where the output is to be fed back to input.
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    def __init__(self, output_dim, init = 'glorot_uniform', inner_init = 'orthogonal', activation = 'tanh', W_regularizer = None, U_regularizer = None, b_regularizer = None, dropout_W = 0.0, dropout_U = 0.0, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(myRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            self.states = [K.random_normal(shape=(self.output_dim,), mean=0.0, std=0.5)]
        input_dim = input_shape[2]
        self.input_dim = input_dim
        self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, self.output_dim), name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        self.trainable_weights = [self.W, self.U, self.b]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' + 'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0], np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W, input_dim, self.output_dim, timesteps)
        else:
            return x

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]
        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W) + self.b
        output = self.activation(h + K.dot(prev_output * B_U, self.U) + K.random_normal(shape=K.shape(self.b), mean=0.0, std=0.1))
        return (output, [output])

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.0))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.0))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
         'init': self.init.__name__,
         'inner_init': self.inner_init.__name__,
         'activation': self.activation.__name__,
         'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
         'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
         'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
         'dropout_W': self.dropout_W,
         'dropout_U': self.dropout_U}
        base_config = super(myRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

