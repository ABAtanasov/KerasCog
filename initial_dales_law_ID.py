import numpy as np
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
#from matplotlib import pyplot as plt


def rnn_1(weights_path = None, nb_neurons = 100):

    inputs = Input(shape=(5000, 2))

    #note: want to add noise before relu in recurrent connection, do this by tweaking SimpleRNN

    rnn = myRNN(return_sequences = True, output_dim = nb_neurons, activation='relu', consume_less = 'mem', unroll=False)(inputs)

    # We may not want to incorporate outputs
    outputs = TimeDistributed(Dense(2, activation = 'sigmoid'))(rnn)

    model = Model(input=inputs, output=outputs)

    if weights_path:
        model.load_weights(weights_path)
        
    adam = Adam(lr=.0001, clipnorm = 1)
    model.compile(optimizer=adam, loss = 'binary_crossentropy', metrics=['binary_crossentropy'], sample_weight_mode='temporal')
    
    return model



def create_input_output_pair(first_fire_neuron, second_fire_neuron, delay, length):

	lo = .2
	hi = 1.0

	X = np.zeros((length, 2))
	y = np.zeros((length, 2))

	X[:500, :] = lo
	X[500:1000, first_fire_neuron] = hi
	X[500:1000, 1 - first_fire_neuron] = lo
	X[1000:1000+delay, :] = lo
	X[1000+delay:1500+delay, second_fire_neuron] = hi
	X[1000+delay:1500+delay, 1 - second_fire_neuron] = lo
	X[1500+delay:, :] = lo

	noise = np.random.normal(scale = .1, size = X.shape)
	X = X + noise

	y[500:1000, :] = lo
	y[1000+delay:1500+delay, :] = lo
	if first_fire_neuron == second_fire_neuron:
		y[1500+delay:, 0] = hi
		y[1500+delay:, 1] = lo
	else:
		y[1500+delay:, 0] = lo
		y[1500+delay:, 1] = hi

	return X, y


def generate_input_batch(batch_size, delay, length):

	#this masks the cost function
    sample_weights = np.zeros((batch_size, length))
    non_zero = range(1500+delay, length)
    sample_weights[:, non_zero] = np.ones((batch_size, (length - 1500 - delay)))

    while True:
        X = np.zeros((batch_size, length, 2))
        y = np.zeros((batch_size, length, 2))
        for i in range(batch_size):
            if i % 4 == 0:
                X[i,:,:], y[i,:,:] = create_input_output_pair(0,0,delay, length)
            elif i % 4 == 1:
                X[i,:,:], y[i,:,:] = create_input_output_pair(0,1,delay, length)
            elif i % 4 == 2:
                X[i,:,:], y[i,:,:] = create_input_output_pair(1,0,delay, length)
            else:
				X[i,:,:], y[i,:,:] = create_input_output_pair(1,1,delay, length)
                
                
        yield X,y,sample_weights
        
        
def train_rnn_1():

    model = rnn_1()

    checkpoint = ModelCheckpoint('../weights/rnn_weights_{epoch:02d}_{val_loss:.2f}.h5')

    model.fit_generator(generate_input_batch(20, 2000, 5000), samples_per_epoch=1000, nb_epoch = 20, 
		validation_data = generate_input_batch(20, 2000, 5000), nb_val_samples = 200, callbacks=[checkpoint])

    return



class myRNN(Recurrent):
    '''Fully-connected RNN where the output is to be fed back to input.
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
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        
        # Implementing Dale's law
        dales_vector = np.ones()
        for i in range(4*output_dim/5, output_dim):
            dales_vector[i] = -1
            
        diagonal_mask = np.diag(dales_vector)
        self.diag = K.variable(diagonal_mask)

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
            
        super(myRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states =  [K.random_normal(shape=(self.output_dim,), mean=0.,std=0.1)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U'.format(self.name))
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
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, self.b, self.dropout_W,
                                          input_dim, self.output_dim,
                                          timesteps)
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

        #THIS IS THE PART WE CHANGED, CHECK SIGMA

        output = self.activation(h + K.dot(prev_output * B_U, 
                            #Changed:
                                    K.dot(abs(self.U), self.diag)) f+ K.random_normal(shape=K.shape(self.b), mean=0.,std=0.1))
        return output, [output]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))
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



train_rnn_1()

